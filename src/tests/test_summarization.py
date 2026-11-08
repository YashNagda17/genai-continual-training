"""Seq2seq executor and summarization weight artifacts. No downloads."""

from pathlib import Path
import io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from ..datasets.dataset_sources import (
    CLINC_DOMAINS, ClincDomainSource, ClincOOSSource, XLSumSource, bitext_rows, load_source,
)
from ..models.model_seq2seq import Seq2SeqLMModel, Seq2SeqLoRA, Seq2SeqSettings
from ..utils.contracts import (
    AdapterPlan, Objective, ScoreSpec, TokenTargets, TrainLimits, TrainingBatch,
)
from ..utils.state import Checkpoints, read_json


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    special_tokens_map = {"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}

    def __len__(self):
        return 32

    def get_vocab(self):
        return {str(index): index for index in range(32)}

    def decode(self, ids):
        pieces = {4: "a", 5: "\u0905"}
        return "".join(pieces.get(int(index), "") for index in ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row) for row in rows]

    def encode(self, text, add_special_tokens=True):
        return [3, 1]


class TinyMT5(Seq2SeqLMModel):
    model_id = "tiny-mt5"
    model_type = "mt5"
    untie_head = True


class Batch:
    def __init__(self, ids, payload, valid_tokens=4):
        self.ids, self.payload, self.valid_tokens = ids, payload, valid_tokens


def configuration():
    from transformers import MT5Config
    return MT5Config(vocab_size=32, d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=2,
                     num_decoder_layers=2, dropout_rate=0.0, feed_forward_proj="gated-gelu",
                     tie_word_embeddings=False, decoder_start_token_id=0, eos_token_id=1,
                     pad_token_id=0, initializer_factor=0.5)


def model():
    from transformers import MT5ForConditionalGeneration
    settings = Seq2SeqSettings(rank=4, alpha=8.0, learning_rate=0.05, max_source_length=8,
                               max_target_length=8, max_batch_size=4, seed=3)
    role = TinyMT5(settings, device="cpu", pretrained=False, config=configuration(),
                   tokenizer=TinyTokenizer(), model_factory=MT5ForConditionalGeneration)
    role.runtime_root = Path(tempfile.mkdtemp())
    return role


def batch(labels=None):
    source = torch.tensor([[3, 4, 5, 1], [6, 7, 8, 1]])
    payload = {"input_ids": source, "attention_mask": torch.ones_like(source)}
    if labels is not None:
        payload["labels"] = labels
    return Batch(("a", "b"), payload)


def labels():
    return torch.tensor([[3, 4, 1, -100], [6, 7, 1, -100]])


def train_once(role, objective=None, targets=None):
    item = batch(labels())
    return role.train(1, batch_factory=lambda: [TrainingBatch(item, ("current", "current"), token_targets=targets)],
                      objective=objective or Objective("per_example_mean"), hooks=None,
                      limits=TrainLimits(4, 4, 100))


class SourceConversionTests(unittest.TestCase):
    def test_clinc_converters_keep_domains_splits_and_oos_counts(self):
        names = [name for intents in CLINC_DOMAINS.values() for name in intents] + ["oos"]
        splits = {split: [(name, i) for i, name in enumerate(names)]
                  for split in ("train", "validation", "test")}
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            with patch.object(ClincOOSSource, "_read", return_value=(splits, names)), \
                    patch.object(ClincOOSSource, "download", return_value=snapshot):
                single, domains = ClincOOSSource(), ClincDomainSource()
                self.assertEqual(len(single.convert(snapshot)), 453)
                published = domains.fetch(directory)
            manifest = read_json(published / "manifest.json")
            self.assertEqual(manifest["counts"], dict.fromkeys(splits, 150))
            self.assertEqual(manifest["oos_dropped"], 3)
            self.assertEqual(manifest["task_order"], list(CLINC_DOMAINS))

    def test_xlsum_archives_keep_both_languages_and_official_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            (snapshot / "data").mkdir(parents=True)
            for language in ("english", "hindi"):
                with tarfile.open(snapshot / "data" / f"{language}_XLSum_v2.0.tar.bz2", "w:bz2") as archive:
                    for split in ("train", "val", "test"):
                        payload = (json.dumps({"id": split, "text": "article", "summary": "summary"}) + "\n").encode()
                        entry = tarfile.TarInfo(f"{language}_{split}.jsonl")
                        entry.size = len(payload)
                        archive.addfile(entry, io.BytesIO(payload))
            with patch.object(XLSumSource, "download", return_value=snapshot):
                published = XLSumSource().fetch(directory)
            dataset, rows = load_source(published, SummarizationDataset)
            self.assertEqual(len(rows), 6)
            self.assertEqual({row.task_id for row in rows}, {"english", "hindi"})
            self.assertEqual({row.split for row in rows}, {"train", "validation", "test"})
            self.assertEqual(dataset.manifest["source"]["empty_dropped"], 0)

    def test_bitext_duplicate_prompts_stay_in_one_split_without_hashes(self):
        records = [{"category": "support", "instruction": f"question {i}", "response": "reply", "intent": "help"}
                   for i in range(20)]
        rows = bitext_rows(records + records)
        for prompt in {row["prompt"] for row in rows}:
            matching = [row for row in rows if row["prompt"] == prompt]
            self.assertEqual(len({row["split"] for row in matching}), 1)
            self.assertEqual({row["group"] for row in matching}, {prompt})
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})


class SummarizationModelTests(unittest.TestCase):
    def test_planned_resume_reproduces_the_next_dropout_update(self):
        role, restored = model(), model()
        role.config.dropout_rate = restored.config.dropout_rate = 0.1
        caller_rng = torch.random.get_rng_state().clone()
        role.activate()
        self.assertTrue(torch.equal(caller_rng, torch.random.get_rng_state()))
        role.apply_plan(AdapterPlan(("decoder.2",), {"decoder.2": 2}, {"decoder.2": 0.5}, {}))
        train_once(role)
        with tempfile.TemporaryDirectory() as directory:
            role.save(directory, "resume")
            restored.load(directory, "resume")
            restored.activate()
            train_once(role)
            train_once(restored)
            for name, parameter in role.module.named_parameters():
                self.assertTrue(torch.equal(parameter, restored.module.get_parameter(name)), name)
            self.assertEqual(role.scheduler.state_dict(), restored.scheduler.state_dict())
            self.assertTrue(torch.equal(role.rng_state["cpu"], restored.rng_state["cpu"]))
            self.assertEqual(role.steps, restored.steps)
        role.release()
        restored.release()

    def test_documented_adapter_counts(self):
        encoder, decoder, rank = 8192, 11776, 16
        self.assertEqual((8 * encoder + 8 * decoder) * rank, 2_555_904)
        self.assertEqual((2 * encoder + 2 * decoder) * rank, 638_976)

    def test_load_dtype_is_never_fp16(self):
        with self.assertRaises(ValueError):
            TinyMT5(Seq2SeqSettings(optimizer_dtype="float16"), device="cpu", pretrained=False)
        role = model()
        self.assertNotEqual(role._load_dtype(), torch.float16)
        role.activate()
        self.assertNotEqual(next(role.module.parameters()).dtype, torch.float16)
        head, shared = role.module.lm_head.weight, role.module.shared.weight
        self.assertIsNot(head, shared)

    def test_example_loss_matches_separate_rows_and_ignores_padding(self):
        role = model()
        role.activate()
        role.module.eval()
        scored = role.score(batch(labels()), ScoreSpec("example_loss", "none"))
        separate = []
        source = batch(labels())
        for row, name in enumerate(source.ids):
            payload = {key: value[row:row + 1] for key, value in source.payload.items()}
            separate.append(role.score(Batch((name,), payload), ScoreSpec("example_loss", "none")).values)
        self.assertTrue(torch.allclose(scored.values, torch.cat(separate)))
        tokens = role.score(batch(labels()), ScoreSpec("token_loss", "none"))
        self.assertEqual(tokens.values.numel(), int((labels() != -100).sum()))

    def test_late_blocks_do_not_train_or_backprop_through_earlier_blocks(self):
        role = model()
        role.activate()
        role.module.eval()
        before = role._forward(batch(labels()).payload).logits.detach()
        plan = AdapterPlan(("decoder.2",), {"decoder.2": 2}, {"decoder.2": 0.5}, {"arm": "late"})
        role.apply_plan(plan)
        after = role._forward(batch(labels()).payload).logits.detach()
        self.assertTrue(torch.allclose(before, after))
        rates = {group["lr"] for group in role.optimizer.param_groups}
        self.assertIn(role.settings.learning_rate * 0.5, rates)
        seen = []
        role.module.encoder.block[0].layer[0].SelfAttention.q.register_full_backward_hook(
            lambda *_args: seen.append("encoder"))
        role.module.decoder.block[0].layer[0].SelfAttention.q.register_full_backward_hook(
            lambda *_args: seen.append("decoder.1"))
        frozen = {name: parameter.detach().clone() for name, parameter in role.module.named_parameters()
                  if "encoder" in name and name.endswith((".A", ".B"))}
        reports = train_once(role)
        self.assertTrue(reports[0].success)
        self.assertEqual(seen, [])
        for name, parameter in role.module.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]))

    def test_probe_gradients_leave_the_learner_unchanged(self):
        role = model()
        role.activate()
        before = {name: parameter.detach().clone() for name, parameter in role.module.named_parameters()}
        steps, identity = role.steps, role._weights_identity
        probe = role.fork("probe", "english-checkpoint")
        for view in ("block", "cell"):
            result = probe.score(batch(labels()), ScoreSpec("probe_gradient", "per_example_mean", parameter_view=view))
            self.assertGreater(len(result.values["groups"]), 0)
        block = probe.score(batch(labels()), ScoreSpec("probe_gradient", "per_example_mean", parameter_view="block"))
        cell = probe.score(batch(labels()), ScoreSpec("probe_gradient", "per_example_mean", parameter_view="cell"))
        self.assertTrue(torch.allclose(torch.cat(block.values["gradients"]).sum(),
                                       torch.cat(cell.values["gradients"]).sum()))
        for name, parameter in role.module.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertEqual((role.steps, role._weights_identity), (steps, identity))

    def test_full_rank_shrink_and_merged_copy_match(self):
        role = model()
        role.activate()
        train_once(role)
        role.module.eval()
        payload = batch(labels()).payload
        before = role._forward(payload).logits.detach()
        role.shrink_adapters(role.settings.rank)
        self.assertTrue(all(module.scale == 1.0 and not getattr(module, "extra_scale", None)
                            for module in role.module.modules() if isinstance(module, Seq2SeqLoRA)))
        self.assertTrue(torch.allclose(before, role._forward(payload).logits, atol=1e-4, rtol=1e-4))
        serving = role.merged_copy()
        serving.activate()
        self.assertTrue(torch.allclose(role._forward(payload).logits, serving._forward(payload).logits,
                                       atol=1e-4, rtol=1e-4))
        with self.assertRaises(ValueError):
            serving.save(Path(tempfile.mkdtemp()) / "merged", "frozen")

    def test_script_guard_blocks_a_script_and_keeps_eos(self):
        role = model()
        scores = torch.zeros(1, 32)
        role.activate()
        role._guard({"script_guard": {"block": "devanagari"}})(torch.zeros(1, 1, dtype=torch.long), scores)
        self.assertTrue(torch.isfinite(scores[0, role.tokenizer.eos_token_id]))
        self.assertEqual(scores[0, 5].item(), float("-inf"))
        source = batch()
        first = role.predict(source, {"max_new_tokens": 3, "script_guard": {"block": "devanagari"}})
        second = role.predict(source, {"max_new_tokens": 3, "script_guard": {"block": "devanagari"}})
        self.assertEqual(first["token_ids"], second["token_ids"])
        self.assertTrue(all(5 not in row for row in first["token_ids"]))

    def test_zero_weights_take_no_step(self):
        role = model()
        role.activate()
        counted = role.score(batch(labels()), ScoreSpec("token_loss", "none")).values.numel()
        zeros = torch.zeros(counted)
        reports = train_once(role, Objective("per_example_mean", token_mode="weighted"), TokenTargets(zeros, zeros))
        self.assertFalse(reports[0].success)
        self.assertEqual(reports[0].skip_reason, "no_active_tokens")
        self.assertEqual(role.steps, 0)

    def test_english_checkpoint_pool_and_hindi_export(self):
        role = model()
        role.activate()
        train_once(role)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = Checkpoints(root, role.info.identity)
            record = {"kind": "english", "writer": "english_checkpoint", "prefix": "none",
                      "passes": 1, "validation": {"english": 0.2, "hindi": 0.1}}
            first = manager.pin(role, record)
            self.assertEqual(manager.pin(role, record), first)
            other = dict(record, prefix="summarize")
            self.assertNotEqual(manager.pin(role, other), first)
            pool = {"id": "english/1", "summary": "a short summary", "baseline": [0.2, 0.1], "loss_before": 0.4}
            saved = manager.pin_pool([pool], english_checkpoint=first)
            self.assertEqual(manager.pin_pool([pool], english_checkpoint=first), saved)
            self.assertEqual(manager.pin_pool([{**pool, "summary": "different"}], english_checkpoint=first), saved)
            left, right = model(), model()
            left.load(root / first, "english")
            right.load(root / first, "english")
            left.activate()
            right.activate()
            self.assertEqual(left._weights_identity, right._weights_identity)
            self.assertEqual(left.artifact["prefix"], "none")
            manager.prune([])
            self.assertTrue((root / first / "metadata.json").is_file())
            self.assertTrue((root / saved / "manifest.json").is_file())
            role.apply_plan(AdapterPlan(("decoder.1", "decoder.2"), {}, {}, {"arm": "hindi"}))
            role.save_hindi_checkpoint(root / "hindi")
            self.assertEqual(read_json(root / "hindi" / "metadata.json")["plan_record"]["trainable"],
                             ["decoder.1", "decoder.2"])
            role.shrink_adapters(2)
            role.save_shrunk_adapter(root / "shrunk", k_energy=2, k_generation=2,
                                     test_table=[{"k": 2, "passed": True}])
            exported = read_json(root / "shrunk" / "metadata.json")
            self.assertEqual(exported["state_kind"], "inference")
            self.assertEqual(exported["record"]["k_generation"], 2)
            restored = model()
            restored.load(root / "shrunk", "inference")
            restored.activate()
            self.assertEqual(restored.settings.rank, 2)
            resume = model()
            train_once(resume)
            resume.save(root / "resume", "resume")
            steps = resume.steps
            copied = model()
            copied.load(root / "resume", "resume")
            copied.activate()
            self.assertEqual(copied.steps, steps)


def _encode(text):
    return [ord(character) % 30 + 2 for character in text] + [1]


def _write_rows(directory):
    rows = [
        {"id": "english/1", "task": "english", "language": "english", "split": "train",
         "text": "abcd", "summary": "xy"},
        {"id": "english/2", "task": "english", "language": "english", "split": "train",
         "text": "hi", "summary": "z"},
        {"id": "english/3", "task": "english", "language": "english", "split": "validation",
         "text": "valid", "summary": "v"},
        {"id": "english/4", "task": "english", "language": "english", "split": "test",
         "text": "test", "summary": "t"},
        {"id": "hindi/9", "task": "hindi", "language": "hindi", "split": "train",
         "text": "khabar", "summary": "short"},
        {"id": "english/5", "task": "english", "language": "english", "split": "train",
         "text": "   ", "summary": "gone"},
    ]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    directory.mkdir(parents=True)
    (directory / "records.jsonl").write_bytes(payload)
    manifest = {"dataset_id": "xlsum", "schema": "summarization", "source": "csebuetnlp/xlsum"}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


class SummarizationDatasetTests(unittest.TestCase):
    def _prepared(self, directory, prefix=None):
        dataset, raw = load_source(directory, SummarizationDataset)
        dataset.split(raw, {})
        spec = Seq2SeqPreprocessing(_encode, eos_id=1, pad_id=0, identity={"tokenizer": "tiny"},
                                    max_source=4, max_target=3, prefix=prefix)
        prepared, report = dataset.preprocess(dataset.views, spec)
        return dataset, prepared, report

    def test_prefix_truncation_and_batches(self):
        import json as json_module
        with tempfile.TemporaryDirectory() as directory:
            source = _write_rows(Path(directory) / "src")
            plain, _prepared, report = self._prepared(source)
            prefixed, _, _ = self._prepared(source, prefix="summarize in {language}: ")
            self.assertNotEqual(plain.fingerprint, prefixed.fingerprint)
            self.assertEqual([item["reason"] for item in report["dropped"]], ["empty_article"])
            english = {record.sample_id: record for record in plain.views["train"] if record.task_id == "english"}
            self.assertEqual(english["english/1"].inputs["ids"], _encode("abc")[:3] + [1])
            self.assertTrue(english["english/1"].inputs["source_truncated"])
            self.assertFalse(english["english/2"].target["target_truncated"])
            self.assertGreater(report["pilot"]["english"]["source_over_limit"], 0)
            self.assertIn("hindi", report["pilot"])
            plain.expose("english/train", ["english/2", "english/1"])
            plain.expose("english/pool", ["english/1"])
            self.assertEqual([record.sample_id for record in plain.training_view("english")],
                             ["english/2", "english/1"])
            self.assertEqual([record.sample_id for record in plain.pool_view()], ["english/1"])
            with self.assertRaises(ValueError):
                plain.expose("english/pool", ["english/3"])
            batch = plain.collate(plain.training_view("english"))
            self.assertTrue(bool((batch.payload["labels"][0] == -100).any()))
            self.assertEqual(batch.ids, ("english/2", "english/1"))
            sources = plain.collate_sources(plain.views["test"])
            self.assertNotIn("labels", sources.payload)
            saved = Path(directory) / "prepared"
            plain.save(saved)
            restored = SummarizationDataset()
            restored.load({"path": saved})
            self.assertEqual([record.sample_id for record in restored.pool_view()], ["english/1"])
            self.assertEqual(restored.fingerprint, plain.fingerprint)
            self.assertEqual(json.loads((saved / "manifest.json").read_text())["schema"], "summarization")


if __name__ == "__main__":
    unittest.main()
