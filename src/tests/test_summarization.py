"""Seq2seq executor and summarization weight artifacts. No downloads."""

from pathlib import Path
import io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from ..datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from ..datasets.dataset_sources import (
    XLSumSource, load_source,
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
    return MT5Config(vocab_size=32, d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=2,
                     num_decoder_layers=2, dropout_rate=0.0, feed_forward_proj="gated-gelu",
                     tie_word_embeddings=False, decoder_start_token_id=0, eos_token_id=1,
                     pad_token_id=0, initializer_factor=0.5)


def model():
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



class CheckpointBudgetTests(unittest.TestCase):
    def test_scan_counts_nested_audit_payload_external_and_staging_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "weights.pt").write_bytes(b"weights")
            (root / "nested" / "complete.json").write_bytes(b"audit")
            (root / "Desktop.INI").write_bytes(b"ignored" * 100)
            (root / "nested" / "Thumbs.db").write_bytes(b"ignored" * 100)
            manager = Checkpoints(root, "budget", disk_bytes=18, external_bytes=lambda: 4)
            manager.reserve(2)  # 12 payload/audit + 4 external + 2 staging.
            with self.assertRaisesRegex(OSError, "budget"):
                manager.reserve(3)
            (root / "nested" / "extra.json").write_bytes(b"new")
            with self.assertRaisesRegex(OSError, "budget"):
                manager.reserve(0)  # Fresh scans must see newly written files.


class PositionLossTests(unittest.TestCase):
    """Loss arithmetic only: no model loading, training loop, or optimizer updates."""

    def test_chunked_loss_and_weighted_gradients_match_torch_cross_entropy(self):
        generator = torch.Generator().manual_seed(19)
        values = torch.randn(3, 11, 7, generator=generator).transpose(1, 2)
        targets = torch.randint(11, (3, 7), generator=generator)
        targets[0, 2] = -100
        targets[2] = -100
        weights = torch.randn(3, 7, generator=generator)
        for chunk_size in (1, 5, 32):
            with self.subTest(chunk_size=chunk_size):
                role = TinyMT5(Seq2SeqSettings(logit_chunk_size=chunk_size), device="cpu", pretrained=False)
                logits = values.detach().clone().requires_grad_()
                reference = values.detach().clone().requires_grad_()
                actual = role._position_nll(logits, targets)
                expected = torch.nn.functional.cross_entropy(
                    reference.reshape(-1, 11), targets.reshape(-1),
                    ignore_index=-100, reduction="none").view_as(targets)
                actual_grad, = torch.autograd.grad((actual * weights).sum(), logits)
                expected_grad, = torch.autograd.grad((expected * weights).sum(), reference)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)
                self.assertTrue(torch.equal(actual_grad[2], torch.zeros_like(actual_grad[2])))

    def test_large_logits_and_per_example_reduction_match_torch(self):
        role = TinyMT5(Seq2SeqSettings(logit_chunk_size=2), device="cpu", pretrained=False)
        logits = torch.tensor([[[1000., -1000., 0.], [2., 3., 4.]],
                               [[-1000., 1000., 0.], [7., 8., 9.]]], requires_grad=True)
        targets = torch.tensor([[2, -100], [1, 2]])
        counts = (targets != -100).sum(-1)
        actual = role._position_nll(logits, targets)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), targets.reshape(-1), ignore_index=-100,
            reduction="none").view_as(targets)
        left, = torch.autograd.grad((actual.sum(-1) / counts).mean(), logits)
        right, = torch.autograd.grad((expected.sum(-1) / counts).mean(), logits)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(left, right)

    def test_all_padding_and_empty_sequences_have_zero_gradients(self):
        role = TinyMT5(Seq2SeqSettings(logit_chunk_size=3), device="cpu", pretrained=False)
        for length in (0, 4):
            with self.subTest(length=length):
                logits = torch.zeros(2, length, 5, requires_grad=True)
                targets = torch.full((2, length), -100)
                losses = role._position_nll(logits, targets)
                gradient, = torch.autograd.grad(losses.sum(), logits)
                self.assertEqual(losses.shape, targets.shape)
                self.assertTrue(torch.equal(losses, torch.zeros_like(losses)))
                self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))


class SummarizationModelTests(unittest.TestCase):
    def test_fused_lora_matches_equation_gradients_and_dense_norm(self):
        for extra in (False, True):
            adapter = Seq2SeqLoRA(torch.nn.Linear(7, 5), 3, 4.5, block_key="encoder.1",
                block_index=0, part="self_attention", projection="q", deterministic=True, seed=7)
            if extra:
                adapter.add_extra(2, 11)
            with torch.no_grad():
                adapter.B.normal_()
                if extra:
                    adapter.extra_B.normal_()
            inputs = torch.randn(2, 3, 7).transpose(0, 1).requires_grad_()
            reference = adapter.base(inputs) + torch.nn.functional.linear(
                torch.nn.functional.linear(inputs, adapter.A), adapter.B) * adapter.scale
            if extra:
                reference = reference + torch.nn.functional.linear(
                    torch.nn.functional.linear(inputs, adapter.extra_A), adapter.extra_B) * adapter.extra_scale
            actual = adapter(inputs)
            torch.testing.assert_close(actual, reference)
            parameters = (inputs, adapter.A, adapter.B, *((adapter.extra_A, adapter.extra_B) if extra else ()))
            expected_gradients = torch.autograd.grad(reference.square().sum(), parameters)
            actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
            for expected, observed in zip(expected_gradients, actual_gradients):
                torch.testing.assert_close(observed, expected)
            torch.testing.assert_close(adapter.delta_norm(), torch.linalg.vector_norm(adapter.delta()))

    def test_preallocated_generation_handles_mixed_eos_and_cacheless_prefixes(self):
        from types import SimpleNamespace
        role = model()
        role.activate()
        source = batch().payload
        for cached in (False, True):
            seen = []
            def forward(**kwargs):
                seen.append(kwargs['decoder_input_ids'].clone())
                chosen = ((1, 7), (8, 1), (9, 9))[len(seen) - 1]
                logits = torch.full((2, 1, 32), -100.)
                logits[torch.arange(2), 0, torch.tensor(chosen)] = 100.
                return SimpleNamespace(logits=logits, past_key_values=object() if cached else None)
            with patch.object(role.module.encoder, 'forward', return_value=None), \
                    patch.object(role.module, 'forward', side_effect=forward), torch.no_grad():
                output = role._greedy_ids(source['input_ids'], source['attention_mask'], 3, None)
            self.assertEqual(output.tolist(), [[0, 1, 0, 0], [0, 7, 1, 0]])
            self.assertEqual([item.shape[1] for item in seen], [1, 1, 1] if cached else [1, 2, 3])
        role.release()

    def test_generation_stops_only_when_all_rows_finish_and_preserves_tokens(self):
        from types import SimpleNamespace
        role = model()
        role.activate()
        source = batch().payload
        try:
            for cached in (False, True):
                for unfinished in (False, True):
                    with self.subTest(cached=cached, unfinished=unfinished):
                        calls = []
                        def forward(**kwargs):
                            calls.append(kwargs['decoder_input_ids'].shape[1])
                            step = len(calls)
                            chosen = (1 if step >= 2 else 3,
                                      1 if step >= 5 and not unfinished else 7)
                            logits = torch.full((2, 1, 32), -100.)
                            logits[torch.arange(2), 0, torch.tensor(chosen)] = 100.
                            return SimpleNamespace(logits=logits, past_key_values=object() if cached else None)
                        with patch.object(role.module.encoder, 'forward', return_value=None), \
                                patch.object(role.module, 'forward', side_effect=forward), torch.no_grad():
                            reference = role._greedy_ids(source['input_ids'], source['attention_mask'],
                                                         64, None, stop_on_eos=False)
                            self.assertEqual(len(calls), 64)
                            calls.clear()
                            actual = role._greedy_ids(source['input_ids'], source['attention_mask'], 64, None)
                        self.assertEqual(role._trim_generated(actual[:, 1:].tolist()),
                                         role._trim_generated(reference[:, 1:].tolist()))
                        self.assertEqual(len(calls), 64 if unfinished else 8)
        finally:
            role.release()

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

    def test_zero_probe_preserves_the_loaded_checkpoint_function(self):
        learner = model()
        learner.activate()
        train_once(learner)
        probe = learner.fork("probe", "english-zero-B")
        probe.set_weights(learner.get_weights())
        probe.module.eval()
        payload = batch(labels()).payload
        before = probe._forward(payload).logits.detach()
        probe.reset_probe()
        after = probe._forward(payload).logits.detach()
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)
        for name, parameter in probe.module.named_parameters():
            if name.endswith(".B"):
                self.assertTrue(torch.equal(parameter, torch.zeros_like(parameter)))
        self.assertTrue(probe._save_full_weights)
        learner.release()
        probe.release()


    def test_load_dtype_is_fp32(self):
        for name in ("float16", "bfloat16", "float32"):
            role = TinyMT5(Seq2SeqSettings(optimizer_dtype=name), device="cpu", pretrained=False)
            self.assertEqual(role._load_dtype(), torch.float32)
        role = model()
        role.activate()
        self.assertEqual(next(role.module.parameters()).dtype, torch.float32)
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
        spec = {"max_new_tokens": 3, "script_guard": {"block": "devanagari"}}
        with role._execution(), torch.no_grad():
            payload = role._payload(source)
            reference = role.module.generate(
                input_ids=payload["input_ids"], attention_mask=payload["attention_mask"],
                max_new_tokens=3, do_sample=False, use_cache=False,
                logits_processor=[role._guard(spec)])
        role.module.decoder.block[0].eval()
        modes = [module.training for module in role.module.modules()]
        first = role.predict(source, spec)
        second = role.predict(source, spec)
        self.assertEqual([module.training for module in role.module.modules()], modes)
        self.assertEqual(first["token_ids"], second["token_ids"])
        self.assertEqual(first["token_ids"], role._trim_generated(
            [row[1:] if row and row[0] == role.config.decoder_start_token_id else row
             for row in reference.tolist()]))
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

    def test_weighted_rows_exclude_inactive_rows_and_preserve_gradients(self):
        flat = torch.tensor([2., 4., 9., 6., 10., 14.], requires_grad=True)
        weights = torch.tensor([1., 0., 0., 0., 2., 1.])
        loss, total, count = model()._weighted_loss(
            flat, TokenTargets(torch.zeros_like(weights), weights), torch.tensor([2, 1, 3]))
        self.assertEqual(float(loss), 9.5)  # Active row means: 2 and 17.
        self.assertEqual(float(total), 19.)
        self.assertEqual(int(count), 2)
        loss.backward()
        torch.testing.assert_close(flat.grad, torch.tensor([.5, 0., 0., 0., .5, .25]))

    def test_positive_weight_fast_path_matches_general_loss_and_gradients(self):
        role = model()
        weights = torch.tensor([1., 2., .5, 1.5, 3.])
        for counts in (None, torch.tensor([2, 3])):
            general = torch.tensor([2., 4., 6., 8., 10.], requires_grad=True)
            positive = general.detach().clone().requires_grad_()
            expected = role._weighted_loss(general, TokenTargets(None, weights), counts)
            actual = role._weighted_loss(positive, TokenTargets(None, weights, all_positive=True), counts)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(torch.as_tensor(left), torch.as_tensor(right))
            actual[0].backward()
            expected[0].backward()
            torch.testing.assert_close(positive.grad, general.grad)

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

    def test_partial_pinned_checkpoint_is_never_published(self):
        role = model()
        role.activate()
        with tempfile.TemporaryDirectory() as directory:
            manager = Checkpoints(directory, 'tiny')
            record = {'kind': 'english', 'writer': 'english_checkpoint', 'prefix': 'plain'}
            def partial(path, *args, **kwargs):
                (Path(path) / 'weights.pt').write_bytes(b'partial')
                raise OSError('interrupted write')
            with patch.object(role, 'save', side_effect=partial), self.assertRaises(OSError):
                manager.pin(role, record)
            self.assertFalse((Path(directory) / 'pinned' / 'english' / 'plain').exists())
            published = manager.pin(role, record)
            self.assertTrue((Path(directory) / published / 'complete.json').is_file())
        role.release()


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
            self.assertEqual(batch.payload["supervised_indices"].tolist(), [0, 1, 3, 4, 5])
            sources = plain.collate_sources(plain.views["test"])
            self.assertNotIn("labels", sources.payload)
            saved = Path(directory) / "prepared"
            plain.save(saved)
            restored = SummarizationDataset()
            restored.load({"path": saved})
            self.assertEqual([record.sample_id for record in restored.pool_view()], ["english/1"])
            self.assertEqual(restored.fingerprint, plain.fingerprint)
            self.assertEqual(json.loads((saved / "manifest.json").read_text())["schema"], "summarization")


class SummarizationPipelineTests(unittest.TestCase):
    def test_probe_means_resume_with_unequal_groups_and_partial_batches(self):
        from types import SimpleNamespace
        from ..pipelines.adapter_room import AdapterRoomConfig, AdapterRoomPipeline
        config = AdapterRoomConfig(.5, analysis_batch=2)
        probe = SimpleNamespace(device=torch.device("cpu"))
        calls, saved = [], []

        def score(part, *_args):
            calls.append(tuple(part))
            mean = sum(part) / len(part)
            return {"groups": ("a", "b"), "gradients": (
                torch.tensor([mean, mean + 1]), torch.tensor([2 * mean]))}

        pipeline = AdapterRoomPipeline(config)
        pipeline._score = score
        def interrupt():
            saved.append(json.loads(json.dumps(pipeline.policy)))
            raise KeyboardInterrupt()
        pipeline._commit = interrupt
        with self.assertRaises(KeyboardInterrupt):
            pipeline._mean_gradient(tuple(range(5)), "cell", "sample", probe)

        restored = AdapterRoomPipeline(config)
        restored.policy, restored._score = saved[0], score
        restored._commit = lambda: json.dumps(restored.policy)
        means = restored._mean_gradient(tuple(range(5)), "cell", "sample", probe)
        self.assertEqual(calls, [(0, 1), (2, 3), (4,)])
        torch.testing.assert_close(means["a"], torch.tensor([2., 3.]))
        torch.testing.assert_close(means["b"], torch.tensor([4.]))

    def fixture(self):
        from ..utils.contracts import Record
        dataset = SummarizationDataset()
        dataset.manifest = {"preprocessing": {"pad_id": 0, "eos_id": 1, "identity": {"tokenizer": "tiny"}}}
        dataset.views = {}
        for split in ("train", "validation", "test"):
            rows = []
            for language in ("english", "hindi"):
                for index in range(4 if split == "train" else 2):
                    rows.append(Record(f"{language}/{split}/{index}", "tiny", "v1", str(index), language,
                                       len(rows), split, {"ids": [3, 4 + index, 1], "text": "article", "language": language},
                                       {"ids": [4, 5, 1] if index % 2 else [4, 1], "text": "a short English news summary"},
                                       6, 200))
            dataset.views[split] = tuple(rows)
        dataset.expose("english/pool", [r.sample_id for r in dataset.training_view("english")])
        return dataset

    def run_spec(self, stage, passes=1):
        from ..utils.contracts import RunSpec, ReplayBudget, ResourceBudget
        return RunSpec((stage,), 11, 2, 100, 100, 100000, passes, "repeat_admitted", "per_example_mean",
                       ReplayBudget(4, 10000, 1000000), ResourceBudget(10_000_000_000, 0), 1,
                       {"protocol": "smoke"}, {})

    def context(self, dataset, stage, output, passes=1):
        from ..utils.contracts import RunContext
        return RunContext(model(), dataset, self.run_spec(stage, passes), output)

    def english(self, directory, dataset):
        from ..pipelines.revision import RevisionConfig, RevisionPipeline, load_pool
        config = RevisionConfig("english", hindi_batch=2, analysis_batch=2,
                                generation={"max_new_tokens": 3})
        pipeline = RevisionPipeline(config, pool=dataset.pool_view())
        pipeline.run(self.context(dataset, "english", directory))
        checkpoint = Path(directory) / pipeline.policy["english_checkpoint"]
        return checkpoint, load_pool(Path(directory) / "pinned" / "pool")

    def test_metrics_and_plan_arithmetic(self):
        from ..utils.numerics import chrf, script_share, energy_rank, paired_bootstrap, cohen_kappa, interaction, top_overlap
        from ..pipelines.adapter_room import AdapterRoomConfig, AdapterRoomPipeline
        self.assertEqual(chrf("a b", "ab"), 100)
        self.assertAlmostEqual(chrf("ab", "ac", n=1), 50)
        self.assertEqual(script_share("aअ!"), 0.5)
        self.assertEqual(energy_rank([torch.tensor([3., 1.]), torch.tensor([2., 0.])], .9), 1)
        self.assertEqual(interaction(0, 3, 2, 3), -2)
        self.assertEqual(top_overlap(["a", "b"], ["b", "a"], 2), 1)
        self.assertEqual(cohen_kappa([1, 1, 2, 2], [1, 2, 2, 2]), .5)
        self.assertEqual(paired_bootstrap([1, 3], [0, 1], 20, 7), paired_bootstrap([1, 3], [0, 1], 20, 7))
        pipeline = AdapterRoomPipeline(AdapterRoomConfig(.5))
        self.assertEqual(pipeline.adapter_plan("all").identity["trainable_parameters"], 2555904)
        self.assertEqual(pipeline.adapter_plan("late").identity["trainable_parameters"], 638976)
        self.assertEqual(pipeline.adapter_plan("early_late").identity["trainable_parameters"], 638976)
        maps = {"stable": True, "blocks": {f"{stack}.{i}": (-.5 if stack == "decoder" and i >= 5 else .2)
                                               for stack in ("encoder", "decoder") for i in range(1, 9)}}
        placed = pipeline.adapter_plan("room_placed_slow", maps=maps)
        spread = pipeline.adapter_plan("room_spread", maps=maps)
        self.assertEqual(set(placed.new_rank.values()), {14})
        self.assertEqual(set(placed.lr_scale.values()), {.5})
        self.assertEqual(placed.identity["new_parameters"], 659456)
        self.assertEqual(spread.identity["new_parameters"], 638976)
        with self.assertRaises(ValueError):
            pipeline.validate_room_plans([placed, spread])
        self.assertIsNone(pipeline.adapter_plan("room_placed", maps={"stable": False}))

    def test_revision_rank_slots_filter_and_same_forward_weights(self):
        from dataclasses import replace
        from ..pipelines.revision import RevisionConfig, RevisionPipeline
        dataset = self.fixture()
        rows = tuple(replace(r, practice={"answer_text": "a short English news summary", "answer_ids": r.target["ids"],
                            "baseline": [.6] * len(r.target["ids"]), "past_baseline": [.6] * len(r.target["ids"]),
                            "loss_before": .6 if i == 0 else 2.5, "writer": "english_checkpoint"})
                     for i, r in enumerate(dataset.pool_view()))
        pipeline = RevisionPipeline(RevisionConfig("hindi", selector="loss", repair_stop=True, hindi_batch=2,
            warmup_fraction=0, analysis_batch=4), model=model(), dataset=dataset, pool=rows)
        pipeline._score = lambda *args: torch.tensor([1.4, 2.6, 2.6, 2.6])
        self.assertEqual(pipeline.rank_pool()[0], rows[0].sample_id)
        slots = pipeline.replay_slots(2)
        self.assertEqual(len({r.sample_id for r in slots}), 2)
        self.assertEqual(slots[0].sample_id, rows[0].sample_id)
        batch_ = dataset.collate((dataset.training_view("hindi")[0], rows[0]))
        targets = pipeline.token_targets(batch_, torch.tensor([1., 2., .6, 1.6]))
        self.assertTrue(torch.equal(targets.weight[:2], torch.ones(2)))
        self.assertAlmostEqual(float(targets.weight[2:].mean()), 1)
        self.assertGreater(float(targets.weight[-1]), float(targets.weight[-2]))
        mixed = dataset.collate((batch_.records[0], rows[0], rows[1]))
        packed = pipeline.token_targets(mixed, torch.tensor([1., 2., .6, 1.6, .6, 1.6, 2.6]))
        self.assertTrue(packed.all_positive)
        torch.testing.assert_close(packed.weight, torch.tensor([1., 1., 2 / 3, 4 / 3, .5, 1., 1.5]))
        worse = replace(rows[1], practice={**rows[1].practice, "answer_text": "the the the weather"})
        filtered = pipeline.filter_past((rows[0], worse, *rows[2:]))
        self.assertNotIn(worse.sample_id, {r.sample_id for r in filtered})
        pipeline.policy["active"] = [rows[0].sample_id]
        pipeline.progress = {"steps": 1}
        pipeline._generate = lambda *args: {"texts": [rows[0].practice["answer_text"]]}
        self.assertIn(rows[0].sample_id, pipeline.retire())
        self.assertNotEqual(pipeline.replay_slots(2)[0].sample_id, rows[0].sample_id)
        # The hook's all-one weights must preserve the combined per-example objective
        # even when summaries have different numbers of supervised tokens.
        plain, weighted = model(), model()
        batch_ = dataset.collate((rows[0], rows[1]))
        from ..utils.contracts import UpdateHooks
        training = TrainingBatch(batch_, ("current", "current"))
        plain.train(1, batch_factory=lambda: [training], objective=Objective("per_example_mean"), hooks=None,
                    limits=TrainLimits(1, 1, 100))
        hook = UpdateHooks(token_targets=lambda batch, nll: TokenTargets(torch.zeros_like(nll), torch.ones_like(nll)))
        weighted.train(1, batch_factory=lambda: [training], objective=Objective("per_example_mean", token_mode="weighted"),
                       hooks=hook, limits=TrainLimits(1, 1, 100))
        left, right = plain.get_weights(), weighted.get_weights()
        for name in left["parameters"]:
            torch.testing.assert_close(left["parameters"][name], right["parameters"][name])
        plain.release()
        weighted.release()

    def test_english_hindi_maps_and_shrink_smoke(self):
        from ..pipelines.revision import RevisionConfig, RevisionPipeline
        from ..pipelines.adapter_room import AdapterRoomConfig, AdapterRoomPipeline
        from ..pipelines.generation_shrink import ShrinkConfig, GenerationShrinkPipeline
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = Path(directory), self.fixture()
            english, pool = self.english(root / "english", dataset)
            revision = RevisionPipeline(RevisionConfig("hindi", selector="loss", hindi_batch=2, analysis_batch=2,
                generation={"max_new_tokens": 3}), pool=pool, english_checkpoint=english)
            result = revision.run(self.context(dataset, "hindi", root / "hindi"))
            self.assertEqual(result.counters["steps"], 2)
            self.assertEqual(len(result.metrics), 8)
            self.assertEqual(result.counters["attempts"], 2)
            hindi = root / "hindi" / "hindi"
            maps = AdapterRoomPipeline(AdapterRoomConfig(-1, sample_size=1, analysis_batch=1, encoder_blocks=2, decoder_blocks=2),
                pool=pool, hindi=dataset.training_view("hindi"), english_checkpoint=english, hindi_checkpoint=hindi)
            maps.run(self.context(dataset, "analysis", root / "maps"))
            self.assertEqual(len(maps.policy["result"]["samples"]), 2)
            self.assertEqual(len(maps.policy["result"]["blocks"]), 4)
            shrink = GenerationShrinkPipeline(ShrinkConfig(4, tolerance=100, grid=(2, 1), analysis_batch=2,
                generation={"max_new_tokens": 3}), checkpoint=hindi)
            shrink.run(self.context(dataset, "analysis", root / "shrink"))
            self.assertEqual(shrink.policy["result"]["k_G"], 1)
            self.assertTrue((root / "shrink" / "shrunk-adapter" / "weights.pt").is_file())
            with self.assertRaises(ValueError):
                GenerationShrinkPipeline(ShrinkConfig(4), validation={
                    "english": dataset.evaluation_view("english"), "hindi": dataset.evaluation_view("hindi")})._views()

    def test_resume_rank_and_pending_replay_match_uninterrupted(self):
        from dataclasses import replace
        from ..pipelines.revision import RevisionConfig, RevisionPipeline
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = Path(directory), self.fixture()
            english, pool = self.english(root / "english", dataset)
            config = RevisionConfig("hindi", selector="loss", hindi_batch=2, analysis_batch=1,
                                    generation={"max_new_tokens": 3})
            baseline = RevisionPipeline(config, pool=pool, english_checkpoint=english)
            baseline.run(self.context(dataset, "hindi", root / "baseline"))
            interrupted = RevisionPipeline(config, pool=pool, english_checkpoint=english)
            score, calls = interrupted._score, []
            def fail_after_committed_batch(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return score(*args, **kwargs)
            interrupted._score = fail_after_committed_batch
            with self.assertRaises(KeyboardInterrupt):
                interrupted.run(self.context(dataset, "hindi", root / "resumed"))
            restored = RevisionPipeline(config, pool=pool, english_checkpoint=english)
            restored.run(self.context(dataset, "hindi", root / "resumed"))
            self.assertEqual(restored.counters, baseline.counters)
            self.assertEqual(restored.policy["ranking"], baseline.policy["ranking"])
            events = lambda pipeline: [event["ids"] for event in pipeline.events if event["kind"] == "learner_update"]
            self.assertEqual(events(restored), events(baseline))
            left, right = restored.model.get_weights(), baseline.model.get_weights()
            for name in left["parameters"]:
                self.assertTrue(torch.equal(left["parameters"][name], right["parameters"][name]), name)
            restored.model.release()
            baseline.model.release()


    def test_writing_and_stop_regeneration_resume_at_batch_boundaries(self):
        from ..pipelines.revision import RevisionConfig, RevisionPipeline, load_pool
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = Path(directory), self.fixture()
            english, pool = self.english(root / "english", dataset)
            for category in ("pool_generation", "stop_regeneration"):
                stage = "english" if category == "pool_generation" else "hindi"
                config = RevisionConfig(stage, selector="random" if stage == "english" else "loss",
                    repair_stop=stage == "hindi", warmup_fraction=0, refresh_steps=1,
                    top_fraction=1, hindi_batch=2, analysis_batch=1, generation={"max_new_tokens": 3})
                arguments = {"pool": dataset.pool_view()} if stage == "english" else {
                    "pool": pool, "english_checkpoint": english}
                baseline = RevisionPipeline(config, **arguments)
                baseline.run(self.context(dataset, stage, root / (category + "-baseline")))
                interrupted = RevisionPipeline(config, **arguments)
                generate, calls = interrupted._generate, []
                def fail_in_second_batch(records, event_category, *args, **kwargs):
                    if event_category == category:
                        calls.append(1)
                        if len(calls) == 2:
                            raise KeyboardInterrupt()
                    return generate(records, event_category, *args, **kwargs)
                interrupted._generate = fail_in_second_batch
                output = root / (category + "-resume")
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.run(self.context(dataset, stage, output))
                restored = RevisionPipeline(config, **arguments)
                restored.run(self.context(dataset, stage, output))
                self.assertEqual(restored.counters, baseline.counters)
                self.assertEqual(restored.policy["retired"], baseline.policy["retired"])
                if stage == "english":
                    a = load_pool(output / "pinned" / "pool")
                    b = load_pool(root / (category + "-baseline") / "pinned" / "pool")
                    self.assertEqual([r.practice for r in a], [r.practice for r in b])
                else:
                    a, b = restored.model.get_weights(), baseline.model.get_weights()
                    for name in a["parameters"]:
                        self.assertTrue(torch.equal(a["parameters"][name], b["parameters"][name]), name)
                restored.model.release()
                baseline.model.release()

    def test_shrink_search_stops_downward_and_recovers_upward(self):
        from ..pipelines.generation_shrink import ShrinkConfig, GenerationShrinkPipeline
        with tempfile.TemporaryDirectory() as directory:
            root, dataset = Path(directory), self.fixture()
            source = model()
            train_once(source)
            source.save_hindi_checkpoint(root / "source")
            source.release()
            for direction in ("down", "up"):
                shrink = GenerationShrinkPipeline(ShrinkConfig(4, grid=(3, 1), tolerance=1), checkpoint=root / "source")
                seen = []
                def scores(candidate):
                    rank = candidate.settings.rank
                    seen.append(rank)
                    passed = rank >= (2 if direction == "down" else 3)
                    return {lang: dict.fromkeys(("rouge_l", "chrf", "right_language"), 100. if passed else 50.)
                            for lang in ("english", "hindi")}
                shrink._evaluate_rank = scores
                with patch("src.pipelines.generation_shrink.energy_rank", return_value=2):
                    shrink.run(self.context(dataset, "analysis", root / direction))
                self.assertEqual(seen, [4, 2, 1] if direction == "down" else [4, 2, 3])
                self.assertEqual(shrink.policy["result"]["k_G"], 2 if direction == "down" else 3)
                shrink.model.release()


if __name__ == "__main__":
    unittest.main()
