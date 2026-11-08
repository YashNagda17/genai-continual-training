"""Generation-aware replay checks for dataset room, prefix collation, and causal-LM token losses."""

import tempfile
import unittest

import torch
from torch.nn import functional as F
from transformers import Lfm2Config, Lfm2ForCausalLM, LlamaConfig, LlamaForCausalLM

from ..datasets.dataset_instruction import InstructionDataset, InstructionPreprocessing
from ..models.model_causal_lm import CausalLMModel, CausalLMSettings
from ..utils.contracts import (
    Batch, Objective, Record, ScoreSpec, StepReport, TokenTargets, TrainLimits, TrainingBatch,
)
from ..utils.numerics import rouge_l, token_similarity
from ..utils.runtime import PipelineRuntime


def _record(sample_id, prompt, response, split="train"):
    return Record(sample_id, "toy", "v1", sample_id, "meetings", 0, split,
                  {"messages": [{"role": "user", "content": prompt}]},
                  {"text": response}, 0, 1)


def _spec(**kwargs):
    fields = {"encode": lambda text: [ord(c) for c in text],
              "render": lambda messages: messages[-1]["content"],
              "eos_id": 1, "pad_id": 0, "max_length": 8, "identity": {"tokenizer": "chars"},
              "truncation": "prompt_left"}
    fields.update(kwargs)
    return InstructionPreprocessing(**fields)


def _prepare(records, **kwargs):
    dataset = InstructionDataset()
    dataset.manifest = {"dataset_id": "toy", "revision": "v1"}
    views = {name: tuple(r for r in records if r.split == name)
             for name in ("train", "validation", "test")}
    prepared, report = dataset.preprocess(views, _spec(**kwargs))
    return dataset, prepared, report


class InstructionReplayTests(unittest.TestCase):
    def test_reserve_zero_keeps_the_prompt_plus_gold_bound(self):
        # "ab" plus EOS is 3 tokens, so a 10-token prompt keeps its last 5.
        _, prepared, report = _prepare((_record("m1", "0123456789", "ab"),))
        row = prepared["train"][0]
        self.assertEqual(row.inputs["prompt_ids"], [ord(c) for c in "56789"])
        self.assertEqual(len(row.inputs["prompt_ids"]) + len(row.target["ids"]), 8)
        self.assertEqual(report["truncated"], ["m1"])

    def test_reserve_leaves_room_to_generate(self):
        # Gold is 3 tokens and reserve is 6, so the prompt keeps 2 tokens.
        reserved, prepared, report = _prepare((_record("m1", "0123456789", "ab"),), reserve_response=6)
        row = prepared["train"][0]
        self.assertEqual(row.inputs["prompt_ids"], [ord(c) for c in "89"])
        self.assertLessEqual(len(row.inputs["prompt_ids"]), 8 - 6)
        self.assertEqual(report["truncated"], ["m1"])
        bare, bare_rows, _ = _prepare((_record("m1", "0123456789", "ab"),))
        self.assertNotEqual(reserved.fingerprint, bare.fingerprint)
        # A reserve shorter than the gold response uses the gold bound.
        smaller, smaller_rows, _ = _prepare((_record("m1", "0123456789", "ab"),), reserve_response=2)
        self.assertEqual(smaller_rows["train"][0].inputs["prompt_ids"], bare_rows["train"][0].inputs["prompt_ids"])
        self.assertNotEqual(smaller.fingerprint, bare.fingerprint)
        # A reserve equal to max_length leaves an empty prompt, which still fits.
        _, empty_rows, empty_report = _prepare((_record("m1", "hi", "ab"),), reserve_response=8)
        self.assertEqual(empty_rows["train"][0].inputs["prompt_ids"], [])
        self.assertEqual(empty_report["dropped"], [])
        self.assertEqual(empty_report["truncated"], ["m1"])

    def test_reserve_drops_a_prompt_it_cannot_fit(self):
        _, _, too_large = _prepare((_record("m1", "hi", "ab"),), reserve_response=9, truncation="drop")
        self.assertEqual(too_large["dropped"], [{"sample_id": "m1", "reason": "reserve_exceeds_max_length"}])
        _, _, long_prompt = _prepare(
            (_record("m1", "0123456789", "ab"),), reserve_response=6, truncation="drop")
        self.assertEqual(long_prompt["dropped"], [{"sample_id": "m1", "reason": "exceeds_max_length"}])
        _, kept, short = _prepare((_record("m1", "hi", "ab"),), reserve_response=6, truncation="drop")
        self.assertEqual(short["dropped"], [])
        self.assertEqual(len(kept["train"][0].inputs["prompt_ids"]), 2)
        _, _, full = _prepare((_record("m1", "hi", "abcdefg"),))
        self.assertEqual(full["dropped"], [{"sample_id": "m1", "reason": "response_exceeds_max_length"}])
        with self.assertRaises(ValueError):
            _prepare((_record("m1", "hi", "ab"),), reserve_response=-1)

    def test_collate_prefixes_preserves_order_and_left_padding(self):
        dataset, prepared, _ = _prepare((
            _record("m1", "abc", "xy"),
            _record("m2", "z", "qrs"),
        ))
        first, second = prepared["train"]
        batch = dataset.collate_prefixes((first, first, second), (1, 0, 3))
        self.assertEqual(batch.ids, ("m1", "m1", "m2"))
        self.assertNotIn("labels", batch.payload)
        self.assertEqual(batch.supervised_tokens, 0)
        sequences = [first.inputs["prompt_ids"] + first.target["ids"][:1],
                     first.inputs["prompt_ids"],
                     second.inputs["prompt_ids"] + second.target["ids"][:3]]
        width = max(len(seq) for seq in sequences)
        padded = [[0] * (width - len(seq)) + seq for seq in sequences]
        masks = [[False] * (width - len(seq)) + [True] * len(seq) for seq in sequences]
        self.assertEqual(batch.payload["input_ids"].tolist(), padded)
        self.assertEqual(batch.payload["attention_mask"].tolist(), masks)
        self.assertEqual(batch.valid_tokens, sum(len(seq) for seq in sequences))
        with self.assertRaises(ValueError):
            dataset.collate_prefixes((first,), (len(first.target["ids"]) + 1,))
        with self.assertRaises(ValueError):
            dataset.collate_prefixes((first, first), (0,))
        with self.assertRaises(ValueError):
            dataset.collate_prefixes((first,), (-1,))

    def test_saved_manifest_keeps_the_reserve(self):
        dataset, prepared, _ = _prepare((_record("m1", "hi", "ab"),), reserve_response=4)
        with tempfile.TemporaryDirectory() as tmp:
            dataset.save(tmp)
            restored = InstructionDataset()
            views, manifest = restored.load({"path": tmp})
        self.assertEqual(restored.fingerprint, dataset.fingerprint)
        self.assertEqual(manifest["preprocessing"]["reserve_response"], 4)
        self.assertEqual(views["train"][0].inputs["prompt_ids"], prepared["train"][0].inputs["prompt_ids"])


class _Words:
    """Tiny tokenizer. Token 4, 5 and 9 decode to words so text metrics are hand-checkable."""
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "plain"
    special_tokens_map = {"eos_token": "</s>", "pad_token": "<pad>"}
    words = {4: "the", 5: "cat", 9: "sat"}

    def get_vocab(self):
        return {str(i): i for i in range(32)}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 29 + 1 for c in text] or [1]

    def batch_decode(self, rows, skip_special_tokens=True):
        texts = []
        for row in rows:
            pieces = []
            for token in row:
                if skip_special_tokens and token in (self.eos_token_id, self.pad_token_id):
                    continue
                pieces.append(self.words.get(token, "x"))
            texts.append(" ".join(pieces))
        return texts

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[-1]["content"]


class _LlamaTiny(CausalLMModel):
    model_id = "tiny-llama"
    model_type = "llama"


class _LfmTiny(CausalLMModel):
    model_id = "tiny-lfm2"
    model_type = "lfm2"


def _settings():
    return CausalLMSettings(mode="full", learning_rate=1e-2, weight_decay=0.2, max_length=16,
                            max_batch_size=4, logit_chunk_size=2, seed=3,
                            max_score_bytes=8_000_000, working_bytes=8_000_000)


def _config(kind):
    if kind == "llama":
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                             attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    else:
        config = Lfm2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                            full_attn_idxs=[0], bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    return config


def _factory(kind):
    if kind == "llama":
        return LlamaForCausalLM
    return Lfm2ForCausalLM


def _model(kind="llama"):
    cls = _LlamaTiny if kind == "llama" else _LfmTiny
    model = cls(_settings(), device="cpu", revision="local", pretrained=False, config=_config(kind),
                tokenizer=_Words(), model_factory=_factory(kind))
    model.activate()
    model.module.eval()
    return model


def _row(sample_id, token_ids, label_ids):
    return Record(sample_id, "toy", "v1", sample_id, "meetings", 0, "train",
                  {"messages": [{"role": "user", "content": "hi"}], "prompt_ids": token_ids},
                  {"text": "the cat", "ids": label_ids}, len(token_ids), 1)


def _batch(rows):
    """Right-pad token rows. labels use -100 for prompt and padding positions."""
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), _Words.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    records, supervised = [], 0
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(labs)] = torch.tensor(labs)
        mask[i, :len(ids)] = True
        supervised += int((labels[i, 1:] != -100).sum())
        records.append(_row(f"r{i}", ids, labs))
    payload = {"input_ids": input_ids, "attention_mask": mask, "labels": labels}
    return Batch(tuple(records), payload, int(mask.sum()), supervised)


def _supervised_counts(batch):
    labels = batch.payload["labels"][:, 1:]
    return [int((row != -100).sum()) for row in labels]


def _capture(model):
    params = {name: value.detach().clone() for name, value in model.module.named_parameters()}
    state = {}
    for name, value in model.module.named_parameters():
        slot = model.optimizer.state.get(value)
        if slot:
            state[name] = {key: item.detach().clone() if torch.is_tensor(item) else item
                           for key, item in slot.items()}
    return params, state, model.steps


class CausalLMReplayTests(unittest.TestCase):
    def test_text_metrics_match_the_hand_values(self):
        self.assertEqual(token_similarity([], []), 1.0)
        self.assertAlmostEqual(token_similarity([1, 2, 3], [1, 2, 4]), 2. / 3)
        self.assertAlmostEqual(rouge_l("the cat sat", "the cat"), 0.8)
        self.assertAlmostEqual(rouge_l("cat the", "the cat"), 0.5)
        report = StepReport(("a",), False, 0, 1, {"supervised": 0.}, {"supervised": 0},
                            {"token_surprise": [1.25], "class_gradients": [1], "freeze_prefix": 0},
                            skip_reason="no_active_tokens")
        saved = PipelineRuntime.report_summary(report)
        self.assertNotIn("token_surprise", saved["diagnostics"])
        self.assertNotIn("class_gradients", saved["diagnostics"])
        self.assertEqual(saved["diagnostics"]["freeze_prefix"], 0)

    def test_token_loss_matches_the_response_mean(self):
        model = _model()
        batch = _batch([([1, 4, 5, 6, 7, 8], [-100, -100, 5, 6, 7, 8]),
                        ([1, 3, 4, 5], [-100, -100, 4, 5])])
        with torch.no_grad():
            features = model._features(model._payload(batch))
            logits = model.module.lm_head(features[:, :-1]).float()
            targets = batch.payload["labels"][:, 1:]
            valid = targets != -100
            manual = F.cross_entropy(logits[valid], targets[valid], reduction="none")
        scored = model.score(batch, ScoreSpec("token_loss", "none")).values
        loss = model.score(batch, ScoreSpec("loss", "per_example_mean")).values
        self.assertTrue(torch.allclose(scored, manual, atol=1e-5))
        offset = 0
        for index, count in enumerate(_supervised_counts(batch)):
            self.assertAlmostEqual(loss[index].item(), scored[offset:offset + count].mean().item(), places=5)
            offset += count
        self.assertEqual(offset, scored.numel())
        self.assertTrue({"generation", "token_loss", "prompt_embedding"} <= model.info.capabilities)

    def test_truncated_rows_keep_the_same_surprise(self):
        full = ([1, 4, 5, 6, 7, 8], [-100, -100, 5, 6, 7, 8])
        short = ([1, 4, 5, 6], [-100, -100, 5, 6])
        for kind in ("llama", "lfm2"):
            model = _model(kind)
            alone = model.score(_batch([full]), ScoreSpec("token_loss", "none")).values
            prefix = model.score(_batch([short]), ScoreSpec("token_loss", "none")).values
            together = model.score(_batch([full, short]), ScoreSpec("token_loss", "none")).values
            self.assertGreater(alone.abs().max().item(), 0)
            self.assertTrue(torch.allclose(alone[:prefix.numel()], prefix, atol=1e-4, rtol=1e-4))
            self.assertTrue(torch.allclose(together[:prefix.numel()], together[alone.numel():], atol=1e-4, rtol=1e-4))

    def test_prompt_embedding_uses_attended_tokens_only(self):
        model = _model()
        short = _batch([([1, 4, 5], [-100, 4, 5])])
        bare = Batch(short.records, {key: value for key, value in short.payload.items() if key != "labels"},
                     short.valid_tokens, 0)
        wide_ids = torch.tensor([[1, 4, 5, 0, 0]])
        wide_mask = torch.tensor([[True, True, True, False, False]])
        wide = Batch(short.records, {"input_ids": wide_ids, "attention_mask": wide_mask}, 3, 0)
        left = model.score(bare, ScoreSpec("prompt_embedding", "none")).values
        right = model.score(wide, ScoreSpec("prompt_embedding", "none")).values
        self.assertEqual(tuple(left.shape), (1, model.config.hidden_size))
        self.assertTrue(torch.allclose(left.norm(dim=-1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(left, right, atol=1e-4, rtol=1e-4))
        with self.assertRaises(ValueError):
            model.score(short, ScoreSpec("prompt_embedding", "none"))

    def test_token_mode_all_keeps_the_supervised_step(self):
        model = _model()
        batch = _batch([([1, 4, 5, 6], [-100, -100, 5, 6])])
        training = TrainingBatch(batch, ("current",))
        report = model.train(1, batch_factory=lambda: iter((training,)), objective=Objective("per_token_mean"),
                             hooks=None, limits=TrainLimits(1, 1, batch.valid_tokens))[0]
        self.assertTrue(report.success)
        self.assertEqual(model.steps, 1)
        self.assertNotIn("token_surprise", report.diagnostics)
        rejected = TrainingBatch(batch, ("current",), token_targets=TokenTargets(
            torch.zeros(batch.supervised_tokens), torch.ones(batch.supervised_tokens)))
        with self.assertRaises(ValueError):
            model.train(1, batch_factory=lambda: iter((rejected,)), objective=Objective("per_token_mean"),
                        hooks=None, limits=TrainLimits(1, 1, batch.valid_tokens))

    def test_weighted_and_hinge_losses_and_the_inactive_skip(self):
        model = _model()
        batch = _batch([([1, 4, 5, 6, 7, 8], [-100, -100, 5, 6, 7, 8])])
        count = batch.supervised_tokens
        weight = torch.tensor([1., 0., 2., 1.5], dtype=torch.float32)
        baseline = torch.zeros(count, dtype=torch.float32)
        self.assertEqual(count, 4)
        seen = {}
        real_update = model.optimizer_update

        def record_update(loss, revision_context=None):
            seen["context"] = revision_context
            return real_update(loss, revision_context=revision_context)

        model.optimizer_update = record_update
        report = self._step(model, batch, "weighted", weight, baseline, allowance=0.0)
        surprise = report.diagnostics["token_surprise"]
        numer = sum(w * s for w, s in zip(weight.tolist(), surprise))
        denom = sum(w > 0 for w in weight.tolist())
        self.assertEqual(report.loss_counts["supervised"], denom)
        self.assertAlmostEqual(report.loss_sums["supervised"], numer, places=5)
        self.assertTrue(report.success)
        self.assertEqual(seen["context"]["objective"]["token_mode"], "weighted")
        self.assertTrue(model.optimizer.state)
        params, state, steps = _capture(model)
        for mode, weights, base, allowance in (
                ("weighted", torch.zeros(count), baseline, 0.0),
                ("hinge", torch.ones(count), torch.full((count,), 1e6), 0.1)):
            skipped = self._step(model, batch, mode, weights, base, allowance)
            self.assertFalse(skipped.success)
            self.assertEqual(skipped.skip_reason, "no_active_tokens")
            self.assertEqual(len(skipped.diagnostics["token_surprise"]), count)
            self.assertEqual(_capture(model)[2], steps)
            for name, value in model.module.named_parameters():
                self.assertTrue(torch.equal(value, params[name]))
                self.assertIsNone(value.grad)
            for name, slot in state.items():
                live = model.optimizer.state[dict(model.module.named_parameters())[name]]
                for key, item in slot.items():
                    current = live[key]
                    if torch.is_tensor(item):
                        self.assertTrue(torch.equal(current, item))
                    else:
                        self.assertEqual(current, item)
        hinge_weight = torch.tensor([1., 2., 3., 0.], dtype=torch.float32)
        hinge_base = torch.tensor([0., 0., 100., 0.], dtype=torch.float32)
        hinge = self._step(model, batch, "hinge", hinge_weight, hinge_base, allowance=0.1)
        expected_num, expected_den = 0.0, 0
        for score, w, b in zip(hinge.diagnostics["token_surprise"], hinge_weight.tolist(), hinge_base.tolist()):
            excess = score - b - 0.1
            if w > 0 and excess > 0:
                expected_num += w * excess
                expected_den += 1
        self.assertGreater(expected_den, 0)
        self.assertEqual(hinge.loss_counts["supervised"], expected_den)
        self.assertAlmostEqual(hinge.loss_sums["supervised"], expected_num, places=4)
        self.assertTrue(hinge.success)

    def test_predict_stops_at_eos_without_changing_the_text(self):
        model = _model()
        prompt = _batch([([1, 3], [-100, -100])])
        bare = Batch(prompt.records, {key: value for key, value in prompt.payload.items() if key != "labels"},
                     prompt.valid_tokens, 0)

        def generate(**kwargs):
            ids = kwargs["input_ids"]
            extra = torch.tensor([[4, 5, _Words.eos_token_id, 9]], dtype=torch.long)
            return torch.cat((ids.cpu(), extra), dim=1)

        model.module.generate = generate
        generated = model.predict(bare, {"max_new_tokens": 4, "do_sample": False, "seed": 0})
        self.assertEqual(generated["token_ids"], [[4, 5, _Words.eos_token_id]])
        self.assertEqual(generated["texts"], ["the cat sat"])
        self.assertEqual(generated["ids"], bare.ids)

    def test_evaluate_adds_named_text_metrics(self):
        model = _model()
        record = _row("held", [1, 3, 4, 5, 2], [-100, -100, 4, 5, 2])
        record = Record(record.sample_id, record.dataset_id, record.revision, record.source_id, record.task_id,
                        record.arrival, "test", record.inputs, record.target, record.tokens, record.storage_bytes)

        def collate(records):
            return _batch([([1, 3, 4, 5, 2], [-100, -100, 4, 5, 2])])

        def collate_prompts(records):
            packed = _batch([([1, 3], [-100, -100])])
            payload = {key: value for key, value in packed.payload.items() if key != "labels"}
            return Batch(tuple(records), payload, int(payload["attention_mask"].sum()), 0)

        def generate(**kwargs):
            ids = kwargs["input_ids"]
            extra = torch.tensor([[4, 5, _Words.eos_token_id, 9]], dtype=torch.long)
            return torch.cat((ids.cpu(), extra.expand(ids.shape[0], -1)), dim=1)

        model.collate, model.collate_prompts = collate, collate_prompts
        model.module.generate = generate
        result = model.evaluate((record,), {
            "batch_size": 1, "return_predictions": True,
            "generation": {"max_new_tokens": 4, "do_sample": False, "seed": 0},
            "text_metrics": ("rouge_l", "token_similarity")})
        prediction = result["_predictions"][0]["prediction"]
        self.assertEqual(prediction, "the cat sat")
        self.assertAlmostEqual(result["rouge_l"], 0.8)
        self.assertAlmostEqual(result["token_similarity"], 2. / 3)
        self.assertIn("exact_match", result)

    def _step(self, model, batch, mode, weight, baseline, allowance):
        objective = Objective("per_token_mean", token_mode=mode, token_allowance=allowance)
        training = TrainingBatch(batch, ("replay",), token_targets=TokenTargets(baseline, weight))
        return model.train(1, batch_factory=lambda: iter((training,)), objective=objective, hooks=None,
                           limits=TrainLimits(1, 1, batch.valid_tokens))[0]
