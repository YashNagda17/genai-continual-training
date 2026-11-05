"""Generation-aware replay checks. These cover dataset room and prefix collation only."""

import tempfile
import unittest

from ..datasets.dataset_instruction import InstructionDataset, InstructionPreprocessing
from ..utils.contracts import Record


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
