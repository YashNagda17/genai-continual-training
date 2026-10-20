from pathlib import Path
import json
import tempfile
import unittest

from ..datasets.dataset_instruction import IGNORE_INDEX, InstructionDataset, InstructionPreprocessing

ROWS = (
    {"id": "a0", "task": "a", "prompt": "hi", "response": "yo", "class": "greet"},
    {"id": "a1", "task": "a", "prompt": "long prompt", "response": "ok", "class": "other"},
    {"id": "a2", "task": "a", "prompt": "dup", "response": "same", "class": "greet"},
    {"id": "a3", "task": "a", "prompt": "dup", "response": "same", "class": "greet", "split": "test"},
    {"id": "a4", "task": "a", "prompt": "x", "response": "", "class": "other"},
    {"id": "b0", "task": "b", "messages": [{"role": "system", "content": "s"},
                                           {"role": "user", "content": "q"}],
     "response": "answer", "class": "qa"},
    {"id": "b1", "task": "b", "prompt": "p", "response": "r", "class": "qa", "split": "test"},
)


def tokenizer(max_length=12, truncation="drop"):
    return InstructionPreprocessing(
        encode=lambda text: [ord(c) for c in text],
        render=lambda messages: "".join(m["content"] for m in messages) + ">",
        eos_id=1, pad_id=0, max_length=max_length, identity={"tokenizer": "chars-v1"},
        truncation=truncation)


def prepare(root, rows=ROWS, max_length=12, truncation="drop", validation_fraction=0.0):
    source = Path(root, "source.jsonl")
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = InstructionDataset()
    raw, _ = dataset.load({"path": source, "dataset_id": "toy"})
    views = dataset.split(raw, {"validation_fraction": validation_fraction, "seed": 3})
    _, report = dataset.preprocess(views, tokenizer(max_length, truncation))
    return dataset, report, source


class InstructionDatasetTests(unittest.TestCase):
    def test_collate_supervises_only_response_and_eos(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, _, _ = prepare(root)
            a0, a2 = dataset.training_view("a")[0], dataset.training_view("a")[-1]
            batch = dataset.collate((a2, a0, a2))
            payload = batch.payload
            self.assertEqual(batch.ids, ("toy/a2", "toy/a0", "toy/a2"))
            prompt, response = [ord(c) for c in "hi>"], [ord(c) for c in "yo"] + [1]
            width = payload["input_ids"].shape[1]
            self.assertEqual(width, a2.tokens)
            self.assertEqual(payload["input_ids"][1].tolist(), prompt + response + [0] * (width - 6))
            self.assertEqual(payload["labels"][1].tolist(),
                             [IGNORE_INDEX] * 3 + response + [IGNORE_INDEX] * (width - 6))
            self.assertEqual(payload["attention_mask"][1].sum().item(), 6)
            self.assertEqual(batch.valid_tokens, a2.tokens * 2 + a0.tokens)
            self.assertEqual(batch.supervised_tokens, (payload["labels"] != IGNORE_INDEX).sum().item())

            prompts = dataset.collate_prompts((a0, a2))
            self.assertEqual(prompts.payload["input_ids"][0].tolist(), [0] + prompt)
            self.assertEqual(prompts.supervised_tokens, 0)

    def test_task_views_are_causal_and_drops_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, report, _ = prepare(root)
            self.assertEqual([r.sample_id for r in dataset.training_view("a")], ["toy/a0", "toy/a2"])
            self.assertEqual([r.sample_id for r in dataset.training_view("b")], ["toy/b0"])
            self.assertEqual([r.sample_id for r in dataset.evaluation_view("a")], ["toy/a3"])
            self.assertTrue(all(r.split == "test" for r in dataset.evaluation_view("b")))
            self.assertEqual({d["sample_id"]: d["reason"] for d in report["dropped"]},
                             {"toy/a1": "exceeds_max_length", "toy/a4": "empty_response"})
            self.assertTrue(dataset.has_classes)

            dataset, report, _ = prepare(root, truncation="prompt_left")
            a1 = next(r for r in dataset.training_view("a") if r.source_id == "a1")
            self.assertEqual(report["truncated"], ["toy/a1"])
            self.assertEqual(a1.tokens, 12)
            self.assertEqual(a1.target["ids"], [ord("o"), ord("k"), 1])
            self.assertEqual(a1.target["text"], "ok")

    def test_split_keeps_groups_whole_and_reports_overlap(self):
        rows = [{"id": f"r{i}", "task": "a", "prompt": f"p{i % 4}", "response": "r", "class": "c"}
                for i in range(12)]
        with tempfile.TemporaryDirectory() as root:
            dataset, _, _ = prepare(root, rows, validation_fraction=.5)
            train, validation = dataset.views["train"], dataset.views["validation"]
            self.assertEqual(len({r.group_id for r in validation}), 2)
            self.assertFalse({r.group_id for r in train} & {r.group_id for r in validation})
            again, _, _ = prepare(root, rows, validation_fraction=.5)
            self.assertEqual(dataset.fingerprint, again.fingerprint)

            dataset, _, _ = prepare(root)
            self.assertEqual(dataset.manifest["split"]["group_overlap"]["train/test"], 1)

    def test_manifest_reload_and_rejected_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, _, source = prepare(root)
            saved = Path(root, "prepared")
            dataset.save(saved)
            reloaded = InstructionDataset()
            reloaded.load({"path": saved})
            self.assertEqual(reloaded.fingerprint, dataset.fingerprint)
            self.assertEqual(reloaded.training_view("a"), dataset.training_view("a"))
            self.assertEqual(reloaded.collate(reloaded.training_view("b")).payload["labels"].tolist(),
                             dataset.collate(dataset.training_view("b")).payload["labels"].tolist())

            with self.assertRaises(ValueError):
                InstructionDataset().load({"path": source, "dataset_id": "toy", "sha256": "0" * 64})
            records = saved / "records.jsonl"
            records.write_bytes(records.read_bytes().replace(b'"yo"', b'"no"'))
            with self.assertRaises(ValueError):
                InstructionDataset().load({"path": saved})
            with self.assertRaises(ValueError):
                InstructionDataset().training_view("a")


if __name__ == "__main__":
    unittest.main()
