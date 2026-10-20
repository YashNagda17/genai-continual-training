"""Instruction/SFT records from canonical JSONL; the tokenizer and prompt template are injected.

Each source line is one observation:
    {"id": str, "task": str, "split": "train"|"validation"|"test" (default "train"),
     "messages": [{"role", "content"}, ...] or "prompt": str, "response": str,
     "class": genuine label or null, "group": duplicate/leakage group or null}
Sources with identical processing (CLINC150, TRACE, ...) are converted to this schema
and differ only by configuration.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable
import hashlib
import json
import random

import torch

from .dataset_base import DatasetBase
from ..utils.contracts import Batch, Record, fingerprint

SPLITS = ("train", "validation", "test")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class InstructionPreprocessing:
    encode: Callable[[str], list]
    render: Callable[[list], str]
    eos_id: int
    pad_id: int
    max_length: int
    identity: dict
    truncation: str = "drop"

    def describe(self):
        """JSON identity of the pinned tokenizer/template; callables are identified by `identity`."""
        return {"identity": self.identity, "eos_id": self.eos_id, "pad_id": self.pad_id,
                "max_length": self.max_length, "truncation": self.truncation}


def _storage_bytes(record):
    return len(json.dumps(asdict(record), sort_keys=True, allow_nan=False).encode())


class InstructionDataset(DatasetBase):
    def __init__(self, evaluation_split="test"):
        self.evaluation_split = evaluation_split
        self.manifest = {}
        self.views = {}

    @property
    def fingerprint(self):
        identity = {k: v for k, v in self.manifest.items() if k != "source_path"}
        membership = {name: [(r.task_id, r.sample_id) for r in view] for name, view in self.views.items()}
        return fingerprint({"manifest": identity, "evaluation_split": self.evaluation_split,
                            "membership": membership})

    @property
    def has_classes(self):
        train = self.views.get("train", ())
        return bool(train) and all(r.class_id is not None for r in train)

    @property
    def pad_id(self):
        return self.manifest["preprocessing"]["pad_id"]

    def load(self, source_spec):
        """Load {"path", "dataset_id", "sha256"?} JSONL, or a directory written by save().

        A source returns (records, manifest); a saved directory returns (prepared views, manifest).
        """
        path = Path(source_spec["path"])
        if path.is_dir():
            return self._load_saved(path)
        data = path.read_bytes()
        revision = hashlib.sha256(data).hexdigest()
        if source_spec.get("sha256") not in (None, revision):
            raise ValueError(f"Source checksum mismatch for {path}")
        dataset_id = source_spec["dataset_id"]
        records, seen = [], set()
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row["id"])
            if source_id in seen:
                raise ValueError(f"Duplicate source id {source_id}")
            seen.add(source_id)
            split = row.get("split", "train")
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r} for {source_id}")
            messages = row["messages"] if "messages" in row else [{"role": "user", "content": row["prompt"]}]
            label = row.get("class")
            # Exact duplicates share a content group so they cannot straddle splits unnoticed.
            group = row.get("group") or fingerprint([messages, row["response"]])
            record = Record(f"{dataset_id}/{source_id}", dataset_id, revision, source_id,
                            str(row["task"]), len(records), split, {"messages": messages},
                            {"text": row["response"]}, 0, 0,
                            None if label is None else str(label), str(group))
            records.append(replace(record, storage_bytes=_storage_bytes(record)))
        self.manifest = {"dataset_id": dataset_id, "revision": revision, "source_path": str(path)}
        self.views = {}
        return tuple(records), dict(self.manifest)

    def split(self, raw_view, split_spec):
        """Keep official splits; optionally move whole train groups of each task to validation.

        split_spec: {"validation_fraction": float (default 0), "seed": int (default 0)}.
        Group overlap between splits and groups shared across tasks are reported, not repaired.
        """
        fraction, seed = split_spec.get("validation_fraction", 0.0), split_spec.get("seed", 0)
        held = set()
        for task in sorted({r.task_id for r in raw_view}):
            groups = sorted({r.group_id for r in raw_view if r.task_id == task and r.split == "train"})
            random.Random(f"{seed}:{task}").shuffle(groups)
            held.update((task, g) for g in groups[:round(fraction * len(groups))])
        records = [replace(r, split="validation") if r.split == "train" and (r.task_id, r.group_id) in held
                   else r for r in raw_view]
        views = {name: tuple(r for r in records if r.split == name) for name in SPLITS}
        groups = {name: {r.group_id for r in view} for name, view in views.items()}
        tasks_per_group = {}
        for r in views["train"]:
            tasks_per_group.setdefault(r.group_id, set()).add(r.task_id)
        self.manifest["split"] = {
            "spec": {"validation_fraction": fraction, "seed": seed},
            "group_overlap": {f"{a}/{b}": len(groups[a] & groups[b])
                              for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))},
            "cross_task_train_groups": sum(len(t) > 1 for t in tasks_per_group.values()),
        }
        self.views = views
        return views

    def preprocess(self, view, preprocessing_spec):
        """Tokenize prompt and response separately so only response tokens (plus EOS) are supervised.

        `view` maps split names to records. Over-length records are dropped, or with
        truncation="prompt_left" lose their earliest prompt tokens. Empty responses and responses
        that leave no room for a prompt are always dropped rather than becoming zero-loss examples.
        """
        spec = preprocessing_spec
        prepared, dropped, truncated = {}, [], []
        for name in SPLITS:
            kept = []
            for r in view.get(name, ()):
                prompt = list(spec.encode(spec.render(r.inputs["messages"])))
                response = list(spec.encode(r.target["text"])) + [spec.eos_id]
                if len(response) == 1:
                    dropped.append({"sample_id": r.sample_id, "reason": "empty_response"})
                    continue
                if len(response) >= spec.max_length:
                    dropped.append({"sample_id": r.sample_id, "reason": "response_exceeds_max_length"})
                    continue
                excess = len(prompt) + len(response) - spec.max_length
                if excess > 0:
                    if spec.truncation != "prompt_left":
                        dropped.append({"sample_id": r.sample_id, "reason": "exceeds_max_length"})
                        continue
                    prompt = prompt[excess:]
                    truncated.append(r.sample_id)
                record = replace(r, inputs={**r.inputs, "prompt_ids": prompt},
                                 target={**r.target, "ids": response}, tokens=len(prompt) + len(response))
                kept.append(replace(record, storage_bytes=_storage_bytes(record)))
            prepared[name] = tuple(kept)
        report = {"dropped": dropped, "truncated": truncated,
                  "kept": {name: len(records) for name, records in prepared.items()}}
        self.manifest["preprocessing"] = {**spec.describe(), "report": report}
        self.views = prepared
        return prepared, report

    def fetch(self, view, batch_spec, cursor):
        """Return (Batch or None, next_cursor) over the caller's view; never wraps around."""
        if cursor >= len(view):
            return None, cursor
        records = tuple(view[cursor:cursor + batch_spec["size"]])
        return self.collate(records), cursor + len(records)

    def training_view(self, task_id):
        return tuple(sorted((r for r in self._prepared("train") if r.task_id == task_id),
                            key=lambda r: (r.arrival, r.sample_id)))

    def evaluation_view(self, task_id):
        return tuple(r for r in self._prepared(self.evaluation_split) if r.task_id == task_id)

    def collate(self, records):
        """Right-pad prompt+response into Torch tensors on CPU.

        Labels align with input_ids (the model applies the causal shift) and are -100 on prompt
        and padding positions. Record order, including repeated replay IDs, is preserved.
        """
        records = tuple(records)
        width = max((r.tokens for r in records), default=0)
        input_ids = torch.full((len(records), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(records), width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(records), width), dtype=torch.bool)
        for i, r in enumerate(records):
            prompt, response = r.inputs["prompt_ids"], r.target["ids"]
            end = len(prompt) + len(response)
            input_ids[i, :end] = torch.tensor(prompt + response, dtype=torch.long)
            labels[i, len(prompt):end] = torch.tensor(response, dtype=torch.long)
            attention_mask[i, :end] = True
        return Batch(records, {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
                     sum(r.tokens for r in records), sum(len(r.target["ids"]) for r in records))

    def collate_prompts(self, records):
        """Left-pad prompts only, for generation; the gold response never enters the inputs."""
        records = tuple(records)
        width = max((len(r.inputs["prompt_ids"]) for r in records), default=0)
        input_ids = torch.full((len(records), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(records), width), dtype=torch.bool)
        for i, r in enumerate(records):
            prompt = r.inputs["prompt_ids"]
            if prompt:
                input_ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
                attention_mask[i, width - len(prompt):] = True
        return Batch(records, {"input_ids": input_ids, "attention_mask": attention_mask},
                     sum(len(r.inputs["prompt_ids"]) for r in records), 0)

    def save(self, path):
        """Write prepared records and a manifest binding their checksum and fingerprint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(asdict(r), sort_keys=True, allow_nan=False) + "\n"
                       for name in SPLITS for r in self.views.get(name, ())).encode("utf-8")
        (path / "records.jsonl").write_bytes(data)
        manifest = {**self.manifest, "records_sha256": hashlib.sha256(data).hexdigest(),
                    "fingerprint": self.fingerprint}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _load_saved(self, path):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        data = (path / "records.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest.pop("records_sha256"):
            raise ValueError(f"Prepared records in {path} do not match their manifest checksum")
        expected = manifest.pop("fingerprint")
        records = [Record(**json.loads(line)) for line in data.decode("utf-8").splitlines() if line]
        self.manifest = manifest
        self.views = {name: tuple(r for r in records if r.split == name) for name in SPLITS}
        if self.fingerprint != expected:
            raise ValueError(f"Saved membership in {path} does not match its fingerprint "
                             f"(or evaluation_split differs from the saving run)")
        return self.views, dict(self.manifest)

    def _prepared(self, split):
        if "preprocessing" not in self.manifest:
            raise ValueError("Call preprocess() before opening task views")
        return self.views.get(split, ())
