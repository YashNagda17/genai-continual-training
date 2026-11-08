"""Instruction/SFT records from canonical JSONL; the tokenizer and prompt template are injected."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable
import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset_base import DatasetBase
from ..utils.contracts import Batch, Record

SPLITS = ("train", "validation", "test")
IGNORE_INDEX = -100


def padded(sequences, value=0, *, left=False):
    """Pad token lists on CPU, including an empty batch."""
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    if not tensors:
        return torch.empty((0, 0), dtype=torch.long)
    if left:
        tensors = [tensor.flip(0) for tensor in tensors]
    result = pad_sequence(tensors, batch_first=True, padding_value=value)
    return result.flip(1) if left else result


@dataclass(frozen=True)
class InstructionPreprocessing:
    encode: Callable[[str], list]
    render: Callable[[list], str]
    eos_id: int
    pad_id: int
    max_length: int
    identity: dict
    truncation: str = "drop"
    allow_reasoning: bool = True
    reserve_response: int = 0

    def describe(self):
        """JSON identity of the pinned tokenizer/template; callables are identified by `identity`."""
        return {"identity": self.identity, "eos_id": self.eos_id, "pad_id": self.pad_id,
                "max_length": self.max_length, "truncation": self.truncation,
                "allow_reasoning": self.allow_reasoning, "reserve_response": self.reserve_response}


def storage_bytes(record):
    return len(json.dumps(asdict(record), sort_keys=True, allow_nan=False).encode())


class _JsonlDataset(DatasetBase):
    """Shared local I/O and finite views; concrete formats own tokenization and collation."""

    def __init__(self, evaluation_split="test"):
        self.evaluation_split = evaluation_split
        self.manifest = {}
        self.views = {}

    @property
    def fingerprint(self):
        """Plain JSON settings for existing callers; no content hashing."""
        settings = {"preprocessing": self.manifest.get("preprocessing"),
                    "evaluation_split": self.evaluation_split}
        if getattr(self, "schema", None) == "summarization":
            settings["protocol"] = self.manifest.get("protocol")
        return json.dumps(settings, sort_keys=True, default=str)

    @property
    def pad_id(self):
        return self.manifest["preprocessing"]["pad_id"]

    def load(self, source_spec):
        """Read canonical JSONL or a saved prepared dataset, without downloading."""
        path = Path(source_spec["path"])
        saved = path.is_dir()
        if saved:
            self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            path = path / "records.jsonl"
        else:
            self.manifest = {"dataset_id": source_spec["dataset_id"],
                             "revision": str(source_spec.get("revision") or source_spec["dataset_id"]),
                             "source_path": str(path)}
        if hasattr(self, "schema"):
            self.manifest["schema"] = self.schema
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    record = Record(**row) if saved else self._record(row, len(records))
                    records.append(record)
        self.views = {}
        if not saved:
            return tuple(records), dict(self.manifest)
        self.views = {name: tuple(r for r in records if r.split == name) for name in SPLITS}
        for name, ids in self.manifest.get("protocol", {}).items():
            self.expose(name, ids)
        return self.views, dict(self.manifest)

    def fetch(self, view, batch_spec, cursor):
        """Return (Batch or None, next_cursor) over the caller's view; never wraps around."""
        if cursor >= len(view):
            return None, cursor
        records = tuple(view[cursor:cursor + batch_spec["size"]])
        return self.collate(records), cursor + len(records)

    def training_view(self, task_id):
        named = self.views.get(f"{task_id}/train")
        if named is not None:
            return named
        return tuple(sorted((r for r in self.views.get("train", ()) if r.task_id == task_id),
                            key=lambda r: (r.arrival, r.sample_id)))

    def evaluation_view(self, task_id):
        name = f"{task_id}/{self.evaluation_split}"
        if name in self.views:
            return self.views[name]
        return tuple(r for r in self.views.get(self.evaluation_split, ()) if r.task_id == task_id)

    def save(self, path):
        """Stream prepared records and their settings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "records.jsonl").open("w", encoding="utf-8") as handle:
            for name in SPLITS:
                for record in self.views.get(name, ()):
                    handle.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
        (path / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")



class InstructionDataset(_JsonlDataset):

    @property
    def has_classes(self):
        train = self.views.get("train", ())
        return bool(train) and all(r.class_id is not None for r in train)

    def _record(self, row, arrival):
        dataset_id, revision = self.manifest["dataset_id"], self.manifest["revision"]
        source_id = str(row["id"])
        messages = row.get("messages", [{"role": "user", "content": row.get("prompt", "")}])
        label = row.get("class")
        group = row.get("group")
        if group is None:
            group = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        record = Record(f"{dataset_id}/{source_id}", dataset_id, revision, source_id,
                        str(row["task"]), arrival, row.get("split", "train"),
                        {"messages": messages}, {"text": row["response"]}, 0, 0,
                        None if label is None else str(label), str(group))
        return replace(record, storage_bytes=storage_bytes(record))

    def split(self, raw_view, split_spec):
        """Keep official splits; optionally move whole train groups of each task to validation."""
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
        """Tokenize prompt and response separately so only response tokens (plus EOS) are supervised."""
        spec = preprocessing_spec
        if type(spec.reserve_response) is not int or spec.reserve_response < 0:
            raise ValueError("reserve_response must be a non-negative int")
        prepared, dropped, truncated = {}, [], []
        for name in SPLITS:
            kept = []
            for r in view.get(name, ()):
                if not spec.allow_reasoning and any(tag in r.target["text"].lower()
                        for tag in ("<think>", "</think>", "<analysis>", "</analysis>")):
                    dropped.append({"sample_id": r.sample_id, "reason": "reasoning_response"})
                    continue
                prompt = list(spec.encode(spec.render(r.inputs["messages"])))
                response = list(spec.encode(r.target["text"])) + [spec.eos_id]
                if len(response) == 1:
                    dropped.append({"sample_id": r.sample_id, "reason": "empty_response"})
                    continue
                if len(response) >= spec.max_length:
                    dropped.append({"sample_id": r.sample_id, "reason": "response_exceeds_max_length"})
                    continue
                # Room for the gold response and, when larger, for a later generation.
                limit = max(len(response), spec.reserve_response)
                if limit > spec.max_length:
                    dropped.append({"sample_id": r.sample_id, "reason": "reserve_exceeds_max_length"})
                    continue
                excess = len(prompt) + limit - spec.max_length
                if excess > 0:
                    if spec.truncation != "prompt_left":
                        dropped.append({"sample_id": r.sample_id, "reason": "exceeds_max_length"})
                        continue
                    prompt = prompt[excess:]
                    truncated.append(r.sample_id)
                record = replace(r, inputs={**r.inputs, "prompt_ids": prompt},
                                 target={**r.target, "ids": response}, tokens=len(prompt) + len(response))
                kept.append(replace(record, storage_bytes=storage_bytes(record)))
            prepared[name] = tuple(kept)
        report = {"dropped": dropped, "truncated": truncated,
                  "kept": {name: len(records) for name, records in prepared.items()}}
        self.manifest["preprocessing"] = {**spec.describe(), "report": report}
        self.views = prepared
        return prepared, report

    def collate(self, records):
        """Right-pad causal inputs; supervise responses and EOS only."""
        records = tuple(records)
        sequences = [list(r.inputs["prompt_ids"]) + list(r.target["ids"]) for r in records]
        targets = [[IGNORE_INDEX] * len(r.inputs["prompt_ids"]) + list(r.target["ids"]) for r in records]
        ids = padded(sequences, self.pad_id)
        mask = padded([[1] * len(seq) for seq in sequences]).bool()
        return Batch(records, {"input_ids": ids, "attention_mask": mask,
                               "labels": padded(targets, IGNORE_INDEX)},
                     sum(map(len, sequences)), sum(len(r.target["ids"]) for r in records))

    def collate_prompts(self, records):
        """Left-pad prompts without including gold responses."""
        records = tuple(records)
        return self.collate_prefixes(records, [0] * len(records))

    def collate_prefixes(self, records, lengths):
        """Left-pad prompt plus the requested target prefix, preserving row order."""
        records, lengths = tuple(records), tuple(lengths)
        if len(lengths) != len(records) or any(type(n) is not int or n < 0 for n in lengths):
            raise ValueError("Prefix lengths must be one non-negative int per record")
        sequences = []
        for record, length in zip(records, lengths):
            if length > len(record.target["ids"]):
                raise ValueError(f"Prefix exceeds the target of {record.sample_id}")
            sequences.append(list(record.inputs["prompt_ids"]) + list(record.target["ids"][:length]))
        return Batch(records, {"input_ids": padded(sequences, self.pad_id, left=True),
                               "attention_mask": padded([[1] * len(seq) for seq in sequences], left=True).bool()},
                     sum(map(len, sequences)), 0)


@dataclass(frozen=True)
class Seq2SeqPreprocessing:
    """Pinned tokenizer and length limits for an encoder-decoder summary."""

    encode: Callable[[str], list]
    eos_id: int
    pad_id: int
    identity: dict
    max_source: int = 256
    max_target: int = 64
    prefix: str | None = None

    def describe(self):
        return {"identity": self.identity, "eos_id": self.eos_id, "pad_id": self.pad_id,
                "max_source": self.max_source, "max_target": self.max_target, "prefix": self.prefix}


def fit_tokens(ids, limit, eos_id):
    """Keep the first tokens and end with EOS. The flag is true when any token was cut."""
    body = list(ids[:-1] if ids and ids[-1] == eos_id else ids)
    over = len(body) + 1 > limit
    if over:
        body = body[:max(limit - 1, 0)]
    return body + [eos_id], over


class SummarizationDataset(_JsonlDataset):
    """Article-summary records with language tasks and separate encoder/decoder tokens."""

    schema = "summarization"

    @property
    def has_classes(self):
        return False

    def _record(self, row, arrival):
        sample_id = str(row["id"])
        language, _, article_id = sample_id.partition("/")
        record = Record(sample_id, self.manifest["dataset_id"], self.manifest["revision"], article_id,
                        language, arrival, row.get("split", "train"),
                        {"text": row["text"], "language": language}, {"text": row["summary"]}, 0, 0, None, None)
        return replace(record, storage_bytes=storage_bytes(record))

    def split(self, raw_view, split_spec):
        """Keep the official train, validation, and test membership."""
        if split_spec.get("validation_fraction", 0):
            raise ValueError("Summarization keeps the official splits")
        views = {name: tuple(record for record in raw_view if record.split == name) for name in SPLITS}
        identifiers = {name: {record.sample_id for record in view} for name, view in views.items()}
        self.manifest["split"] = {
            "spec": {"protocol": "official", "seed": split_spec.get("seed", 0)},
            "id_overlap": {f"{left}/{right}": len(identifiers[left] & identifiers[right])
                           for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))},
        }
        self.views = views
        return views

    def preprocess(self, view, preprocessing_spec):
        """Tokenize the article and the summary separately. Both keep their first tokens."""
        spec = preprocessing_spec
        prepared, dropped = {}, []
        pilot = {}
        for name in SPLITS:
            kept = []
            for record in view.get(name, ()):
                language = record.inputs["language"]
                article = str(record.inputs["text"]).strip()
                summary = str(record.target["text"]).strip()
                if not article or not summary:
                    dropped.append({"sample_id": record.sample_id,
                                    "reason": "empty_article" if not article else "empty_summary"})
                    continue
                source_text = article if not spec.prefix else spec.prefix.format(language=language) + article
                full_source = list(spec.encode(source_text))
                source_ids, source_over = fit_tokens(full_source, spec.max_source, spec.eos_id)
                summary_ids = list(spec.encode(summary))
                summary_body = summary_ids[:-1] if summary_ids and summary_ids[-1] == spec.eos_id else summary_ids
                target_ids, target_over = fit_tokens(summary_ids, spec.max_target, spec.eos_id)
                bucket = pilot.setdefault(language, {"source_tokens": 0, "characters": 0, "rows": 0,
                                                     "source_over": 0, "target_over": 0,
                                                     "source_truncated": 0, "target_truncated": 0})
                bucket["source_tokens"] += len(full_source)
                bucket["characters"] += len(source_text)
                bucket["rows"] += 1
                bucket["source_over"] += int(len(full_source) > spec.max_source)
                bucket["target_over"] += int(len(summary_body) + 1 > spec.max_target)
                bucket["source_truncated"] += int(source_over)
                bucket["target_truncated"] += int(target_over)
                updated = replace(record,
                                  inputs={**record.inputs, "ids": source_ids, "source_truncated": source_over},
                                  target={**record.target, "ids": target_ids, "target_truncated": target_over},
                                  tokens=len(source_ids) + len(target_ids))
                kept.append(replace(updated, storage_bytes=storage_bytes(updated)))
            prepared[name] = tuple(kept)
        ratios = {language: {"tokens_per_character": stats["source_tokens"] / stats["characters"],
                             "source_over_limit": stats["source_over"] / stats["rows"],
                             "target_over_limit": stats["target_over"] / stats["rows"],
                             "source_truncated": stats["source_truncated"],
                             "target_truncated": stats["target_truncated"],
                             "rows": stats["rows"]}
                  for language, stats in pilot.items() if stats["rows"] and stats["characters"]}
        report = {"dropped": dropped, "pilot": ratios,
                  "kept": {name: len(records) for name, records in prepared.items()}}
        self.manifest.pop("protocol", None)
        self.manifest["preprocessing"] = {**spec.describe(), "report": report}
        self.views = prepared
        return prepared, report

    def expose(self, name, sample_ids):
        """Bind the runner's ordered ID list to a view and store that list in the manifest."""
        sample_ids = tuple(sample_ids)
        available = {record.sample_id: record for split in SPLITS for record in self.views.get(split, ())}
        chosen = tuple(available[sample_id] for sample_id in sample_ids)
        if name == "english/pool":
            train_ids = {record.sample_id for record in self.views.get("train", ()) if record.task_id == "english"}
            if any(record.sample_id not in train_ids for record in chosen):
                raise ValueError("The English pool must be inside English train")
        self.views[name] = chosen
        self.manifest.setdefault("protocol", {})[name] = list(sample_ids)
        return chosen

    def pool_view(self):
        """The English revision pool. The runner chooses its members."""
        return self.views["english/pool"]

    def collate(self, records, labels=True):
        """Right-pad encoder inputs and mask target padding with -100."""
        records = tuple(records)
        sources = [r.inputs["ids"] for r in records]
        payload = {"input_ids": padded(sources, self.pad_id),
                   "attention_mask": padded([[1] * len(seq) for seq in sources]).bool()}
        if labels:
            payload["labels"] = padded([r.target["ids"] for r in records], IGNORE_INDEX)
        return Batch(records, payload, sum(map(len, sources)),
                     sum(len(r.target["ids"]) for r in records) if labels else 0)

    def collate_sources(self, records):
        """Article tokens only, for generation."""
        return self.collate(records, labels=False)
