"""Instruction/SFT records from canonical JSONL; the tokenizer and prompt template are injected."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable
import json

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset_base import DatasetBase
from ..utils.contracts import Batch, Record

SPLITS = ("train", "validation", "test")
IGNORE_INDEX = -100


def padded(sequences, value=0):
    """Pad token lists on CPU, including an empty batch."""
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    if not tensors:
        return torch.empty((0, 0), dtype=torch.long)
    return pad_sequence(tensors, batch_first=True, padding_value=value)


def storage_bytes(record):
    return len(json.dumps(asdict(record), sort_keys=True, allow_nan=False).encode())


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


class SummarizationDataset(DatasetBase):
    """Article-summary records with language tasks and separate encoder/decoder tokens."""

    schema = "summarization"

    def __init__(self, evaluation_split="test"):
        self.evaluation_split = evaluation_split
        self.manifest = {}
        self.views = {}

    @property
    def fingerprint(self):
        """Plain JSON settings for existing callers; no content hashing."""
        settings = {"dataset_id": self.manifest.get("dataset_id"), "revision": self.manifest.get("revision"),
                    "preprocessing": self.manifest.get("preprocessing"), "evaluation_split": self.evaluation_split,
                    "protocol": self.manifest.get("protocol")}
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
            payload["supervised_indices"] = (payload["labels"].reshape(-1) != IGNORE_INDEX).nonzero().flatten()
        return Batch(records, payload, sum(map(len, sources)),
                     sum(len(r.target["ids"]) for r in records) if labels else 0)

    def collate_sources(self, records):
        """Article tokens only, for generation."""
        return self.collate(records, labels=False)
