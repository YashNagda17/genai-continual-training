"""XL-Sum records: fixed subsets, tokenization, and collation."""

from dataclasses import dataclass, replace
import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence

SPLITS = ("train", "validation", "test")
LANGUAGES = ("english", "hindi")
IGNORE = -100


@dataclass(frozen=True)
class Record:
    id: str
    language: str
    split: str
    arrival: int
    text: str
    summary: str
    source: tuple = ()
    target: tuple = ()
    practice: dict | None = None  # Pool rows only: the English checkpoint's summary and token losses

    def past(self):
        """The same article, with the English checkpoint's own summary as the target (G2)."""
        return replace(self, target=tuple(self.practice["answer_ids"]))


def choose(records, count, seed):
    """A stable random subset, returned in arrival order."""
    ordered = sorted(records, key=lambda r: (r.arrival, r.id))
    picked = set(random.Random(seed).sample([r.id for r in ordered], count))
    return [r for r in ordered if r.id in picked]


def fit(ids, limit, eos):
    """Keep the first tokens and end with EOS."""
    body = list(ids[:-1] if ids and ids[-1] == eos else ids)
    return tuple(body[:limit - 1] + [eos])


class SummarizationDataset:
    """Records by ID, and the ordered ID lists of the named views (`english/train`, `english/pool`, ...)."""

    def __init__(self, records, ids, pad=0, stats=None):
        self.records, self.ids, self.pad, self.stats = records, ids, pad, stats or {}

    @classmethod
    def xlsum(cls, directory, seed, counts, pool):
        groups = {}
        with (directory / "records.jsonl").open(encoding="utf-8") as handle:
            for arrival, line in enumerate(handle):
                row = json.loads(line)
                groups.setdefault(f"{row['language']}/{row['split']}", []).append(
                    Record(row["id"], row["language"], row["split"], arrival, row["text"], row["summary"]))
        records, ids = {}, {}
        for name, rows in groups.items():
            chosen = choose(rows, counts[name.split("/")[1]], seed)
            ids[name] = [r.id for r in chosen]
            records.update((r.id, r) for r in chosen)
        for offset, language in enumerate(LANGUAGES, 1):  # The English pool, and the reverse order's Hindi pool
            ids[f"{language}/pool"] = [r.id for r in choose([records[i] for i in ids[f"{language}/train"]], pool, seed + offset)]
        return cls(records, ids)

    def tokenize(self, tokenizer, max_source=256, max_target=64, prefix=None):
        """Tokenize article and summary separately. Records tokens per character for the cost report."""
        eos, counts = tokenizer.eos_token_id, {}

        def prepare(record):
            text = prefix.format(language=record.language) + record.text if prefix else record.text
            full = tokenizer.encode(text)
            count = counts.setdefault(record.language, [0, 0, 0, 0])
            count[0], count[1] = count[0] + len(full), count[1] + len(text)
            count[2], count[3] = count[2] + 1, count[3] + (len(full) > max_source)
            return replace(record, source=fit(full, max_source, eos),
                           target=fit(tokenizer.encode(record.summary), max_target, eos))
        records = {i: prepare(r) for i, r in self.records.items()}
        stats = {language: {"tokens_per_character": tokens / characters, "source_over_limit": over / rows}
                 for language, (tokens, characters, rows, over) in counts.items()}
        return SummarizationDataset(records, self.ids, tokenizer.pad_token_id, stats)

    def view(self, name):
        return tuple(self.records[i] for i in self.ids[name])

    def collate(self, records, labels=True):
        """Right-pad articles; mask summary padding with -100."""
        sources = [torch.tensor(r.source) for r in records]
        batch = {"input_ids": pad_sequence(sources, batch_first=True, padding_value=self.pad),
                 "attention_mask": pad_sequence([torch.ones(len(s), dtype=torch.long) for s in sources],
                                                batch_first=True)}
        if labels:
            batch["labels"] = pad_sequence([torch.tensor(r.target) for r in records], batch_first=True,
                                           padding_value=IGNORE)
        return batch
