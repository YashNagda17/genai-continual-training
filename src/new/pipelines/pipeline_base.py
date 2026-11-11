"""Shared batching, cost accounting, and resume state for the three pipelines."""

from pathlib import Path
import random
import shutil
import time

from ..utils.numerics import read_json, synchronize, write_json


class PipelineBase:
    def __init__(self, settings, model, dataset, output, seed=0):
        self.settings, self.model, self.dataset = settings, model, dataset
        self.output, self.seed, self.rng = Path(output), seed, random.Random(seed)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = {}
        self.costs = {"update_seconds": 0.0, "analysis_seconds": {}, "forward_pass_units": 0,
                      "tokens_per_language": {"english": 0, "hindi": 0}}

    def timed(self, function, category=None, rows=(), units=1):
        """Synchronized wall time. Analysis passes are charged per category, in forward-pass units."""
        synchronize(self.model.device)
        start = time.perf_counter()
        result = function()
        synchronize(self.model.device)
        seconds = time.perf_counter() - start
        if category is None:
            self.costs["update_seconds"] += seconds
        else:
            spent = self.costs["analysis_seconds"]
            spent[category] = spent.get(category, 0.0) + seconds
        self.costs["forward_pass_units"] += units * len(rows)
        return result

    def batches(self, rows):
        size = self.settings.analysis_batch
        return (rows[start:start + size] for start in range(0, len(rows), size))

    def generate(self, rows, category, guard=None):
        texts, ids = [], []
        for part in self.batches(rows):
            batch = self.dataset.collate(part, labels=False)
            written = self.timed(lambda: self.model.generate(batch, self.settings.max_new_tokens,
                                                             self.settings.no_repeat, guard), category, part)
            texts += written[0]
            ids += written[1]
        return texts, ids

    def token_losses(self, rows, category):
        losses = []
        for part in self.batches(rows):
            batch = self.dataset.collate(part)
            losses += self.timed(lambda: self.model.token_losses(batch), category, part)
        return losses

    def checkpoint(self):
        """Latest recovery state plus one previous fallback; the latest is replaced only once complete."""
        latest, previous, staging = (self.output / name for name in ("resume", "resume.prev", "resume.tmp"))
        shutil.rmtree(staging, ignore_errors=True)
        self.model.save(staging, training=True)
        write_json(staging / "state.json", {"state": self.state, "costs": self.costs, "rng": self.rng.getstate()})
        if latest.exists():
            shutil.rmtree(previous, ignore_errors=True)
            latest.rename(previous)
        staging.rename(latest)

    def restore(self):
        for path in (self.output / "resume", self.output / "resume.prev"):
            if (path / "state.json").is_file():
                saved = read_json(path / "state.json")
                self.model.load(path, training=True)
                self.state, self.costs = saved["state"], saved["costs"]
                version, values, gauss = saved["rng"]
                self.rng.setstate((version, tuple(values), gauss))
                return True
        return False
