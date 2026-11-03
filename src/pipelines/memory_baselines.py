"""Matched baselines for the standalone CSReL experiment.

new_only trains on the current task and stores nothing.
random uses CSReL's uniform replay and replaces only the memory writer.
cumulative retrains from the run's initial frozen weights on every seen training view.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import math
import shutil
import time

from .pipeline_base import PipelineBase
from ..utils.contracts import Objective, training_records
from ..utils.numerics import quotas, sample


@dataclass(frozen=True)
class MemoryBaselineConfig:
    method: str
    replay_weight: float
    replay_batch_size: int
    replacement: bool
    require_full_passes: bool = False

    def __post_init__(self):
        if self.method not in {"new_only", "random", "cumulative"}:
            raise ValueError(f"Unknown memory baseline {self.method}")


class MemoryBaselinePipeline(PipelineBase):
    def __init__(self, config, model=None, dataset=None):
        """Store resumable random-selection progress. Model tensors stay out of this policy."""
        super().__init__(config, model, dataset)
        self.policy = {"work": None}

    def objective(self, reduction):
        """Match standalone CSReL: separate current and replay means, replay scaled by its weight."""
        return Objective(reduction, replay_weight=self.config.replay_weight, separate_means=True)

    def retrieve_replay(self, history, count):
        """Draw uniformly with the replay RNG, leaving the selection RNG untouched."""
        if not history or not count:
            return ()
        count = count if self.config.replacement else min(count, len(history))
        return sample(history, [1 / len(history)] * len(history), count,
                      self.config.replacement, self.rng)

    def open_task(self):
        """Cumulative runs reload the common initial snapshot before any update on this task."""
        if self.config.method == "cumulative":
            if self.config.require_full_passes:
                updates = math.ceil(len(self.current) / self.spec.batch_size) * self.spec.passes
                tokens = sum(record.tokens for record in self.current) * self.spec.passes
                if (updates > min(self.spec.max_steps, self.spec.max_attempts)
                        or tokens > self.spec.max_tokens):
                    raise ValueError("Cumulative budget cannot cover all configured training passes")
            self._restore_initial_weights()
            self.record_event("cumulative_retrain", training_examples=len(self.current),
                              initialization="common_frozen_weights")

    def _restore_initial_weights(self):
        """Save the first learner snapshot once, then install it at every later task opening."""
        path = Path(self.context.output) / "initial_weights"
        ready = (path / "model.json").is_file() or (path / "metadata.json").is_file()
        if not ready:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(path, "frozen")
        self.model.load(path, "frozen")

    def _open_view(self):
        """Cumulative training reads every seen task. Bounded replay methods keep the runtime view."""
        if self.config.method != "cumulative":
            return super()._open_view()
        records = []
        for task in self.spec.task_ids[:self.task_index + 1]:
            chunk = tuple(training_records(self.dataset.training_view(task), task))
            if list(chunk) != sorted(chunk, key=lambda record: (record.arrival, record.sample_id)):
                raise ValueError("Training view must preserve canonical arrival order")
            records.extend(chunk)
        if not records:
            raise ValueError("Required training partition is empty")
        return tuple(records)

    def next_update(self):
        """Current batches for every arm. Only random adds a uniform replay draw."""
        current = self.current_batch()
        if not current:
            return None
        replay = ()
        if self.config.method == "random":
            replay = self.retrieve_replay(self.replay.records, self.config.replay_batch_size)
        return (current + replay, ["current"] * len(current) + ["replay"] * len(replay),
                self.objective(self.spec.reduction), None, None)

    def _choose(self, candidates, count, task):
        """Draw a stable without-replacement subset from an independent selection seed."""
        identity = f"{self.spec.seed}:random-selection:{task}:" + ",".join(
            record.sample_id for record in candidates)
        chooser = random.Random(identity)
        return sample(candidates, [1 / len(candidates)] * len(candidates), count, False, chooser)

    def build_memory(self, current, replay, seen_tasks, owner):
        """Fill equal global quotas from the current task and retained older examples only."""
        started = time.monotonic()
        work = self.policy["work"]
        if work is None:
            work = {"quotas": quotas(seen_tasks, owner.spec.replay.examples),
                    "tasks": sorted(seen_tasks), "index": 0, "selected": []}
            self.policy["work"] = work
        by_id = {record.sample_id: record for record in (*current, *replay.records)}
        while work["index"] < len(work["tasks"]):
            task = work["tasks"][work["index"]]
            candidates = tuple(current) if task == owner.task_id else replay.task(task)
            requested = work["quotas"][task]
            quota = min(requested, len(candidates))
            chosen = self._choose(candidates, quota, task) if quota else ()
            work["selected"].extend(record.sample_id for record in chosen)
            work["index"] += 1
            owner.record_event("random_quota", selection_task=task, requested=requested,
                               retained=len(chosen), underfill=requested - len(chosen),
                               selected=[record.sample_id for record in chosen])
            owner.boundary(force=True)
        owner.record_event("selection_cost", method="random",
                           wall_seconds=time.monotonic() - started)
        return tuple(by_id[sample_id] for sample_id in work["selected"])

    def finish_task(self):
        """Random proposes a new memory. The other baselines commit an empty memory."""
        if self.config.method != "random":
            return ()
        return self.build_memory(self.current, self.replay, self.spec.task_ids[:self.task_index + 1], self)

    def after_memory_commit(self):
        """Drop in-progress selection after the owner accepts the proposal."""
        self.policy["work"] = None
