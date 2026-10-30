"""Task-end CSReL-CL and CL-Prv, with resumable reference and subset fitting."""

from dataclasses import dataclass

import torch

from .pipeline_base import PipelineBase
from ..utils.contracts import Objective, ScoreSpec, TrainingBatch
from ..utils.numerics import quotas, sample, tensor


@dataclass(frozen=True)
class CSReLConfig:
    variant: str
    rounds: int
    reference_passes: int
    subset_passes: int
    reference_initialization: str
    subset_initialization: str
    replay_weight: float
    replay_batch_size: int
    replacement: bool


class CSReLPipeline(PipelineBase):
    def __init__(self, config, model=None, dataset=None):
        """Keep retained reference losses separately from resumable task-end selection work."""
        super().__init__(config, model, dataset)
        self.policy = {"reference_losses": {}, "work": None}

    def auxiliary_roles(self):
        """Declare independent reference and subset initializations; neither is the learner."""
        return (("reference", self.config.reference_initialization),
                ("subset", self.config.subset_initialization))


    def fit_model(self, model, records, passes, selection, owner):
        """Train an auxiliary for ordered finite passes over the supplied records.
        Persist its pass/cursor after each attempt so unfinished fitting can resume exactly.
        An empty selected subset keeps the declared initialization without any optimizer update.
        """
        fit = selection.setdefault("fit", {"pass": 0, "cursor": 0})
        if not records:
            return  # Explicit untrained initialization at an empty selected subset.
        while fit["pass"] < passes:
            start = fit["cursor"]
            batch = owner.collate(records[start:start + owner.spec.batch_size])
            report = owner.execute(
                model, TrainingBatch(batch, ("auxiliary",) * len(batch.ids)),
                Objective(owner.spec.reduction))
            fit["cursor"] += len(batch.ids)
            owner.log_sample_progress(
                f"csrel {selection['stage']} task={selection['task']}",
                fit["pass"] * len(records) + fit["cursor"], passes * len(records))
            if fit["cursor"] == len(records):
                fit["cursor"] = 0
                fit["pass"] += 1
            owner.record_event("auxiliary_update", ids=list(batch.ids), report=owner.report_summary(report),
                               selection_task=selection["task"], phase=selection["stage"])
            owner.boundary()

    def score_losses(self, model, records, selection, owner, key):
        """Build a candidate-ID loss table with model, reduction and loss-fingerprint provenance.
        On resume, score only IDs missing from the table, preserving completed observations.
        Checkpoint each scored batch while the matching auxiliary weights remain available.
        """
        table = selection.setdefault(key, {})
        pending = tuple(r for r in records if r.sample_id not in table)
        for start in range(0, len(pending), owner.spec.batch_size):
            batch = owner.collate(pending[start:start + owner.spec.batch_size])
            spec = ScoreSpec("loss", owner.spec.reduction)
            result = model.score(batch, spec)
            for sample_id, value in zip(batch.ids, tensor(result.values).tolist()):
                table[sample_id] = {"loss": value, "model": result.model_revision,
                                    "reduction": result.reduction,
                                    "loss_fingerprint": model.info.loss_fingerprint}
            owner.log_sample_progress(
                f"csrel {key} task={selection['task']}", len(table), len(records))
            owner.record_event("auxiliary_scoring", ids=list(batch.ids), table=key, cost=result.cost)
            owner.boundary()
        return table

    def reference_pool(self, candidates, replay):
        """CL uses the candidate task; CL-Prv unions it with permitted retained history.
        Deduplicate by observation ID without reopening discarded old task data.
        """
        if self.config.variant == "csrel_cl_prv":
            return tuple({r.sample_id: r for r in (*candidates, *replay.records)}.values())
        return tuple(candidates)

    def train_reference(self, candidates, replay, selection, owner):
        """Fit the isolated reference on the variant's permitted training pool.
        After fitting, clear its pass cursor and transition to reference scoring.
        Checkpoint with the fitted auxiliary still resident so scoring can resume from it.
        """
        model = owner.acquire_auxiliary("reference", self.config.reference_initialization)
        self.fit_model(model, self.reference_pool(candidates, replay),
                       self.config.reference_passes, selection, owner)
        selection.pop("fit", None)
        selection["stage"] = "reference_score"
        owner.boundary(force=True)

    def score_reference(self, candidates, selection, owner):
        """Finish the reference-loss table before releasing its scoring model.
        Transition to subset fitting only after every candidate has a stored reference loss.
        The following checkpoint preserves that table without retaining reference optimizer state.
        """
        model = owner.acquire_auxiliary("reference", self.config.reference_initialization)
        self.score_losses(model, candidates, selection, owner, "reference")
        owner.release_auxiliary()
        selection["stage"] = "subset_fit"
        owner.boundary(force=True)

    def train_subset(self, records, selection, owner):
        """Fit the isolated subset model on the records selected by earlier rounds.
        A new round starts from the declared initialization; a resumed fit reuses its saved role.
        Transition to scoring and checkpoint the fitted model before choosing more records.
        """
        model = owner.acquire_auxiliary("subset", self.config.subset_initialization)
        self.fit_model(model, records, self.config.subset_passes, selection, owner)
        selection.pop("fit", None)
        selection["stage"] = "subset_score"
        owner.boundary(force=True)

    @staticmethod
    def reducible_losses(subset, reference, device=None):
        """Compute ReL(i) = subset loss - reference loss for each unselected candidate.
        Preserve negative differences; a high subset loss alone does not imply high reducibility.
        """
        ids = list(subset)
        losses = tensor([subset[i]["loss"] for i in ids], device)
        baseline = tensor([reference[i]["loss"] for i in ids], device)
        return dict(zip(ids, (losses - baseline).tolist()))

    @staticmethod
    def select_top(scores, count, device=None):
        """Choose the largest reducible losses, resolving equal scores by stable sample ID."""
        ids = sorted(scores)
        values = tensor([scores[i] for i in ids], device)
        order = torch.argsort(values, descending=True, stable=True)[:count].tolist()
        return [ids[index] for index in order]

    def select_round(self, candidates, selection, count, owner):
        """Score only unselected candidates with the current round's fitted subset model.
        Append the highest reducible-loss IDs and discard this round's transient loss table.
        Release the subset role and checkpoint the next round with its selected membership.
        """
        model = owner.acquire_auxiliary("subset", self.config.subset_initialization)
        remaining = tuple(r for r in candidates if r.sample_id not in selection["selected"])
        losses = self.score_losses(model, remaining, selection, owner, "subset")
        scores = self.reducible_losses(losses, selection["reference"], model.device)
        chosen = self.select_top(scores, count, model.device)
        round_index = selection["round"]
        selection["selected"].extend(chosen)
        selection["round"] += 1
        selection["stage"] = "subset_fit"
        selection.pop("subset", None)
        owner.release_auxiliary()
        owner.record_event("coreset_round", selection_task=selection["task"], round=round_index,
                           scores=scores, selected=chosen)
        owner.boundary(force=True)

    def select_coreset(self, candidates, replay, quota, selection, owner):
        """Resume reference preparation, then alternate subset fitting and selection.
        Each round fits the growing selected set and recomputes remaining candidate scores.
        Assign the integer allocation remainder to the final round; zero-size rounds do no fitting.
        """
        if selection["stage"] == "reference_fit":
            self.train_reference(candidates, replay, selection, owner)
        if selection["stage"] == "reference_score":
            self.score_reference(candidates, selection, owner)
        by_id = {r.sample_id: r for r in candidates}
        while selection["round"] < self.config.rounds:
            round_index = selection["round"]
            count = quota // self.config.rounds
            if round_index == self.config.rounds - 1:
                count = quota - count * round_index
            if count == 0:
                selection["round"] += 1
                continue
            if selection["stage"] == "subset_fit":
                selected = tuple(by_id[i] for i in selection["selected"])
                self.train_subset(selected, selection, owner)
            self.select_round(candidates, selection, count, owner)
        return tuple(by_id[i] for i in selection["selected"])

    def initialize_selection(self, task, candidates, current_task):
        """Create the selected-ID, round and stage cursors for one candidate task.
        For old CL tasks, reuse original retained-example reference losses and skip reference fitting.
        Current CL and all CL-Prv selections instead begin with a fresh reference fit.
        """
        selection = {"task": task, "selected": [], "round": 0, "stage": "reference_fit"}
        if task != current_task and self.config.variant == "csrel_cl":
            selection["reference"] = {r.sample_id: self.policy["reference_losses"][r.sample_id]
                                      for r in candidates}
            selection["stage"] = "subset_fit"
        return selection

    def build_memory(self, current, replay, seen_tasks, owner):
        """Allocate equal task quotas and construct a resumable replay proposal.
        Use all permitted current candidates, but only retained records from older tasks.
        Save task/round progress and reference losses without updating the learner or active replay.
        Record underfilled quotas rather than duplicating examples or redistributing spare slots.
        """
        work = self.policy["work"]
        if work is None:
            work = {"quotas": quotas(seen_tasks, owner.spec.replay.examples),
                    "tasks": sorted(seen_tasks), "index": 0, "selected": [],
                    "references": {}, "selection": None}
            self.policy["work"] = work
        by_id = {r.sample_id: r for r in (*current, *replay.records)}
        while work["index"] < len(work["tasks"]):
            task = work["tasks"][work["index"]]
            candidates = tuple(current) if task == owner.task_id else replay.task(task)
            requested = work["quotas"][task]
            quota = min(requested, len(candidates))
            if quota == 0:
                owner.record_event("coreset_quota", selection_task=task,
                                   requested=requested, retained=0)
                work["index"] += 1
                owner.boundary()
                continue
            if work["selection"] is None:
                work["selection"] = self.initialize_selection(task, candidates, owner.task_id)
            selection = work["selection"]
            self.select_coreset(candidates, replay, quota, selection, owner)
            work["selected"].extend(selection["selected"])
            work["references"].update({i: selection["reference"][i] for i in selection["selected"]})
            work["selection"] = None
            work["index"] += 1
            owner.record_event("coreset_quota", selection_task=task,
                               requested=requested, retained=quota, underfill=requested - quota)
            owner.boundary(force=True)
        return tuple(by_id[i] for i in work["selected"])

    def commit(self):
        """Promote selected reference losses only after the owner commits the proposed replay."""
        work = self.policy["work"]
        self.policy["reference_losses"] = work["references"]
        self.policy["work"] = None

    def open_task(self):
        """Standalone CSReL needs no additional policy work before learner training."""
        pass

    def retrieve_replay(self, history, count):
        """Sample retained history uniformly, allowing a smaller batch without replacement."""
        if not history or not count:
            return ()
        count = count if self.config.replacement else min(count, len(history))
        return sample(history, [1 / len(history)] * len(history), count,
                      self.config.replacement, self.rng)

    def objective(self, reduction):
        """Keep separate current/replay means, scaling only the replay term by its configured weight."""
        return Objective(reduction, replay_weight=self.config.replay_weight, separate_means=True)

    def next_update(self):
        """Pair the next current batch with a uniform sample of retained history.
        Preserve origin labels so the model can compute separate current/replay loss means.
        Return the learner update without invoking task-end auxiliary selection.
        """
        current = self.current_batch()
        if not current:
            return None
        replay = self.retrieve_replay(self.replay.records, self.config.replay_batch_size)
        return (current + replay, ["current"] * len(current) + ["replay"] * len(replay),
                self.objective(self.spec.reduction), None, None)

    def finish_task(self):
        """Construct task-end memory from the current chunk and currently retained history."""
        return self.build_memory(self.current, self.replay, self.spec.task_ids[:self.task_index + 1], self)

    def after_memory_commit(self):
        """Finalize cached reference losses after the shared lifecycle accepts the new memory."""
        self.commit()
