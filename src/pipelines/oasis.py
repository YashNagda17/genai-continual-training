"""OASIS: separately callable gradient scoring, SIREN, ORIS and admission."""

from dataclasses import dataclass
from itertools import combinations
import math

import torch

from .pipeline_base import PipelineBase
from ..utils.contracts import Objective, ScoreSpec, SelectionResult
from ..utils.numerics import cosine_tensor, tensor


@dataclass(frozen=True)
class OASISConfig:
    beta: float
    initial_mean: float
    initial_variance: float
    variance_floor: float
    target_rate: float
    max_selection_batch: int
    gradient_view: str
    profile: str


class OASISPipeline(PipelineBase):
    def __init__(self, config, model=None, dataset=None):
        """Initialize arrival statistics and the normal-model admission threshold.
        Rates zero and one bypass calibration and act as deterministic admission controls.
        """
        super().__init__(config, model, dataset)
        self.policy = {
            "mean": config.initial_mean, "variance": config.initial_variance, "arrivals": 0,
            "threshold": self.calibrate_threshold(config.target_rate) if 0 < config.target_rate < 1 else None,
        }

    def required_capabilities(self):
        """Request per-example last-layer gradients from the supplied model adapter."""
        return {"last_layer_gradients"}

    @staticmethod
    def expected_rate(threshold, intervals=4000):
        """Approximate Appendix A.7 with midpoint quadrature over a standard normal on [-10, 10].
        The expectation maps a shared sigmoid threshold to its modeled admission rate.
        """
        step = 20 / intervals
        points = -10 + (torch.arange(intervals, dtype=torch.float64) + .5) * step
        density = torch.exp(-points.square() / 2) / math.sqrt(2 * math.pi)
        return (torch.sigmoid(points - threshold) * density).sum().item() * step

    @classmethod
    def calibrate_threshold(cls, rate):
        """Invert the normal-model expected rate by bisection over the threshold.
        Higher thresholds lower admission, which determines which half of the interval survives.
        This calibrates an expected rate rather than imposing a fixed batch quota.
        """
        lo, hi = -80.0, 80.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if cls.expected_rate(mid) > rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def score_gradients(self, batch, model=None, reduction=None):
        """Request non-mutating last-layer scores from the explicit or shared learner.
        A bound run supplies the reduction; callers of an isolated step provide it explicitly.
        """
        model = self.model if model is None else model
        reduction = self.spec.reduction if reduction is None else reduction
        return model.score(batch, ScoreSpec("last_layer_gradients", reduction, self.config.gradient_view))

    @staticmethod
    def informativeness(gradients):
        """Equation 2: use squared last-layer gradient norms as raw example informativeness."""
        return tuple(tensor(gradients).square().sum(dim=-1).tolist())

    @staticmethod
    def adjusted_score(index, selected, gradients, raw):
        """Equation 8 adjusts one candidate against the already selected set H.
        Enumerate every nonempty subset of H, with alternating inclusion-exclusion signs.
        Each correction uses mean subset gradients and raw energies, preserving signed cosine.
        """
        gradients = tensor(gradients)
        raw = tensor(raw, gradients.device)
        value = raw[index].clone()
        for size in range(1, len(selected) + 1):
            for subset in combinations(selected, size):
                indices = list(subset)
                mean_gradient = gradients[indices].mean(dim=0)
                mean_energy = raw[indices].mean()
                value += (-1) ** size * cosine_tensor(gradients[index], mean_gradient) * mean_energy
        return value.item()

    @classmethod
    def siren(cls, gradients, ids):
        """Greedily extend H using the largest currently adjusted informativeness.
        Recompute remaining scores from raw energies after each addition to H.
        Break ties by sample ID and return each chosen score in the original input order.
        """
        gradients = tensor(gradients)
        raw = gradients.square().sum(dim=-1)
        selected, remaining, output = [], set(range(len(ids))), [0.0] * len(ids)
        while remaining:
            scores = {i: cls.adjusted_score(i, selected, gradients, raw) for i in remaining}
            chosen = min(remaining, key=lambda i: (-scores[i], ids[i]))
            output[chosen] = scores[chosen]
            selected.append(chosen)
            remaining.remove(chosen)
        return tuple(output)

    def relative_informativeness(self, adjusted):
        """Equation 4 normalizes adjusted scores with the statistics preceding this arrival.
        Use the configured variance floor before taking the standard deviation.
        """
        values = tensor(adjusted, self.device)
        scale = tensor(max(self.policy["variance"], self.config.variance_floor), self.device).sqrt()
        return tuple(((values - self.policy["mean"]) / scale).tolist())

    def admission_probabilities(self, relative):
        """Apply the calibrated sigmoid to normalized scores, or the deterministic zero/one control."""
        rate = self.config.target_rate
        values = tensor(relative, self.device)
        probabilities = (torch.full_like(values, rate) if rate in (0, 1)
                         else torch.sigmoid(values - self.policy["threshold"]))
        return tuple(probabilities.tolist())

    def sample_admission(self, ids, probabilities):
        """Use one shared uniform threshold for all candidate probabilities in this batch.
        Return the draw with selected IDs so the admission decision can be inspected.
        """
        draw = self.rng.random() if self.config.target_rate not in (0, 1) else 0.5
        return tuple(i for i, probability in zip(ids, probabilities) if probability > draw), draw

    def update_statistics(self, raw):
        """Equation 3 updates statistics from the mean of all raw arrival energies.
        Compute the variance increment relative to the previous mean, then update both EMAs.
        Rejected examples still contribute, and the arrival counter advances once per batch.
        """
        average = tensor(raw, self.device).mean()
        mean, variance, beta = self.policy["mean"], self.policy["variance"], self.config.beta
        self.policy.update(mean=(beta * average + (1 - beta) * mean).item(),
                           variance=(beta * (average - mean).square() + (1 - beta) * variance).item(),
                           arrivals=self.policy["arrivals"] + 1)

    def select_current(self, batch, model=None, reduction=None):
        """Compose gradient scoring, exact SIREN, normalization and probabilistic admission.
        Make the decision using the previous statistics before applying the arrival EMA update.
        Return aligned scores, selected IDs and provenance; this step never trains the learner.
        """
        result = self.score_gradients(batch, model, reduction)
        source = self.model if model is None else model
        gradients = tensor(result.values, source.device)
        raw = self.informativeness(gradients)
        adjusted = self.siren(gradients, batch.ids)
        relative = self.relative_informativeness(adjusted)
        probabilities = self.admission_probabilities(relative)
        selected, draw = self.sample_admission(batch.ids, probabilities)
        mean, variance = self.policy["mean"], self.policy["variance"]
        self.update_statistics(raw)
        return SelectionResult(batch.ids, selected, adjusted, probabilities, {
            "profile": "paper_equations", "model": result.model_revision,
            "parameter_view": result.representation, "reduction": result.reduction,
            "previous_mean": mean, "previous_variance": variance,
            "threshold": self.policy["threshold"], "draw": draw, "cost": result.cost})

    def open_task(self):
        """Start or restore the current task's growing pool of observed record IDs."""
        self.progress.setdefault("pool", [])

    def _admit(self, records):
        """Collate candidates, run admission and record its decision before returning selected IDs."""
        result = self.select_current(self.collate(records), self.model, self.spec.reduction)
        self.record_event("oasis_selection", selection=result.__dict__)
        return result.selected_ids

    def next_update(self):
        """For current_only, consume admitted arrivals through the common batch cursor.
        For historical_pool, add arrivals and sample candidates from all observed retained data.
        Persist pending selected IDs before training so resume reuses the same admission.
        Return the batch and supervised objective, or stop when no further batch is available.
        """
        if self.config.profile == "current_only":
            records = self.current_batch(self._admit)
        else:
            current_by_id = {r.sample_id: r for r in self.current}
            available = {r.sample_id: r for r in (*self.current, *self.replay.records)}
            records = tuple(available[i] for i in self.progress.get("pending_pool_ids", ()))
            # Empty admission consumes an opportunity, not an optimizer attempt.
            # An iterative scan also handles arbitrarily long rejection streaks.
            while not records and self.progress["cursor"] < len(self.current):
                start = self.progress["cursor"]
                incoming = self.current[start:start + self.spec.batch_size]
                self.progress["cursor"] += len(incoming)
                self.progress["pool"].extend(r.sample_id for r in incoming)
                pool = self.replay.records + tuple(current_by_id[i] for i in self.progress["pool"])
                candidates = tuple(self.rng.sample(pool, min(len(pool), self.spec.batch_size)))
                admitted = set(self._admit(candidates))
                records = tuple(r for r in candidates if r.sample_id in admitted)
                self.record_event("arrival", observed=[r.sample_id for r in incoming])
                self.progress["pending_pool_ids"] = [r.sample_id for r in records]
                self.boundary()
        if not records:
            return None
        origins = ["current" if r.task_id == self.task_id else "replay" for r in records]
        return records, origins, Objective(self.spec.reduction), None, None

    def observe(self, report, records):
        """Clear the pending pool selection after its update attempt has been consumed."""
        self.progress.pop("pending_pool_ids", None)

    def finish_task(self):
        """Keep observed history only for the historical_pool profile; current_only retains none."""
        if self.config.profile == "current_only":
            return ()
        by_id = {r.sample_id: r for r in self.current}
        return self.replay.records + tuple(by_id[i] for i in self.progress["pool"])
