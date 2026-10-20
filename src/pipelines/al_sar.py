"""Signed similarity-aware retrieval and adaptive prefix freezing (Eqs. 4–9)."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from .pipeline_base import PipelineBase
from ..utils.contracts import FreezePlan, Objective, UpdateHooks
from ..utils.numerics import sample, softmax, tensor, unit_vectors


@dataclass(frozen=True)
class ALSARConfig:
    temperature: float
    frequency_scale: float
    ema: float
    refresh_every: int
    gradient_subset: str
    replacement: bool
    enable_sar: bool
    enable_al: bool
    count_profile: str


class ALSARPipeline(PipelineBase):
    def __init__(self, config, model=None, dataset=None):
        """Keep exposure, signed class similarity and freezing statistics as separate policy state."""
        super().__init__(config, model, dataset)
        self.policy = {"counts": {}, "classes": {}, "class_totals": {}, "similarity": {},
                       "fisher": None, "feature_square": None, "updates": 0,
                       "last_prefix": 0}

    def required_capabilities(self):
        """Request only the gradient/freeze features needed by the enabled SAR and aL components."""
        required = set()
        if self.config.enable_sar:
            required.add("subset_gradients")
        if self.config.enable_al:
            required.update(("layer_fisher", "feature_gradient", "freeze_aware"))
        return required


    def register(self, records):
        """Register observed IDs and labels with zero initial exposure; preserve existing counts."""
        for record in records:
            self.policy["classes"][record.sample_id] = record.class_id
            self.policy["counts"].setdefault(record.sample_id, 0.0)

    def reconcile(self, records):
        """Restrict per-example counts and classes to the supplied active membership.
        Recompute class totals for the hybrid active_population profile.
        Native class totals intentionally retain exposure history after individual eviction.
        """
        self.register(records)
        ids = {r.sample_id for r in records}
        self.policy["counts"] = {i: c for i, c in self.policy["counts"].items() if i in ids}
        self.policy["classes"] = {i: c for i, c in self.policy["classes"].items() if i in ids}
        if self.config.count_profile == "active_population":
            self.update_class_totals()

    def update_class_totals(self):
        """Sum active per-example counts by genuine class/group label for hybrid exposure."""
        totals = defaultdict(float)
        for sample_id, value in self.policy["counts"].items():
            label = self.policy["classes"][sample_id]
            if label is not None:
                totals[label] += value
        self.policy["class_totals"] = dict(totals)

    def effective_frequencies(self, records):
        """Equation 7 adds direct exposure to exposure through class-gradient similarity.
        Read S[historical_class, candidate_class] with the corresponding class total.
        Keep negative similarity contributions, which can reduce effective exposure.
        """
        direct = tensor([self.policy["counts"][r.sample_id] for r in records], self.device)
        labels = list(self.policy["class_totals"])
        if not labels or not records:
            return tuple(direct.tolist())
        totals = tensor([self.policy["class_totals"][label] for label in labels], self.device)
        similarity = tensor([[self.policy["similarity"].get(label, {}).get(r.class_id, 0.0)
                              for label in labels] for r in records], self.device)
        return tuple((direct + similarity @ totals).tolist())

    def probabilities(self, records):
        """Equation 8 applies one softmax over negative effective frequency.
        Lower exposure receives more probability; disabling SAR gives uniform retrieval.
        """
        if not records:
            return ()
        if not self.config.enable_sar:
            return tuple(1 / len(records) for _ in records)
        return softmax(-tensor(self.effective_frequencies(records), self.device)
                       / self.config.temperature)

    def retrieve_replay(self, records, count):
        """Sample the supplied eligible records using SAR probabilities and the policy RNG."""
        return sample(records, self.probabilities(records), count, self.config.replacement, self.rng)

    def decay_counts(self, batch_size):
        """Discount exposure by rho = 1 - batch_size / (k * active_record_count).
        Apply the discount to all active example counts, not only retrieved IDs.
        Native class totals decay independently; hybrid totals are rebuilt after increments.
        """
        counts = self.policy["counts"]
        rho = 1 - batch_size / (self.config.frequency_scale * len(counts))
        counts.update(zip(counts, (tensor(list(counts.values()), self.device) * rho).tolist()))
        if self.config.count_profile == "native_history":
            totals = self.policy["class_totals"]
            totals.update(zip(totals, (tensor(list(totals.values()), self.device) * rho).tolist()))
        return rho

    def increment_counts(self, consumed):
        """Increment every consumed occurrence, preserving multiplicity from repeated sampling.
        Update native class history directly or recompute the hybrid active class totals.
        """
        for record in consumed:
            self.policy["counts"][record.sample_id] += 1
        if self.config.count_profile == "active_population":
            self.update_class_totals()
        else:
            totals = self.policy["class_totals"]
            for record in consumed:
                if record.class_id is not None:
                    totals[record.class_id] = totals.get(record.class_id, 0.0) + 1

    def update_counts(self, consumed):
        """Apply decay before increments; callers choose the native-selection or hybrid-success clock."""
        if consumed:
            self.decay_counts(len(consumed))
            self.increment_counts(consumed)

    def information_per_cost(self, info):
        """Equation 4 estimates information per computation for each legal frozen prefix.
        Keep only the suffix Fisher sum and backward cost; forward cost is always paid.
        """
        fisher = tensor(self.policy["fisher"], self.device)
        costs = tensor(info.backward_costs, self.device)
        prefix_fisher = torch.cat((fisher.new_zeros(1), fisher.cumsum(0)))
        prefix_cost = torch.cat((costs.new_zeros(1), costs.cumsum(0)))
        boundaries = list(info.legal_prefixes)
        quality = ((fisher.sum() - prefix_fisher[boundaries])
                   / (info.forward_cost + costs.sum() - prefix_cost[boundaries]))
        return tuple(quality.tolist())

    def freezing_scores(self, feature_square, info):
        """Equation 5 compares saved prefix backward cost with estimated lost information.
        Use the best information-per-cost ratio and the current-to-EMA feature-gradient ratio.
        Return one batch-freezing score per legal prefix in the adapter's declared order.
        """
        quality = tensor(self.information_per_cost(info), self.device).max()
        fisher = tensor(self.policy["fisher"], self.device)
        costs = tensor(info.backward_costs, self.device)
        prefix_fisher = torch.cat((fisher.new_zeros(1), fisher.cumsum(0)))
        prefix_cost = torch.cat((costs.new_zeros(1), costs.cumsum(0)))
        scale = tensor(feature_square, self.device) / self.policy["feature_square"]
        boundaries = list(info.legal_prefixes)
        return tuple((quality * prefix_cost[boundaries] - scale * prefix_fisher[boundaries]).tolist())

    def plan_update(self, statistics, info):
        """Choose a prefix before backward using the previous Fisher and feature statistics.
        Cold start, zero denominator and scheduled refresh steps request full-layer updates.
        Otherwise maximize the batch-freezing score, breaking ties toward less freezing.
        """
        if not self.config.enable_al:
            return FreezePlan(0, False)
        feature_square = statistics["feature_norm_sq"]
        fisher, denominator = self.policy["fisher"], self.policy["feature_square"]
        refresh = (self.policy["updates"] + 1) % self.config.refresh_every == 0
        if fisher is None or not denominator or refresh:
            return FreezePlan(0, True)
        scores = self.freezing_scores(feature_square, info)
        best = min(zip(info.legal_prefixes, scores), key=lambda entry: (-entry[1], entry[0]))[0]
        return FreezePlan(best, False, scores)

    def hooks(self, info):
        """Give the model a pre-backward freeze planner and the requested SAR gradient subset."""
        return UpdateHooks(
            plan_update=(lambda statistics: self.plan_update(statistics, info)),
            gradient_subset=self.config.gradient_subset if self.config.enable_sar else "")

    def update_feature_statistics(self, feature_square):
        """Initialize or update the EMA of squared final-feature gradient norms, following Eq. 9."""
        previous = self.policy["feature_square"]
        feature = tensor(feature_square, self.device)
        self.policy["feature_square"] = (
            feature if previous is None else
            (1 - self.config.ema) * previous + self.config.ema * feature).item()

    def update_fisher(self, traces):
        """Initialize or update ordered layer Fisher EMAs from a full refresh observation."""
        previous = self.policy["fisher"]
        traces = tensor(traces, self.device)
        self.policy["fisher"] = (traces if previous is None else
            (1 - self.config.ema) * tensor(previous, self.device) + self.config.ema * traces).tolist()

    def update_similarities(self, records, gradients):
        # No unfrozen coordinates in the sampled subset: retain previous S.
        """Group ordered distinct-example cosine pairs by their class labels.
        Average each observed class pair, then apply one signed EMA update to that entry.
        Retain prior entries when no gradients or no eligible pair are available.
        """
        if gradients is None:
            return
        normalized = unit_vectors(tensor(gradients, self.device))
        similarities = normalized @ normalized.T
        grouped = defaultdict(list)
        for i, left in enumerate(records):
            for j, right in enumerate(records):
                if i != j:
                    grouped[left.class_id, right.class_id].append((i, j))
        for (left, right), pairs in grouped.items():
            rows, columns = zip(*pairs)
            average = similarities[list(rows), list(columns)].mean()
            row = self.policy["similarity"].setdefault(left, {})
            row[right] = ((1 - self.config.ema) * row.get(right, 0.0)
                          + self.config.ema * average).item()

    def observe_update(self, report, records, info):
        """Commit policy statistics only after a successful learner update.
        Use the pre-update freeze decision, refresh Fisher when required, then update similarity.
        Advance exposure here only for the hybrid; native retrieval already advanced its counts.
        """
        if not report.success:
            return
        diagnostics = report.diagnostics
        if self.config.enable_al:
            plan = self.plan_update(diagnostics, info)
            self.update_feature_statistics(diagnostics["feature_norm_sq"])
            if plan.refresh:
                self.update_fisher(diagnostics["layer_fisher"])
            self.policy["last_prefix"] = plan.prefix
        if self.config.enable_sar:
            self.update_similarities(records, diagnostics["class_gradients"])
        if self.config.count_profile == "active_population":
            self.update_counts(records)
        self.policy["updates"] += 1

    def balanced_insert(self, memory, arriving, capacity):
        """Insert each arrival before enforcing the memory capacity.
        Choose uniformly among largest classes, then uniformly among records in that class.
        Return the new membership so the lifecycle owner can commit it.
        """
        pool = list(memory)
        for record in arriving:
            pool.append(record)
            if len(pool) > capacity:
                counts = Counter(r.class_id for r in pool)
                largest = max(counts.values())
                label = self.rng.choice(sorted(c for c, n in counts.items() if n == largest))
                index = self.rng.choice([i for i, r in enumerate(pool) if r.class_id == label])
                pool.pop(index)
        return tuple(pool)

    def open_task(self):
        """Reconcile exposure state with the retained memory before processing new arrivals."""
        self.reconcile(self.replay.records)

    def next_update(self):
        """Native Algorithm 1 inserts one arrival before drawing a memory-only batch.
        Reconcile retained IDs, retrieve by SAR, then advance the native exposure clock.
        Return freeze hooks and origin labels; even a skipped optimizer attempt consumed this draw.
        """
        cursor = self.progress["cursor"]
        if cursor == len(self.current):
            return None
        # Algorithm 1 inserts one arrival before each memory-only retrieval.
        arriving = self.current[cursor:cursor + 1]
        memory = self.balanced_insert(self.replay.records, arriving, self.spec.replay.examples)
        self.replay = self.replay.proposal(memory)
        self.reconcile(memory)
        self.progress["cursor"] += 1
        count = self.spec.batch_size if self.config.replacement else min(self.spec.batch_size, len(memory))
        records = self.retrieve_replay(memory, count)
        self.update_counts(records)  # Native selection clock, including skipped updates.
        self.record_event("native_retrieval", ids=[r.sample_id for r in records],
                          observed=[r.sample_id for r in arriving])
        origins = ["current" if r.task_id == self.task_id else "replay" for r in records]
        return records, origins, Objective(self.spec.reduction), None, self.hooks(self.model.info)

    def observe(self, report, records):
        """Commit successful update observations using this learner's ordered layer metadata."""
        self.observe_update(report, records, self.model.info)

    def finish_task(self):
        """Keep the memory maintained online by native greedy balanced insertion."""
        return self.replay.records
