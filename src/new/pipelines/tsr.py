"""TSR baseline: task signatures, signature routing, and distillation from the routed era teacher.

Signature g_t = mean over probe batches of grad_B CE at the fixed base model with zero-B LoRA, the same for every
task. Routing w_j = softmax_j(cos(g_j, g_t) / tau) over past tasks; with English as the only past task, w = 1 and the
cosine is reported. Each update is a replay batch drawn uniformly from the English memory with probability alpha,
otherwise the next Hindi batch; both are distilled from the frozen English checkpoint (the argmax-routed teacher).
Training runs until every Hindi batch is used, so Hindi exposure matches the other arms."""

from dataclasses import replace
import random

import torch

from .revision import RevisionPipeline
from ..utils.numerics import cosine, write_json


class TSRPipeline(RevisionPipeline):
    def signature(self, language):
        rows = self.dataset.view(f"{language}/train")[:self.settings.tsr_probe_batches * self.settings.batch_size]
        totals = None
        for start in range(0, len(rows), self.settings.batch_size):
            part = rows[start:start + self.settings.batch_size]
            batch = self.dataset.collate(part)
            gradients = self.timed(lambda: self.model.probe_gradients(batch), "probe_pass", part, units=3)
            flat = torch.cat([gradient.detach() for gradient in gradients.values()]) * len(part)
            totals = flat if totals is None else totals + flat
        return totals / len(rows)

    def route(self):
        """The signatures and routing weights, computed once at the fixed initialization."""
        path = self.output / "signatures.json"
        if not path.is_file():
            self.model.settings = replace(self.model.settings, seed=0)  # theta_0 is the same for every task
            self.model.build()
            current = self.signature(self.config.language)
            similarity = {language: cosine(self.signature(language), current) for language in {r.language for r in self.pool}}
            weights = torch.softmax(torch.tensor(list(similarity.values())) / self.settings.tsr_routing_temperature, 0)
            write_json(path, {"cosine": similarity, "weights": dict(zip(similarity, weights.tolist()))})
        return path

    def total_steps(self, trained):
        """A fixed branch schedule from the arm's seed: -1 is a replay batch, k is the k-th new-language batch."""
        rng, self.schedule = random.Random(f"{self.seed}:tsr-branches"), []
        while len(self.schedule) - self.schedule.count(-1) < trained:
            self.schedule.append(-1 if rng.random() < self.settings.tsr_replay_probability
                                 else len(self.schedule) - self.schedule.count(-1))
        return len(self.schedule)

    def batch_for(self, step, rows, per_pass, trained):
        index, size = self.schedule[step], self.settings.batch_size
        if index < 0:
            return self.rng.sample(list(self.pool), min(size, len(self.pool)))
        return list(rows[(index % per_pass) * size:][:size])

    def loss_options(self, batch):
        return {"teacher": self.teacher, "kd_weight": self.settings.tsr_kd_weight,
                "kd_temperature": self.settings.tsr_kd_temperature}

    def run(self):
        self.route()
        self.teacher = type(self.model)(self.model.settings, self.model.device)
        self.teacher.load(self.first_checkpoint)
        try:
            result = super().run()
        finally:
            self.teacher.release()
        replayed = self.schedule.count(-1)
        return {**result, "replay_batches": replayed, "current_batches": len(self.schedule) - replayed,
                "signatures": str(self.output / "signatures.json")}
