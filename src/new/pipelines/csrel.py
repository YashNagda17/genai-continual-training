"""CSReL baseline: the replay memory chosen by greedy rounds of reducible loss.

ReL(i) = loss(i; theta_S) - loss(i; theta_D). The reference theta_D is the English checkpoint, trained on every English
candidate. Each round fits a fresh LoRA subset model theta_S on the rows selected so far (round 1 uses the untrained
initialization), scores the remaining candidates, and adds the round's top rows. The Hindi learner then replays this
memory uniformly with CSReL's separately weighted loss (RevisionConfig.replay_weight)."""

from dataclasses import asdict, replace

from .pipeline_base import PipelineBase
from ..utils.numerics import read_json, write_json


class CSReLPipeline(PipelineBase):
    def __init__(self, settings, model, dataset, output, reference_checkpoint, seed=0):
        super().__init__(settings, model, dataset, output, seed)
        self.reference_checkpoint = reference_checkpoint

    def example_losses(self, rows, category):
        """Per-example mean target NLL, the same reduction for both scoring models."""
        return [sum(losses) / len(losses) for losses in self.token_losses(rows, category)]

    def fit_subset(self, rows):
        s = self.settings
        self.model.settings = replace(self.model.settings, seed=self.seed, warmup_steps=0)
        self.model.build()
        for _ in range(s.csrel_subset_passes if rows else 0):
            for start in range(0, len(rows), s.batch_size):
                part = rows[start:start + s.batch_size]
                batch = self.dataset.collate(part)
                self.timed(lambda: self.model.train_step(batch), "subset_training", part, units=3)

    def run(self):
        s, path = self.settings, self.output / "selection.json"
        candidates = self.dataset.view("english/train")
        by_id = {r.id: r for r in candidates}
        state = read_json(path) if path.is_file() else {"reference": None, "selected": [], "rounds": [], "costs": None}
        self.costs = state["costs"] or self.costs
        if state["reference"] is None:
            self.model.load(self.reference_checkpoint)
            state["reference"] = dict(zip(by_id, self.example_losses(candidates, "reference_scoring")))
        quotas = [s.pool_rows // s.csrel_rounds] * s.csrel_rounds
        quotas[-1] += s.pool_rows - sum(quotas)
        for quota in quotas[len(state["rounds"]):]:  # A finished round is never repeated.
            self.fit_subset([by_id[i] for i in state["selected"]])
            chosen = set(state["selected"])
            rest = [r for r in candidates if r.id not in chosen]
            scores = {r.id: loss - state["reference"][r.id]
                      for r, loss in zip(rest, self.example_losses(rest, "subset_scoring"))}
            picked = sorted(scores, key=lambda i: (-scores[i], i))[:quota]
            state["selected"] += picked
            state["rounds"].append({"quota": quota, "mean_score": sum(scores[i] for i in picked) / max(1, len(picked))})
            state["costs"] = self.costs
            write_json(path, state)
        memory = [asdict(by_id[i]) for i in sorted(state["selected"], key=lambda i: by_id[i].arrival)]
        write_json(self.output / "memory.json", memory)
        return {"pool": str(self.output / "memory.json"), "rounds": state["rounds"], "costs": self.costs,
                "selection_seconds": sum(self.costs["analysis_seconds"].values()),
                "overlap_with_random_pool": len(set(state["selected"]) & set(self.dataset.ids["english/pool"]))
                / max(1, len(state["selected"]))}
