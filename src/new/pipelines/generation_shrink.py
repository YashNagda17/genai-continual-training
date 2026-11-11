"""F: choose the adapter rank by what the shrunk model writes on validation, against the energy rule."""

import torch

from .pipeline_base import PipelineBase
from ..datasets.dataset_instruction import LANGUAGES
from ..utils.numerics import energy_rank, summary_metrics


class GenerationShrinkPipeline(PipelineBase):
    def __init__(self, settings, model, dataset, output, checkpoint, full_rank, seed=0):
        super().__init__(settings, model, dataset, output, seed)
        self.checkpoint, self.full_rank = checkpoint, full_rank

    def scores(self):
        result = {}
        for language in LANGUAGES:
            rows = self.dataset.view(f"{language}/validation")
            texts, _ = self.generate(rows, "shrink_generation")
            result[language] = summary_metrics(texts, [r.summary for r in rows], language)
        return result

    def candidate(self, rank):
        """Every candidate starts from the original checkpoint, never from the previous truncation."""
        self.model.load(self.checkpoint)
        self.model.shrink(rank)

    def run(self):
        s = self.settings
        self.model.load(self.checkpoint)
        deltas = self.model.deltas()
        shapes = [list(delta.shape) for delta in deltas.values()]
        k_E = energy_rank([torch.linalg.svdvals(delta) for delta in deltas.values()], s.energy_fraction)
        baseline = self.scores()
        table, k_G, rank, direction = [], None, k_E, None
        while rank is not None:  # Start at k_E; step down while passing, or up until one passes.
            self.candidate(rank)
            scores = self.scores()
            passed = all(scores[l][m] >= baseline[l][m] - s.shrink_tolerance
                         for l in LANGUAGES for m in ("rouge_l", "chrf", "right_language"))
            table.append({"rank": rank, "scores": scores, "passed": passed,
                          "adapter_tensor_bytes": sum(min(rank, *shape) * sum(shape) * 4 for shape in shapes)})
            direction = direction or ("down" if passed else "up")
            k_G = rank if passed else k_G
            tested = {row["rank"] for row in table}
            if direction == "down":
                lower = sorted((k for k in set(s.shrink_grid) if k < rank and k not in tested), reverse=True)
                rank = lower[0] if passed and lower else None
            else:
                higher = sorted(k for k in {*s.shrink_grid, self.full_rank} if rank < k <= self.full_rank and k not in tested)
                rank = higher[0] if not passed and higher else None
        if k_G is None:
            raise ValueError("No rank, including the full rank, passed the generation test")
        self.candidate(k_G)
        self.model.save(self.output / "shrunk-adapter", record={"k_E": k_E, "k_G": k_G, "table": table})
        chosen = next(row for row in table if row["rank"] == k_G)
        return {"checkpoint": str(self.output / "shrunk-adapter"), "k_E": k_E, "k_G": k_G, "full_rank": self.full_rank, "baseline": baseline,
                "table": table, "metrics": {f"{l}/shrink": v for l, v in chosen["scores"].items()}, "costs": self.costs}
