"""The first-language checkpoint and its pool, then fixed-pool revision while the second language is learned
(X, A, B, G1, G2, G3; C2, D and E plans). English is first, except in the reverse order."""

from dataclasses import asdict, dataclass, replace
import math
import shutil

import torch

from .pipeline_base import PipelineBase
from ..datasets.dataset_instruction import LANGUAGES, Record
from ..utils.numerics import chrf, peak_mb, read_json, right_language, summary_metrics, write_json


@dataclass(frozen=True)
class RevisionConfig:
    stage: str = "second"          # first (from the base model; writes the pool) or second (replays that pool)
    language: str = "hindi"        # The language this stage learns
    selector: str = "random"       # none (X), random (A), loss (B), drift (G1), full (full-English reference)
    replay_text: str = "reference" # reference, or past (G2)
    repair_stop: bool = False      # G3
    extra_random_steps: int = 0    # G1's compute-matched control
    prefix: str = "plain"
    reference_free: bool = False   # G2 without references: the filter cannot compare with the reference
    replay_weight: float | None = None  # CSReL Eq. 17: mean(current) + weight x mean(replay); None is one combined mean


def load_pool(path):
    """The first stage's pool; its past summaries are never regenerated."""
    return tuple(Record(**row) for row in read_json(path))


class RevisionPipeline(PipelineBase):
    def __init__(self, settings, config, model, dataset, output, pool=(), first_checkpoint=None, plan=None, seed=0):
        super().__init__(settings, model, dataset, output, seed)
        self.config, self.pool, self.first_checkpoint, self.plan = config, tuple(pool), first_checkpoint, plan

    def run(self):
        s, c = self.settings, self.config
        rows, first = self.dataset.view(f"{c.language}/train"), c.stage == "first"
        per_pass = math.ceil(len(rows) / s.batch_size)
        trained = per_pass * (s.first_passes if first else s.second_passes)
        total = self.total_steps(trained)
        self.model.settings = replace(self.model.settings, seed=self.seed,
                                      warmup_steps=math.ceil(s.warmup_fraction * total))
        torch.manual_seed(self.seed)
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)
        if not self.restore():
            if first:
                self.model.build()
            else:
                self.model.load(self.first_checkpoint)
                if self.plan is not None:
                    self.model.apply_plan(self.plan)
            self.state = {"step": 0, "ranking": [], "scores": {}, "retired": [], "active": [], "ever_active": [],
                          "rank_cursor": 0, "full_cursor": 0, "refresh_at": 0, "filtered": [], "eligible": None,
                          "best": None, "validations": [], "trained": False}
            if c.replay_text == "past":
                self.filter_past()
        while self.state["step"] < total:
            step = self.state["step"]
            if not first:
                if c.selector in ("loss", "drift") and not self.state["ranking"] and step >= math.ceil(
                        trained * s.rank_warmup):
                    self.rank_pool()
                if c.repair_stop and step - self.state["refresh_at"] >= s.refresh_steps:
                    self.retire()
            batch = self.batch_for(step, rows, per_pass, trained)
            collated, options = self.dataset.collate(batch), self.loss_options(batch)
            self.timed(lambda: self.model.train_step(collated, **options), rows=batch, units=3)
            for record in batch:
                self.costs["tokens_per_language"][record.language] += len(record.source)
            self.state["step"] += 1
            if first and self.state["step"] % per_pass == 0:
                self.validate(self.state["step"] // per_pass)
            if self.state["step"] % s.checkpoint_every == 0 or self.state["step"] == total:
                self.checkpoint()
        if not self.state["trained"]:
            if first:
                self.finish_first()
            else:
                self.model.save(self.output / c.language, record={"plan": None if self.plan is None else asdict(self.plan)})
            self.state["trained"] = True
            self.checkpoint()
        metrics = self.evaluate()
        shutil.rmtree(self.output / "resume.prev", ignore_errors=True)
        retired, ever = self.state["retired"], self.state["ever_active"]
        return {"checkpoint": str(self.output / c.language), "pool": str(self.output / "pool.json"), "metrics": metrics,
                "steps": total, "costs": self.costs, "peak_memory_mb": peak_mb(self.model.device),
                "trainable_parameters": self.model.trainable_parameters(), "ranking": self.state["ranking"],
                "filtered": self.state["filtered"], "retired": retired, "retired_share": len(retired) / max(1, len(ever)),
                "validations": self.state["validations"], "plan": None if self.plan is None else asdict(self.plan)}

    def total_steps(self, trained):
        return trained + self.config.extra_random_steps

    def batch_for(self, step, rows, per_pass, trained):
        """The step's new-language rows, plus replayed rows in the second stage."""
        size = self.settings.batch_size
        if step < trained:
            current = rows[(step % per_pass) * size:][:size]
        else:  # Extra random-replay updates cycle through the new-language rows.
            start = (step - trained) * size % len(rows)
            current = [rows[(start + i) % len(rows)] for i in range(min(len(rows), size))]
        return list(current) + ([] if self.config.stage == "first" else self.replay_rows())

    def loss_options(self, batch):
        options = {"token_weights": self.token_weights(batch)} if self.config.repair_stop else {}
        if self.config.replay_weight is not None:
            replayed = [r.language != self.config.language for r in batch]
            current, old = replayed.count(False), replayed.count(True)
            options["row_weights"] = torch.tensor([self.config.replay_weight / old if flag else 1 / current
                                                   for flag in replayed])
        return options

    def eligible(self):
        ids = self.state["eligible"]
        return tuple(r for r in self.pool if ids is None or r.id in ids)

    def filter_past(self):
        """G2: keep right-script past summaries of 5+ words whose chrF to the reference is not in the bottom 20%.
        Without references, only the script and length rules apply."""
        scores = {r.id: chrf(r.practice["answer_text"], r.summary) for r in self.pool}
        cut = 0 if self.config.reference_free else math.floor(len(self.pool) * self.settings.past_filter_fraction)
        bottom = set(sorted(scores, key=lambda i: (scores[i], i))[:cut])
        eligible = [r.id for r in self.pool if r.id not in bottom and right_language(r.practice["answer_text"], r.language)
                    and len(r.practice["answer_text"].split()) >= self.settings.past_min_words]
        self.state.update(eligible=eligible, filtered=[r.id for r in self.pool if r.id not in set(eligible)])

    def rank_pool(self):
        """B ranks by the rise in loss since the English checkpoint; G1 by how far its writing drifted."""
        pool = self.eligible()
        if self.config.selector == "loss":
            losses = self.token_losses(pool, "pool_scoring")
            scores = [sum(loss) / len(loss) - r.practice["loss_before"] for r, loss in zip(pool, losses)]
        else:
            texts, _ = self.generate(pool, "pool_generation")
            scores = [1 - chrf(text, r.practice["answer_text"]) / 100 if right_language(text, r.language) else 1.0
                      for r, text in zip(pool, texts)]
        self.state["scores"] = {r.id: score for r, score in zip(pool, scores)}
        self.state["ranking"] = sorted(self.state["scores"], key=lambda i: (-self.state["scores"][i], i))

    def replay_rows(self):
        """Half the slots from the top of the ranking (after warm-up), the rest at random from the pool."""
        s, c = self.settings, self.config
        if c.selector == "none":
            return []
        if c.selector == "full":
            full, start = self.dataset.view(f"{self.pool[0].language}/train"), self.state["full_cursor"]
            self.state["full_cursor"] += s.batch_size
            return [full[(start + i) % len(full)] for i in range(s.batch_size)]
        pool, ranking = self.eligible(), self.state["ranking"]
        retired, ids = set(self.state["retired"]), {r.id for r in pool}
        available = {i for i in ranking if i in ids and i not in retired}
        chosen = []
        while len(chosen) < min(math.ceil(s.top_fraction * s.replay_per_batch), len(available)):
            candidate = ranking[self.state["rank_cursor"] % len(ranking)]
            self.state["rank_cursor"] += 1
            if candidate in available and candidate not in chosen:
                chosen.append(candidate)
        self.state["active"] = list(chosen)
        self.state["ever_active"] = sorted(set(self.state["ever_active"]) | set(chosen))
        chosen += self.rng.sample([r.id for r in pool if r.id not in chosen], s.replay_per_batch - len(chosen))
        by_id = {r.id: r for r in pool}
        return [by_id[i].past() if c.replay_text == "past" else by_id[i] for i in chosen]

    def retire(self):
        """G3: an active example whose new summary is in the right language with similarity >= 0.9 leaves replay."""
        by_id = {r.id: r for r in self.pool}
        rows = [by_id[i] for i in self.state["active"]]
        texts, _ = self.generate(rows, "stop_regeneration")
        for record, text in zip(rows, texts):
            if right_language(text, record.language) and chrf(text, record.practice["answer_text"]) / 100 >= self.settings.stop_similarity:
                self.state["retired"].append(record.id)
        self.state["refresh_at"] = self.state["step"]

    def token_weights(self, batch):
        """G3: w_t = 1 + beta max(0, log p_before - log p_now) on replayed rows, scaled to average 1 per row."""
        key = "past_baseline" if self.config.replay_text == "past" else "baseline"

        def weights(nll, mask):
            before, replayed = torch.zeros(nll.shape), torch.zeros(len(batch), dtype=torch.bool)
            for index, record in enumerate(batch):
                if record.language != self.config.language:
                    baseline = record.practice[key]
                    before[index, :len(baseline)] = torch.tensor(baseline)
                    replayed[index] = True
            before, replayed = before.to(nll.device), replayed.to(nll.device)
            w = (1 + self.settings.repair_beta * (nll - before).clamp_min(0)) * mask
            w = w * mask.sum(-1, keepdim=True) / w.sum(-1, keepdim=True)
            return torch.where(replayed[:, None], w, mask.float())
        return weights

    def validate(self, epoch):
        """First stage: score each pass on validation and keep the best one."""
        language = self.config.language
        rows = self.dataset.view(f"{language}/validation")
        texts, _ = self.generate(rows, "validation_generation")
        scores = summary_metrics(texts, [r.summary for r in rows], language)
        self.state["validations"].append({"pass": epoch, **scores})
        if self.state["best"] is None or scores["chrf"] > self.state["best"]["scores"]["chrf"]:
            self.model.save(self.output / "best")
            self.state["best"] = {"pass": epoch, "scores": scores}
        print(f"{language} pass {epoch}: validation chrF {scores['chrf']:.2f}", flush=True)

    def finish_first(self):
        """Pin the best pass, then write a past summary and token losses for every pool article."""
        best = self.state["best"]
        self.model.load(self.output / "best")
        self.model.save(self.output / self.config.language, record={"writer": "first_checkpoint", "prefix": self.config.prefix,
                                                         "passes": best["pass"], "validation": best["scores"]})
        pool = self.dataset.view(f"{self.config.language}/pool")
        texts, ids = self.generate(pool, "pool_generation")
        baselines = self.token_losses(pool, "pool_scoring")
        written = [replace(r, practice={"answer_text": text, "answer_ids": answer, "baseline": baseline,
                                        "loss_before": sum(baseline) / len(baseline), "past_baseline": []})
                   for r, text, answer, baseline in zip(pool, texts, ids, baselines)]
        nonempty = [r for r in written if r.practice["answer_ids"]]
        for record, losses in zip(nonempty, self.token_losses([r.past() for r in nonempty], "pool_scoring")):
            record.practice["past_baseline"] = losses
        write_json(self.output / "pool.json", [asdict(r) for r in written])

    def evaluate(self):
        """Validation and test, both languages, plain and with G5's script guard. Predictions are saved."""
        metrics = {}
        for split in ("validation", "test"):
            for language in LANGUAGES:
                rows = self.dataset.view(f"{language}/{split}")
                for guarded in (False, True):
                    key = f"{language}/{split}/{'guarded' if guarded else 'plain'}"
                    texts, _ = self.generate(rows, "guard_generation" if guarded else "evaluation_generation",
                                             guard=language if guarded else None)
                    metrics[key] = summary_metrics(texts, [r.summary for r in rows], language)
                    write_json(self.output / "predictions" / f"{key.replace('/', '-')}.json",
                               [{"id": r.id, "target": r.summary, "prediction": t} for r, t in zip(rows, texts)])
            print(f"{self.output.name} {split}: " + ", ".join(
                f"{k} chrF {v['chrf']:.2f}" for k, v in metrics.items() if f"/{split}/" in k), flush=True)
        return metrics
