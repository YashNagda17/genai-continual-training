"""Arm execution and reuse, seed averaging, winner rules, comparisons, and phase reports."""

from dataclasses import asdict
from functools import wraps
from pathlib import Path
import math
import re
import statistics
import time

from ..datasets.dataset_instruction import LANGUAGES
from ..models.model_mt5 import MT5SmallModel
from ..models.model_seq2seq import Seq2SeqSettings
from ..pipelines.revision import RevisionConfig, RevisionPipeline, load_pool
from ..utils.numerics import chrf, paired_bootstrap, read_json, rouge_l, summary_metrics, write_json

REVISION = ("selector", "replay_text", "repair_stop", "extra_random_steps")


def new_model(settings):
    return MT5SmallModel(Seq2SeqSettings(learning_rate=settings.learning_rate, seed=settings.seed), settings.device)


def total(values):
    """Sum numbers, or nested dictionaries of numbers, across seeds."""
    if isinstance(values[0], dict):
        return {key: total([value.get(key, 0) for value in values]) for key in {k for v in values for k in v}}
    return sum(values)


def run_arm(name, directory, settings, make, seeds=(None,), **fields):
    """Run or reuse each seed; a finished seed is never retrained. Several seeds are averaged."""
    runs = []
    for seed in seeds:
        output = Path(directory) / ("" if seed is None else f"seed-{seed}")
        if (output / "arm.json").is_file():
            runs.append(read_json(output / "arm.json"))
            continue
        print(f"arm {name} seed {seed}", flush=True)
        model, start = new_model(settings), time.perf_counter()
        try:
            result = make(model, output, settings.seed if seed is None else seed).run()
        finally:
            model.release()
        record = {"name": name, "seed": seed, "attempt": str(output), **fields, **result,
                  "wall_clock_seconds": time.perf_counter() - start,
                  "storage_bytes": sum(f.stat().st_size for f in output.rglob("*") if f.is_file())}
        write_json(output / "arm.json", record)
        runs.append(record)
    if len(runs) == 1:
        return runs[0]
    tasks = runs[0]["metrics"]
    return {**runs[0], "attempt": str(directory), "seed_runs": runs,
            "metrics": {t: {m: statistics.mean(r["metrics"][t][m] for r in runs) for m in tasks[t]} for t in tasks},
            "metric_std": {t: {m: statistics.stdev(r["metrics"][t][m] for r in runs) for m in tasks[t]} for t in tasks},
            "costs": total([r["costs"] for r in runs]), "steps": sum(r["steps"] for r in runs),
            "wall_clock_seconds": sum(r["wall_clock_seconds"] for r in runs),
            "storage_bytes": sum(r["storage_bytes"] for r in runs)}


def revision_arm(directory, name, settings, dataset, first=None, seeds=None, plan=None, plan_for_seed=None,
                 pipeline=RevisionPipeline, pool_path=None, fields=None, **config):
    """One revision arm. Second-stage arms start from the `first` arm's checkpoint and replay its pool (or the memory
    at `pool_path`), and run the screening seed unless asked for more. `pipeline` swaps in a baseline learner."""
    config = RevisionConfig(**config)
    pool = load_pool(pool_path or first["pool"]) if config.stage == "second" else ()

    def make(model, output, seed):
        return pipeline(settings, config, model, dataset, output, pool,
                                first and first["checkpoint"], plan_for_seed(seed) if plan_for_seed else plan, seed)
    seeds = (None,) if config.stage == "first" else tuple(seeds or settings.hindi_seeds[:1])
    blocks = (plan or plan_for_seed(seeds[0])).identity["arm"] if plan or plan_for_seed else "all"
    return run_arm(name, directory, settings, make, seeds, blocks=blocks, method=pipeline.__name__,
                   **asdict(config), **(fields or {}))


def revision_of(arm):
    return {key: arm[key] for key in REVISION}


def screening(arm):
    """The screening-seed run, so decisions compare arms on the same Hindi batches."""
    return arm.get("seed_runs", [arm])[0]


def score(metrics, task, name="chrf"):
    return metrics[task][name]


def forgetting(before, arm):
    return score(before, "english/validation/plain") - score(arm["metrics"], "english/validation/plain")


def gain(before, arm):
    return score(arm["metrics"], "hindi/validation/plain") - score(before, "hindi/validation/plain")


def lower_forgetting(before, arms):
    """Least English validation forgetting on the screening seed; ties go to the cheaper update."""
    return min(arms, key=lambda arm: (forgetting(before, screening(arm)), screening(arm)["costs"]["update_seconds"], arm["name"]))


def choose_winner(before, candidates, baseline):
    """Least forgetting among arms within 1 Hindi point of the baseline."""
    floor = gain(before, screening(baseline)) - 1
    return lower_forgetting(before, [a for a in candidates if gain(before, screening(a)) >= floor] or [baseline])


def beats(before, arm, others, margin=1.0):
    mine = forgetting(before, screening(arm))
    return all(mine <= forgetting(before, screening(other)) - margin for other in others)


def script_share_of_forgetting(before, plain, guarded):
    """G5: (S_en(X + guard) - S_en(X)) / (S_en(before) - S_en(X))."""
    return (guarded - plain) / (before - plain) if before > plain else None


def lead_baseline(dataset, language):
    """The article's first sentence: the no-training floor."""
    rows = dataset.view(f"{language}/validation")
    lead = [re.split(r"(?<=[.!?।])\s+", " ".join(r.text.split()), maxsplit=1)[0] for r in rows]
    return summary_metrics(lead, [r.summary for r in rows], language)


def gate_decision(forget, hindi_gain, english_score, lead_score):
    if english_score <= lead_score:
        return "fix_english"
    if hindi_gain < 1:
        return "fix_setup"
    return "stop_replay" if forget < 1 else "continue"


def matched_extra_steps(random_arm, drift_arm):
    """Extra random-replay updates that spend G1's extra measured time."""
    def spent(run):
        return run["costs"]["update_seconds"] + sum(run["costs"]["analysis_seconds"].values())
    random_run, drift_run = screening(random_arm), screening(drift_arm)
    per_step = random_run["costs"]["update_seconds"] / max(1, random_run["steps"])
    return max(0, math.ceil((spent(drift_run) - spent(random_run)) / per_step)) if per_step > 0 else 0


def compare_arms(left, right, split="test"):
    """Paired article bootstrap on the seeds both arms ran, averaged per article. Never used to select."""
    runs = [{run["seed"]: run for run in arm.get("seed_runs", [arm])} for arm in (left, right)]
    seeds = sorted(set(runs[0]) & set(runs[1]), key=str)
    result = {}
    for language in LANGUAGES:
        for name, metric in (("chrf", chrf), ("rouge_l", lambda p, t: 100 * rouge_l(p, t))):
            values = [[[metric(row["prediction"], row["target"]) for row in read_json(
                Path(arm[seed]["attempt"]) / "predictions" / f"{language}-{split}-plain.json")] for seed in seeds]
                for arm in runs]
            a, b = ([statistics.mean(column) for column in zip(*side)] for side in values)
            summary = paired_bootstrap(a, b)
            summary["counts"] = abs(summary["delta"]) >= 1 and (summary["low"] > 0 or summary["high"] < 0)
            summary["per_seed"] = {str(seed): statistics.mean(x - y for x, y in zip(sa, sb))
                                   for seed, sa, sb in zip(seeds, *values)}
            result[f"{language}/{name}"] = summary
    return {"left": left["name"], "right": right["name"], "split": split, "seeds": seeds, "metrics": result}


def brief(arms):
    """The report's view of each arm: scores, the four bills, and where its weights are."""
    keys = ("name", "seed", "steps", "metrics", "metric_std", "costs", "peak_memory_mb", "trainable_parameters",
            "storage_bytes", "wall_clock_seconds", "checkpoint", "retired_share", "blocks", "k_E", "k_G")
    return [{**{k: arm[k] for k in keys if k in arm}, "seeds": [r["seed"] for r in arm.get("seed_runs", [arm])]}
            for arm in arms]


def phase(index):
    """Run a phase once. Its saved report is reused, and its context passes to later phases."""
    def decorate(function):
        @wraps(function)
        def run(root, dataset, settings, context):
            directory = Path(root) / f"phase{index}"
            if (directory / "report.json").is_file():
                return read_json(directory / "report.json")["context"]
            directory.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            details, update = function(directory, dataset, settings, context)
            details["phase_seconds"] = time.perf_counter() - start
            write_json(directory / "report.json", {"phase": index, "details": details, "context": update})
            return update
        return run
    return decorate
