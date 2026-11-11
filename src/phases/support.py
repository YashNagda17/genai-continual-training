"""Shared execution and lifecycle; each phase supplies its experiment decisions."""

from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
import copy
import json
import math
import random
import re
import statistics
import time

from ..pipelines.revision import RevisionConfig, RevisionPipeline
from ..pipelines.adapter_room import AdapterRoomConfig, AdapterRoomPipeline
from ..utils.numerics import summary_metrics
from ..utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec, fingerprint
from ..utils.state import read_json, write_json
from .artifacts import PhaseClock, accuracies_of, arm_record, ensure_final, finish_arm, finished_arm, remember_arm, save_report
from .settings import generation_spec

# Pause after every trained arm (English or Hindi) so the machine cools before the next one.
# Smoke runs and tiny test models skip it. Cached arms are reused without training, so they do not pause.
COOLDOWN_SECONDS = 0
CHECKPOINT_EVERY = 100


def run_spec(stage, settings, passes, arm, dataset, model, extra_steps=0):
    rows = tuple(dataset.training_view(stage)) if stage != "analysis" else ()
    updates = math.ceil(len(rows) / settings.batch_size) * passes + extra_steps
    bound = max(1, updates) * (2 * settings.batch_size + settings.replay_per_batch) * (
        model.settings.max_source_length + model.settings.max_target_length)
    return RunSpec((stage,), settings.seed, settings.batch_size, max(1, updates), max(1, updates), bound, passes,
                   "repeat_admitted", "per_example_mean", ReplayBudget(len(dataset.pool_view()), 50_000_000, 50_000_000),
                   ResourceBudget(64_000_000, 0), 1 if settings.protocol == "smoke" else CHECKPOINT_EVERY,
                   {"protocol": settings.protocol, "arm": arm, "training_rows": len(rows)}, {})


def execute(pipeline, model, dataset, output, stage, settings, passes, arm):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = run_spec(stage, settings, passes, arm, dataset, model, getattr(pipeline.config, "extra_random_steps", 0))
    if model.pretrained:
        # Linear warm-up over the first part of this stage's updates, then a constant rate.
        model.settings = replace(model.settings, seed=settings.seed,
                                 warmup_steps=math.ceil(settings.warmup_fraction * spec.max_steps) if passes else 0)
    info = model.info
    accelerator = 0 if model.device.type == "cpu" else info.accelerator_bytes
    spec = replace(spec, resources=ResourceBudget(info.cpu_bytes + spec.resources.cpu_bytes, accelerator))
    result = pipeline.run(RunContext(model, dataset, spec, output))
    return result, read_json(output / "run.json")


def run_pipeline(pipeline, model_factory, dataset, directory, stage, settings, passes, name,
                 describe=None, plan_factory=None, seeds=None, **fields):
    """One implementation for training/analysis, caching, per-seed attempts, and exports.

    Scientific Hindi arms run the screening seed (the first Hindi seed) unless ``seeds`` asks for more.
    Finished seeds are reused, so confirming an arm later trains only the missing seeds."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if stage == "hindi" and settings.protocol == "scientific":
        seeds = tuple(seeds or settings.hindi_seeds[:1])
        if seeds[0] != settings.hindi_seeds[0] or not set(seeds) <= set(settings.hindi_seeds):
            raise ValueError("Hindi seeds must start with the screening seed and come from hindi_seeds")
        runs = []
        for seed in seeds:
            child = copy.copy(pipeline)
            child.policy = copy.deepcopy(pipeline.policy)
            child.rng = random.Random(seed)
            if plan_factory is not None:
                child.adapter_plan = plan_factory(seed)
            runs.append(single_run(child, model_factory, dataset, directory / f"seed-{seed}", stage,
                                   replace(settings, seed=seed), passes, f"{name}-seed-{seed}", describe, fields))
        record = aggregate_runs(name, directory, runs)
        write_json(directory / "run.json", {"status": "completed", "seeds": list(seeds)})
        return remember_arm(directory, record)
    return single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields)


def single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields):
    signature = fingerprint({"version": 2, "settings": asdict(settings), "dataset": dataset.fingerprint,
                             "config": asdict(pipeline.config), "inputs": pipeline.input_identity(), "name": name})
    saved = finished_arm(directory, signature)
    if saved is not None:
        return ensure_final(directory.parent, saved)
    model = model_factory()
    pretrained = bool(getattr(model, "pretrained", False))
    try:
        result, manifest = execute(pipeline, model, dataset, directory, stage, settings, passes, name)
        checkpoint = (directory / pipeline.policy["english_checkpoint"] if stage == "english" else
                      directory / "hindi" if stage == "hindi" else "")
        extra = describe(pipeline, directory) if describe else {}
        checkpoint = extra.pop("checkpoint", checkpoint)
        record = arm_record(name, directory, checkpoint, result, manifest, pipeline,
                            **fields, **extra, signature=signature)
        if record["status"] != "completed":
            raise ValueError(f"{name} did not complete")
    finally:
        model.release()
    finished = finish_arm(directory.parent, directory, record)
    cool_down(settings, stage, pretrained, device=model.device)
    return finished


def cool_down(settings, stage, pretrained=True, *, device=None):
    """Sleep after a training run of a real model, once its weights and report are saved.
    Tiny test models, smoke runs and CUDA hosts do not apply the local cooldown."""
    if device is not None and device.type == "cuda":
        return
    if (stage in ("english", "hindi") and pretrained and settings.protocol != "smoke"
            and COOLDOWN_SECONDS > 0):
        print(f"Cooling down for {COOLDOWN_SECONDS} s after the {stage} run", flush=True)
        time.sleep(COOLDOWN_SECONDS)


def aggregate_runs(name, directory, runs):
    if any(run["status"] != "completed" for run in runs):
        raise ValueError("Cannot aggregate missing or failed seeds")
    metrics = {task: {metric: statistics.mean(run["metrics"][task][metric] for run in runs)
                      for metric in scores} for task, scores in runs[0]["metrics"].items()}
    record = {**runs[0], "name": name, "attempt": str(directory), "metrics": metrics, "seed_runs": runs,
              "steps": sum(run["steps"] for run in runs),
              "wall_clock_seconds": sum(run["wall_clock_seconds"] for run in runs),
              "update_seconds": sum(run["update_seconds"] for run in runs)}
    record["accuracies"] = accuracies_of(metrics)
    record["epochs"] = {"requested": runs[0]["epochs"]["requested"],
                        "completed": statistics.mean(run["epochs"]["completed"] for run in runs), "steps": record["steps"]}
    record["storage_bytes"] = sum(run["storage_bytes"] for run in runs)
    record["costs"] = {"update_seconds": record["update_seconds"],
                       "forward_pass_units": sum(run["costs"]["forward_pass_units"] for run in runs),
                       "tokens_per_language": {language: sum(run["costs"]["tokens_per_language"].get(language, 0)
                                                             for run in runs) for language in ("english", "hindi")},
                       "analysis_seconds": {category: sum(run["costs"]["analysis_seconds"].get(category, 0) for run in runs)
                                            for category in {key for run in runs for key in run["costs"]["analysis_seconds"]}}}
    record["final_models"] = {str(run["seed"]): run.get("final_model") for run in runs}
    record["time"] = {"wall_clock_seconds": record["wall_clock_seconds"],
                       "gpu_hours": sum(run["time"].get("gpu_hours") or 0 for run in runs),
                       "peak_memory_mb": max((run["time"].get("peak_memory_mb") or 0 for run in runs))}
    record["metric_std"] = {task: {metric: statistics.stdev(run["metrics"][task][metric] for run in runs)
                                     if len(runs) > 1 else None for metric in scores} for task, scores in metrics.items()}
    return record


def score(metrics, task, name="chrf"):
    try:
        return metrics[task][name]
    except KeyError as error:
        raise ValueError(f"Missing {name} for {task}") from error


def screening(arm):
    """The arm's screening-seed run. Every arm has it, so decisions compare arms on the same Hindi batches."""
    runs = arm.get("seed_runs")
    return runs[0] if runs else arm


def confirmed(arm):
    return len(arm.get("seed_runs", ())) > 1


def forgetting(before, arm):
    return score(before, "english/validation/plain") - score(arm["metrics"], "english/validation/plain")


def gain(before, arm):
    return score(arm["metrics"], "hindi/validation/plain") - score(before, "hindi/validation/plain")


def choose_winner(before, candidates, baseline):
    """Lowest English validation forgetting among arms within 1 Hindi point of the baseline."""
    if not candidates:
        raise ValueError("Winner selection needs at least one arm")
    floor = gain(before, screening(baseline)) - 1
    eligible = [arm for arm in candidates if gain(before, screening(arm)) >= floor]
    pool = eligible or [baseline]
    return lower_forgetting(before, pool)


def lower_forgetting(before, arms):
    """Compared on the screening seed. Ties go to the cheaper arm by update time."""
    def key(arm):
        run = screening(arm)
        return forgetting(before, run), run.get("update_seconds", run.get("wall_clock_seconds")) or 0, arm["name"]
    return sorted(arms, key=key)[0]


def beats(before, arm, others, margin=1.0):
    """True when ``arm`` forgets at least ``margin`` points less English than every other arm, on the screening seed."""
    mine = forgetting(before, screening(arm))
    return all(mine <= forgetting(before, screening(other)) - margin for other in others)


def script_share_of_forgetting(before, plain, guarded):
    drop = before - plain
    if drop <= 0:
        return None
    return (guarded - plain) / drop


def lead_sentence(text):
    """The article's first sentence: the no-training floor every checkpoint must beat."""
    text = " ".join(text.split())
    return re.split(r"(?<=[.!?\u0964])\s+", text, maxsplit=1)[0]


def lead_baseline(dataset, language, split="validation"):
    named = dataset.views.get(f"{language}/{split}")
    rows = tuple(named if named is not None else
                 (row for row in dataset.views.get(split, ()) if row.task_id == language))
    if not rows:
        raise ValueError(f"No {language} {split} rows for the lead-1 floor")
    return summary_metrics([lead_sentence(row.inputs["text"]) for row in rows],
                           [row.target["text"] for row in rows], language)


def gate_decision(forget, hindi_gain, english_score=None, lead_score=None):
    if english_score is not None and lead_score is not None and english_score <= lead_score:
        return "fix_english"
    if hindi_gain < 1:
        return "fix_setup"
    if forget < 1:
        return "stop_replay"
    return "continue"


def read_context(root, dataset, settings, phases):
    """Context saved by the given phases, applied in the order given (the run order)."""
    context = {}
    for index in phases:
        path = Path(root) / f"phase{index}" / "report.json"
        if path.is_file():
            report = read_json(path)
            identity = fingerprint({"version": 2, "phase": index, "settings": asdict(settings),
                                    "dataset": dataset.fingerprint})
            if report.get("identity") != identity:
                raise ValueError(f"Phase {index} report belongs to different settings or data")
            context.update(report["context"])
    return context


def phase(index):
    """Keep phase lifecycle/report handling in one place, outside experiment decisions."""
    def decorate(function):
        @wraps(function)
        def run(root, dataset, model_factory, settings, context):
            directory = Path(root) / f"phase{index}"
            identity = fingerprint({"version": 2, "phase": index, "settings": asdict(settings),
                                    "dataset": dataset.fingerprint})
            report = directory / "report.json"
            if report.is_file():
                saved = read_json(report)
                if saved.get("identity") != identity or saved.get("status") != "completed":
                    raise ValueError(f"Phase {index} settings changed; choose a new output directory")
                if not all((directory / name).is_file() for name in ("report.csv", "report.md")):
                    save_report(directory, index, saved["details"], saved["context"])
                return saved["context"]
            directory.mkdir(parents=True, exist_ok=True)
            clock = PhaseClock(directory)
            if clock.body.get("identity", identity) != identity:
                raise ValueError(f"Phase {index} progress belongs to different settings")
            clock.update(identity=identity, status="running", phase=index)
            def log(status, error=None):
                with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"phase": index, "status": status, "seconds": clock.seconds(), "error": error}) + "\n")
            log("running")
            try:
                result = function(directory, dataset, model_factory, settings, context, clock)
            except BaseException as error:
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                clock.update(status=status, error=f"{type(error).__name__}: {error}")
                log(status, str(error))
                raise
            else:
                clock.update(status="completed")
                log("completed")
                return result
        return run
    return decorate


def revision_config(settings, **overrides):
    label = {} if "prefix" in overrides or not settings.task_prefix else {"prefix": "summarize"}
    return RevisionConfig(hindi_batch=settings.batch_size, replay_per_batch=settings.replay_per_batch,
                          hindi_passes=settings.hindi_passes, analysis_batch=settings.analysis_batch,
                          generation=generation_spec(settings), **label, **overrides)


def hindi_pipeline(settings, pool, checkpoint, **overrides):
    return RevisionPipeline(revision_config(settings, stage="hindi", **overrides),
                            pool=pool, english_checkpoint=checkpoint)


def run_hindi(directory, name, pipeline, dataset, model_factory, settings, plan_factory=None, seeds=None, **fields):
    fields = {"selector": pipeline.config.selector, "replay_text": pipeline.config.replay_text,
              "repair_stop": pipeline.config.repair_stop, **fields}
    return run_pipeline(pipeline, model_factory, dataset, directory, "hindi", settings,
                        settings.hindi_passes, name, plan_factory=plan_factory, seeds=seeds, **fields)


BLOCK_PLANS = {"random": "all", "late": "late", "early_late": "early_late"}


def revision_pipeline(settings, pool, checkpoint, revision):
    """Rebuild a revision arm from its record. The config matches the screening run, so its seed is reused."""
    return hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                          replay_text=revision["replay_text"], repair_stop=revision["repair_stop"],
                          extra_random_steps=revision.get("extra_random_steps", 0))


def combined_pipeline(settings, pool, checkpoint, revision, blocks):
    """D: the best revision on the best blocks."""
    pipeline = revision_pipeline(settings, pool, checkpoint, revision)
    pipeline.adapter_plan = AdapterRoomPipeline(room_config(settings)).adapter_plan(BLOCK_PLANS[blocks])
    return pipeline


def matched_extra_steps(random_arm, drift_arm):
    """Extra random-replay updates that spend G1's measured extra time (scoring, writing, and updates)."""
    def total(run):
        return (run.get("update_seconds") or 0) + sum((run.get("costs") or {}).get("analysis_seconds", {}).values())
    random_run, drift_run = screening(random_arm), screening(drift_arm)
    per_step = (random_run.get("update_seconds") or 0) / max(1, random_run.get("steps") or 0)
    if per_step <= 0:
        return 0
    return max(0, math.ceil((total(drift_run) - total(random_run)) / per_step))


def room_config(settings):
    return AdapterRoomConfig(settings.stability_threshold, seed=settings.seed, sample_size=settings.sample_size,
        analysis_batch=settings.analysis_batch, encoder_blocks=settings.encoder_blocks,
        decoder_blocks=settings.decoder_blocks, new_room_budget=settings.new_room_budget,
        encoder_block_cost=settings.encoder_block_cost, decoder_block_cost=settings.decoder_block_cost)


ROOM_ARMS = ("room_placed", "room_random", "room_spread")


def room_plans(settings, context, seed=None, names=ROOM_ARMS):
    """Plans for the named E arms. E4 (``room_placed_slow``) and E5 are requested separately, when they run."""
    room = AdapterRoomPipeline(room_config(replace(settings, seed=settings.seed if seed is None else seed)))
    shared = room.adapter_plan(BLOCK_PLANS[context["best_blocks"]])
    plans = [room.adapter_plan(name, maps=context["maps"], shared=shared) for name in names]
    if any(plan is None for plan in plans):
        return (), room.policy.get("room_status", "skipped")
    try:
        return room.validate_room_plans(plans), None
    except ValueError as error:
        return (), str(error)
