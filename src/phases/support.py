"""Shared execution and lifecycle; each phase supplies its experiment decisions."""

from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
import copy
import json
import math
import random
import statistics

from ..pipelines.revision import RevisionConfig, RevisionPipeline
from ..pipelines.adapter_room import AdapterRoomConfig, AdapterRoomPipeline
from ..utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec, fingerprint
from ..utils.state import read_json, write_json
from .artifacts import PhaseClock, accuracies_of, arm_record, ensure_final, finish_arm, finished_arm, remember_arm, save_report
from .settings import generation_spec


def run_spec(stage, settings, passes, arm, dataset, model, extra_steps=0):
    rows = tuple(dataset.training_view(stage)) if stage != "analysis" else ()
    updates = math.ceil(len(rows) / settings.batch_size) * passes + extra_steps
    bound = max(1, updates) * (2 * settings.batch_size + settings.replay_per_batch) * (
        model.settings.max_source_length + model.settings.max_target_length)
    return RunSpec((stage,), settings.seed, settings.batch_size, max(1, updates), max(1, updates), bound, passes,
                   "repeat_admitted", "per_example_mean", ReplayBudget(len(dataset.pool_view()), 50_000_000, 50_000_000),
                   ResourceBudget(64_000_000, 0), 1,
                   {"protocol": settings.protocol, "arm": arm, "training_rows": len(rows),
                    "preregistration": settings.preregistration_hash}, {})


def execute(pipeline, model, dataset, output, stage, settings, passes, arm):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if model.pretrained:
        model.settings = replace(model.settings, seed=settings.seed)
    spec = run_spec(stage, settings, passes, arm, dataset, model, getattr(pipeline.config, "extra_random_steps", 0))
    info = model.info
    accelerator = 0 if model.device.type == "cpu" else info.accelerator_bytes
    spec = replace(spec, resources=ResourceBudget(info.cpu_bytes + spec.resources.cpu_bytes, accelerator))
    result = pipeline.run(RunContext(model, dataset, spec, output))
    return result, read_json(output / "run.json")


def run_pipeline(pipeline, model_factory, dataset, directory, stage, settings, passes, name,
                 describe=None, plan_factory=None, **fields):
    """One implementation for training/analysis, caching, per-seed attempts, and exports."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if stage == "hindi" and settings.protocol == "scientific":
        runs = []
        for seed in settings.hindi_seeds:
            child = copy.copy(pipeline)
            child.policy = copy.deepcopy(pipeline.policy)
            child.rng = random.Random(seed)
            if plan_factory is not None:
                child.adapter_plan = plan_factory(seed)
            runs.append(single_run(child, model_factory, dataset, directory / f"seed-{seed}", stage,
                                   replace(settings, seed=seed), passes, f"{name}-seed-{seed}", describe, fields))
        record = aggregate_runs(name, directory, runs)
        write_json(directory / "run.json", {"status": "completed", "seeds": list(settings.hindi_seeds)})
        return remember_arm(directory, record)
    return single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields)


def single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields):
    signature = fingerprint({"version": 2, "settings": asdict(settings), "dataset": dataset.fingerprint,
                             "config": asdict(pipeline.config), "inputs": pipeline.input_identity(), "name": name})
    saved = finished_arm(directory, signature)
    if saved is not None:
        return ensure_final(directory.parent, saved)
    model = model_factory()
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
    return finish_arm(directory.parent, directory, record)


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


def forgetting(before, arm):
    return score(before, "english/validation/plain") - score(arm["metrics"], "english/validation/plain")


def gain(before, arm):
    return score(arm["metrics"], "hindi/validation/plain") - score(before, "hindi/validation/plain")


def choose_winner(before, candidates, baseline):
    """Lowest English validation forgetting among arms within 1 Hindi point of the baseline."""
    if not candidates:
        raise ValueError("Winner selection needs at least one arm")
    floor = gain(before, baseline) - 1
    eligible = [arm for arm in candidates if gain(before, arm) >= floor]
    pool = eligible or [baseline]
    return sorted(pool, key=lambda arm: (forgetting(before, arm),
                                         arm.get("update_seconds", arm.get("wall_clock_seconds")) or 0, arm["name"]))[0]


def lower_forgetting(before, arms):
    return sorted(arms, key=lambda arm: (forgetting(before, arm),
                                         arm.get("update_seconds", arm.get("wall_clock_seconds")) or 0, arm["name"]))[0]


def script_share_of_forgetting(before, plain, guarded):
    drop = before - plain
    if drop <= 0:
        return None
    return (guarded - plain) / drop


def gate_decision(forget, hindi_gain):
    if hindi_gain < 1:
        return "fix_setup"
    if forget < 1:
        return "stop_replay"
    return "continue"


def read_context(root, dataset, settings, before):
    context = {}
    for index in range(before):
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
    return RevisionConfig(hindi_batch=settings.batch_size, replay_per_batch=settings.replay_per_batch,
                          hindi_passes=settings.hindi_passes, analysis_batch=settings.analysis_batch,
                          generation=generation_spec(settings), **overrides)


def hindi_pipeline(settings, pool, checkpoint, **overrides):
    return RevisionPipeline(revision_config(settings, stage="hindi", **overrides),
                            pool=pool, english_checkpoint=checkpoint)


def run_hindi(directory, name, pipeline, dataset, model_factory, settings, plan_factory=None, **fields):
    fields = {"selector": pipeline.config.selector, "replay_text": pipeline.config.replay_text,
              "repair_stop": pipeline.config.repair_stop, **fields}
    return run_pipeline(pipeline, model_factory, dataset, directory, "hindi", settings,
                        settings.hindi_passes, name, plan_factory=plan_factory, **fields)


def room_config(settings):
    return AdapterRoomConfig(settings.stability_threshold, seed=settings.seed, sample_size=settings.sample_size,
        analysis_batch=settings.analysis_batch, encoder_blocks=settings.encoder_blocks,
        decoder_blocks=settings.decoder_blocks, new_room_budget=settings.new_room_budget)


def room_plans(settings, context, seed=None, frozen=False):
    room = AdapterRoomPipeline(room_config(replace(settings, seed=settings.seed if seed is None else seed)))
    shared = room.adapter_plan("all" if context["best_blocks"] == "random" else context["best_blocks"])
    names = ["room_placed", "room_random", "room_spread", "room_placed_slow"]
    if frozen:
        names.append("room_placed_frozen")
    plans = [room.adapter_plan(name, maps=context["maps"], shared=shared) for name in names]
    if any(plan is None for plan in plans):
        return (), room.policy.get("room_status", "skipped")
    try:
        return room.validate_room_plans(plans), None
    except ValueError as error:
        return (), str(error)
