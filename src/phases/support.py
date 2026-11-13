from dataclasses import asdict, replace
from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import math
import re
import time

from ..pipelines.revision import RevisionConfig, RevisionPipeline
from ..pipelines.block_selection import BlockSelectionConfig, BlockSelectionPipeline
from ..utils.numerics import summary_metrics
from ..utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec, fingerprint
from ..utils.state import read_json, write_json
from .artifacts import PhaseClock, arm_record, finish_arm
from .settings import generation_spec

COOLDOWN_SECONDS = 0
_RUN_ARMS = ContextVar("run_arms", default=None)


@contextmanager
def run_session():
    token = _RUN_ARMS.set({})
    try:
        yield
    finally:
        _RUN_ARMS.reset(token)


def run_spec(stage, settings, passes, arm, dataset, model, extra_steps=0):
    rows = tuple(dataset.training_view(stage)) if stage != "analysis" else ()
    updates = math.ceil(len(rows) / settings.batch_size) * passes + extra_steps
    bound = max(1, updates) * (2 * settings.batch_size + settings.replay_per_batch) * (
        model.settings.max_source_length + model.settings.max_target_length)
    return RunSpec((stage,), settings.seed, settings.batch_size, max(1, updates), max(1, updates), bound, passes,
                   "repeat_admitted", "per_example_mean", ReplayBudget(len(dataset.pool_view()), 50_000_000, 50_000_000),
                   ResourceBudget(64_000_000, 0),
                   {"protocol": settings.protocol, "arm": arm, "training_rows": len(rows)}, {})


def execute(pipeline, model, dataset, output, stage, settings, passes, arm):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    extra = getattr(pipeline.config, "extra_random_steps", 0)
    if hasattr(pipeline, 'extra_updates'):
        extra += pipeline.extra_updates(dataset, settings)
    spec = run_spec(stage, settings, passes, arm, dataset, model, extra)
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
                 describe=None, **fields):
    """Execute each arm once, with one canonical output directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields)


def single_run(pipeline, model_factory, dataset, directory, stage, settings, passes, name, describe, fields):
    signature = fingerprint({"version": 2, "settings": asdict(settings), "dataset": dataset.fingerprint,
                             "config": asdict(pipeline.config), "inputs": pipeline.input_identity(), "name": name})
    cache = _RUN_ARMS.get()
    key = str(directory.resolve())
    if cache is not None and key in cache:
        saved_signature, saved = cache[key]
        if saved_signature != signature:
            raise ValueError(f"Conflicting arm settings in the same run: {directory}")
        return saved
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
    finished = finish_arm(directory, record)
    if cache is not None:
        cache[key] = (signature, finished)
    cool_down(settings, stage, pretrained, device=model.device)
    return finished


def cool_down(settings, stage, pretrained=True, *, device=None):
    """Apply an enabled cooldown after saving real training runs; skip smoke, tiny models and CUDA."""
    if device is not None and device.type == "cuda":
        return
    if (stage in ("english", "hindi") and pretrained and settings.protocol != "smoke" and COOLDOWN_SECONDS > 0):
        print(f"Cooling down for {COOLDOWN_SECONDS} s after the {stage} run", flush=True)
        time.sleep(COOLDOWN_SECONDS)


def score(metrics, task, name="chrf"):
    return metrics[task][name]


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
    return lower_forgetting(before, eligible or [baseline])


def lower_forgetting(before, arms):
    """Compared on each arm's validation result. Ties go to the cheaper arm by update time."""
    def key(arm):
        run = arm
        return forgetting(before, run), run.get("update_seconds", run.get("wall_clock_seconds")) or 0, arm["name"]
    return sorted(arms, key=key)[0]


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


def phase(index):
    def decorate(function):
        @wraps(function)
        def run(root, dataset, model_factory, settings, context):
            directory = Path(root) / f"phase{index}"
            directory.mkdir(parents=True, exist_ok=False)
            return function(directory, dataset, model_factory, settings, context, PhaseClock())
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


def run_hindi(directory, name, pipeline, dataset, model_factory, settings, **fields):
    fields = {"selector": pipeline.config.selector, "replay_text": pipeline.config.replay_text,
              "repair_stop": pipeline.config.repair_stop, **fields}
    return run_pipeline(pipeline, model_factory, dataset, directory, "hindi", settings,
                        settings.hindi_passes, name, **fields)


BLOCK_PLANS = {"random": "all", "late": "late", "early_late": "early_late"}


def revision_pipeline(settings, pool, checkpoint, revision):
    """Rebuild a revision arm from its record. The config matches the original arm."""
    return hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                          replay_text=revision["replay_text"], repair_stop=revision["repair_stop"],
                          extra_random_steps=revision.get("extra_random_steps", 0))


def combined_pipeline(settings, pool, checkpoint, revision, blocks):
    pipeline = revision_pipeline(settings, pool, checkpoint, revision)
    pipeline.adapter_plan = BlockSelectionPipeline(block_config(settings)).adapter_plan(BLOCK_PLANS[blocks])
    return pipeline


def matched_extra_steps(random_arm, drift_arm):
    """Extra random-replay updates that spend G1's measured extra time (scoring, writing, and updates)."""
    def total(run):
        return (run.get("update_seconds") or 0) + sum((run.get("costs") or {}).get("analysis_seconds", {}).values())
    random_run, drift_run = random_arm, drift_arm
    per_step = (random_run.get("update_seconds") or 0) / max(1, random_run.get("steps") or 0)
    if per_step <= 0:
        return 0
    return max(0, math.ceil((total(drift_run) - total(random_run)) / per_step))


def block_config(settings):
    return BlockSelectionConfig(settings.stability_threshold, seed=settings.seed, sample_size=settings.sample_size,
        analysis_batch=settings.analysis_batch, encoder_blocks=settings.encoder_blocks,
        decoder_blocks=settings.decoder_blocks,
        encoder_block_cost=settings.encoder_block_cost, decoder_block_cost=settings.decoder_block_cost)
