"""CSReL comparison: new-only, random memory, CSReL-CL, and full retrain.

From the repository root:

    python -m src.csrel_experiment --protocol backend-smoke
    python -m src.csrel_experiment --protocol smoke --tasks banking,credit_cards --budgets 8
    python -m src.csrel_experiment --protocol scientific --budgets 32,128 --rounds 1,2

backend-smoke needs the SmolLM2 adapter dependencies and writes a tiny local JSONL.
smoke and scientific read src/datasets/data/clinc_oos_domains/<commit>/.
Neither command deletes an existing attempt directory.
"""

from dataclasses import replace
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.dataset_instruction import InstructionDataset
from src.datasets.dataset_sources import CLINC_DOMAINS, load_source
from src.models.model_causal_lm import CausalLMSettings
from src.models.model_smollm2 import SmolLM2Model
from src.pipelines.csrel import CSReLConfig, CSReLPipeline
from src.pipelines.memory_baselines import MemoryBaselineConfig, MemoryBaselinePipeline
from src.utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec
from src.utils.numerics import continual_metrics
from src.utils.state import read_json, write_json

ARMS = ("new_only", "random", "csrel", "cumulative")
FORGETTING = (
    "For task j, forgetting is the best score of j after it was learned and before the "
    "final stage, minus its final score. The reported value averages those tasks. "
    "A task that improves has negative forgetting."
)


class LimitedViews:
    """Expose an arrival-order prefix. The limit is part of the dataset fingerprint."""

    def __init__(self, dataset, max_train=None, max_eval=None):
        self._dataset = dataset
        self.max_train = max_train or None
        self.max_eval = max_eval or None

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    @property
    def fingerprint(self):
        return json.dumps({"base": self._dataset.fingerprint, "max_train": self.max_train,
                           "max_eval": self.max_eval, "order": "arrival_prefix"},
                          sort_keys=True, default=str)

    def training_view(self, task_id):
        rows = self._dataset.training_view(task_id)
        return rows[:self.max_train] if self.max_train else rows

    def evaluation_view(self, task_id):
        rows = self._dataset.evaluation_view(task_id)
        return rows[:self.max_eval] if self.max_eval else rows


class BalancedViews:
    """Choose equal counts per intent after preprocessing; preserve canonical arrival order."""

    def __init__(self, dataset, train_per_class, eval_per_class, seed, max_prompt_tokens=None):
        self._dataset = dataset
        self.train_per_class, self.eval_per_class, self.seed = train_per_class, eval_per_class, seed
        self.max_prompt_tokens = max_prompt_tokens

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    @property
    def fingerprint(self):
        return json.dumps({"base": self._dataset.fingerprint, "train_per_class": self.train_per_class,
                           "eval_per_class": self.eval_per_class, "seed": self.seed,
                           "max_prompt_tokens": self.max_prompt_tokens,
                           "sampling": "seeded-class-balanced-v1"}, sort_keys=True, default=str)

    def _select(self, rows, count, task, split):
        groups = defaultdict(list)
        for record in rows:
            if (self.max_prompt_tokens is not None
                    and len(record.inputs["prompt_ids"]) > self.max_prompt_tokens):
                continue  # Reserve space for the declared generation length.
            if record.class_id is None:
                raise ValueError("Balanced pilot requires genuine class labels")
            groups[record.class_id].append(record)
        chosen = []
        for label, members in sorted(groups.items()):
            if len(members) < count:
                raise ValueError(f"Insufficient {split} examples for {task}/{label}")
            rng = random.Random(f"{self.seed}:{task}:{split}:{label}")
            chosen.extend(rng.sample(sorted(members, key=lambda r: r.sample_id), count))
        return tuple(sorted(chosen, key=lambda r: (r.arrival, r.sample_id)))

    def training_view(self, task_id):
        return self._select(self._dataset.training_view(task_id), self.train_per_class, task_id, "train")

    def evaluation_view(self, task_id):
        return self._select(self._dataset.evaluation_view(task_id), self.eval_per_class, task_id, "validation")


def directory_bytes(path):
    """Sum file sizes under one directory. A missing directory contributes zero."""
    path = Path(path)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cost_seconds(cost):
    """Read the adapter's device time when it recorded one, otherwise the host wall time."""
    if not isinstance(cost, dict):
        return None
    for key in ("device_active_seconds", "synchronized_wall_seconds", "wall_seconds"):
        if cost.get(key) is not None:
            return float(cost[key])
    return None


def performance_report(metrics, tasks, metric):
    """Build the seen-task matrix and the final average, diagonal, and forgetting."""
    index = {task: position for position, task in enumerate(tasks)}
    grid = [[None] * len(tasks) for _ in tasks]
    for row in metrics:
        grid[index[row["checkpoint_task"]]][index[row["task"]]] = row["metrics"][metric]
    observed = []
    for position, row in enumerate(grid):
        if any(value is None for value in row[:position + 1]):
            raise ValueError(f"Missing {metric} after {tasks[position]}")
        observed.append([float(value) for value in row[:position + 1]])
    stats = continual_metrics(observed, len(tasks) - 1)
    final = observed[-1]
    return {
        "metric": metric,
        "higher_is_better": True,
        "tasks": list(tasks),
        "matrix": [{ "after_task": task, "scores": {
            tasks[column]: grid[position][column] for column in range(len(tasks))}}
            for position, task in enumerate(tasks)],
        "final_average": sum(final) / len(final),
        "current_task": {tasks[position]: grid[position][position] for position in range(len(tasks))},
        "forgetting": stats["forgetting"],
        "backward_transfer": stats["backward_transfer"],
        "forgetting_definition": FORGETTING,
    }


def _add(total, value):
    if value is None:
        return total
    return (total or 0.0) + value


def timing_report(events):
    """Separate learner updates from reference fitting, subset fitting, and candidate scoring."""
    learner = aux_reference = aux_subset = score_reference = score_subset = selection = None
    score_batches = {"reference": 0, "subset": 0}
    pending = None
    for event in events:
        kind = event.get("kind")
        if kind == "execution_cost" and event.get("role") == "learner":
            learner = _add(learner, event.get("wall_seconds"))
        elif kind == "execution_cost" and event.get("role") == "auxiliary":
            pending = event.get("wall_seconds")
        elif kind == "auxiliary_update" and pending is not None:
            if event.get("phase") == "reference_fit":
                aux_reference = _add(aux_reference, pending)
            else:
                aux_subset = _add(aux_subset, pending)
            pending = None
        elif kind == "auxiliary_scoring":
            seconds = cost_seconds(event.get("cost"))
            score_batches[event.get("table", "subset")] = score_batches.get(event.get("table"), 0) + 1
            if event.get("table") == "reference":
                score_reference = _add(score_reference, seconds)
            else:
                score_subset = _add(score_subset, seconds)
        elif kind == "selection_cost":
            selection = _add(selection, event.get("wall_seconds"))
    return {
        "learner_training_seconds": learner,
        "reference_fitting_seconds": aux_reference,
        "subset_fitting_seconds": aux_subset,
        "reference_scoring_seconds": score_reference,
        "subset_scoring_seconds": score_subset,
        "reference_scoring_batches": score_batches.get("reference", 0),
        "subset_scoring_batches": score_batches.get("subset", 0),
        "random_selection_seconds": selection,
        "timing_note": (
            "Learner and auxiliary fitting seconds are host wall time around each update. "
            "Scoring seconds are present only when the model adapter recorded them. "
            "A null scoring time means the adapter did not measure it, not that scoring was free."
        ),
    }


def storage_report(output, events):
    """Count the last committed replay separately from checkpoints, teachers, and offload."""
    commits = [event for event in events if event.get("kind") == "memory_commit"]
    usage = commits[-1]["storage"] if commits else {"examples": 0, "tokens": 0, "bytes": 0}
    output = Path(output)
    return {
        "replay_examples": usage.get("examples", 0),
        "replay_tokens": usage.get("tokens", 0),
        "replay_bytes": usage.get("bytes", 0),
        "checkpoint_bytes": directory_bytes(output / "checkpoints"),
        "teacher_bytes": directory_bytes(output / "teachers"),
        "offload_bytes": directory_bytes(output / ".offload"),
        "initial_weights_bytes": directory_bytes(output / "initial_weights"),
    }


def summarize_attempt(output, tasks, metric):
    """Read one finished attempt and return the matrix, times, and storage."""
    output = Path(output)
    events = read_json(output / "events.json")
    metrics = []
    # metrics.csv is the durable table; events also carry the same rows.
    for event in events:
        if event.get("kind") == "evaluation":
            metrics.append({"checkpoint_task": event["task"], "task": event["task_evaluated"],
                            "metrics": event["metrics"]})
    report = performance_report(metrics, tasks, metric)
    report.update(timing_report(events))
    report.update(storage_report(output, events))
    run = read_json(output / "run.json")
    report["status"] = run.get("status")
    report["run_identity"] = run.get("identity")
    return report


def write_comparison(path, rows):
    """Write machine-readable rows and a short markdown table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.with_suffix(".json"), rows)
    lines = [
        "| arm | budget | final average | forgetting | replay examples | learner seconds | selection seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fields = ("reference_fitting_seconds", "subset_fitting_seconds",
                  "reference_scoring_seconds", "subset_scoring_seconds", "random_selection_seconds")
        measured = [row[key] for key in fields if row.get(key) is not None]
        selection = sum(measured) if measured else None
        lines.append(
            f"| {row['arm']} | {row['budget']} | {row.get('final_average')} | "
            f"{row.get('forgetting')} | {row.get('replay_examples')} | "
            f"{row.get('learner_training_seconds')} | {selection} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with path.with_suffix(".md").open("a", encoding="utf-8") as stream:
        stream.write("\nSelection seconds sum measured fitting/scoring components; they exclude unmeasured orchestration.\n")


def smoke_records(tasks=("banking", "credit_cards")):
    """Four train, two validation, and two test rows per task. This is not CLINC."""
    rows = []
    for task in tasks:
        for split, count in (("train", 4), ("validation", 2), ("test", 2)):
            for index in range(count):
                label = f"{task}-intent-{index % 2}"
                rows.append({"id": f"{task}-{split}-{index}", "task": task, "split": split,
                             "prompt": f"{task} request number {index}", "response": label,
                             "class": label})
    return rows


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the standalone CSReL comparison.")
    parser.add_argument("--protocol", choices=("backend-smoke", "smoke", "pilot", "scientific"),
                        default="backend-smoke")
    parser.add_argument("--output", default="results/csrel-replay")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--task-count", type=int, default=0)
    parser.add_argument("--budgets", default="")
    parser.add_argument("--rounds", default="")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--shuffle-training", action="store_true",
                        help="Shuffle learner examples each pass; auxiliary order is unchanged.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--replay-batch-size", type=int, default=0)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--reference-passes", type=int, default=1)
    parser.add_argument("--subset-passes", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=10)
    parser.add_argument("--eval-per-class", type=int, default=5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", choices=("auto", "required", "never"), default="auto")
    parser.add_argument("--metric", default="exact_match")
    return parser.parse_args(argv)


def resolved_config(args):
    """Fill protocol defaults and reject a scientific run that truncates data."""
    config = vars(args).copy()
    if args.protocol == "scientific" and (args.max_train_examples or args.max_eval_examples):
        raise SystemExit("scientific refuses max-train-examples and max-eval-examples")
    if args.protocol == "backend-smoke":
        config.update(batch_size=args.batch_size or 2, replay_batch_size=args.replay_batch_size or 1,
                      max_steps=args.max_steps or 1, max_attempts=args.max_attempts or 4,
                      max_tokens=args.max_tokens or 20000, max_length=args.max_length or 64,
                      budgets=args.budgets or "2", rounds=args.rounds or "1",
                      tasks=args.tasks or "banking,credit_cards")
    elif args.protocol == "smoke":
        config.update(batch_size=args.batch_size or 2, replay_batch_size=args.replay_batch_size or 1,
                      max_steps=args.max_steps or 2, max_attempts=args.max_attempts or 8,
                      max_tokens=args.max_tokens or 200000, max_length=args.max_length or 96,
                      budgets=args.budgets or "8", rounds=args.rounds or "1",
                      tasks=args.tasks or "banking,credit_cards",
                      max_train_examples=args.max_train_examples or 8,
                      max_eval_examples=args.max_eval_examples or 4)
    elif args.protocol == "pilot":
        if args.max_train_examples or args.max_eval_examples:
            raise SystemExit("pilot uses per-class counts, not arrival prefixes")
        if min(args.train_per_class, args.eval_per_class) <= 0:
            raise SystemExit("pilot per-class counts must be positive")
        config.update(batch_size=args.batch_size or 4, replay_batch_size=args.replay_batch_size or 2,
                      max_steps=args.max_steps or 10000, max_attempts=args.max_attempts or 10000,
                      max_tokens=args.max_tokens or 10_000_000, max_length=args.max_length or 128,
                      budgets=args.budgets or "30", rounds=args.rounds or "2",
                      tasks=args.tasks or "banking,credit_cards,travel")
    else:
        config.update(batch_size=args.batch_size or 4, replay_batch_size=args.replay_batch_size or 4,
                      max_steps=args.max_steps or 200, max_attempts=args.max_attempts or 400,
                      max_tokens=args.max_tokens or 2_000_000, max_length=args.max_length or 128,
                      budgets=args.budgets or "32,128", rounds=args.rounds or "1")
    config["budgets"] = [int(item) for item in str(config["budgets"]).split(",") if item]
    config["rounds"] = [int(item) for item in str(config["rounds"]).split(",") if item]
    config["arms"] = [item for item in args.arms.split(",") if item]
    unknown = set(config["arms"]).difference(ARMS)
    if unknown:
        raise SystemExit(f"Unknown arms {sorted(unknown)}")
    return config


def load_real_dataset(config):
    """Load SmolLM2 and either the tiny smoke JSONL or the domain CLINC snapshot."""
    output = ROOT / config["output"] / config["protocol"]
    if config["protocol"] == "backend-smoke":
        data_path = output / "data" / "records.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            lines = [json.dumps(row) for row in smoke_records()]
            data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dataset = InstructionDataset(evaluation_split="test")
        raw, _ = dataset.load({"path": data_path, "dataset_id": "backend-smoke"})
        tasks = tuple(config["tasks"].split(","))
    else:
        root = ROOT / "src" / "datasets" / "data" / "clinc_oos_domains"
        ready = [path for path in root.glob("*") if (path / "records.jsonl").is_file()] if root.exists() else []
        if not ready:
            raise SystemExit(
                "CLINC domain JSONL is missing. Fetch it with:\n"
                "python -m src.datasets.dataset_sources clinc_oos_domains src/datasets/data")
        dataset, raw = load_source(max(ready, key=lambda path: (path / "manifest.json").stat().st_mtime), InstructionDataset)
        order = tuple(dataset.manifest["source"]["task_order"])
        requested = tuple(item for item in config["tasks"].split(",") if item) if config["tasks"] else order
        if config["task_count"]:
            requested = requested[:config["task_count"]]
        missing = [task for task in requested if task not in order]
        if missing or not set(requested).issubset(CLINC_DOMAINS):
            raise SystemExit(f"Tasks {missing or requested} are not official CLINC domains in {order}")
        tasks = requested
        if config["protocol"] == "pilot":
            dataset.evaluation_split = "validation"
            instruction = ("Classify the user's request by its intent. "
                           "Reply with only the exact CLINC intent label, using underscores where needed.")
            raw = tuple(replace(record, inputs={"messages": [
                {"role": "system", "content": instruction}, *record.inputs["messages"]]}) for record in raw)
            dataset.manifest["pilot_instruction"] = instruction
        if config["max_train_examples"] or config["max_eval_examples"]:
            dataset.manifest["smoke_truncation"] = {
                "max_train_examples": config["max_train_examples"] or None,
                "max_eval_examples": config["max_eval_examples"] or None,
                "order": "arrival_prefix",
            }
    settings = CausalLMSettings(
        mode="lora", rank=config["rank"], alpha=config["alpha"], seed=config["seed"],
        learning_rate=config["learning_rate"], max_length=config["max_length"],
        # Current and replay rows are collated into one learner update.
        max_batch_size=config["batch_size"] + config["replay_batch_size"],
        working_bytes=1024 ** 3, max_score_bytes=256 * 1024 ** 2, compile=False,
    )
    model = SmolLM2Model(settings, device=config["device"])
    return model, dataset, raw, tasks


def make_pipeline(arm, config, rounds, model, dataset, raw):
    """Build one arm. Random and CSReL share the replay hyperparameters."""
    shared = dict(replay_weight=config["replay_weight"], replay_batch_size=config["replay_batch_size"],
                  replacement=False)
    if arm == "csrel":
        pipeline = CSReLPipeline(CSReLConfig(
            "csrel_cl", rounds, config["reference_passes"], config["subset_passes"],
            "reference-base", "subset-base", shared["replay_weight"], shared["replay_batch_size"], False))
    else:
        pipeline = MemoryBaselinePipeline(MemoryBaselineConfig(
            arm, **shared, require_full_passes=config["protocol"] == "pilot"))
    pipeline.set_model(model)
    prepared, report = pipeline.bind_instruction_dataset(dataset, raw)
    if config["protocol"] == "pilot":
        pipeline.dataset = BalancedViews(pipeline.dataset, config["train_per_class"],
                                         config["eval_per_class"], config["seed"], config["max_length"] - 16)
        for component in pipeline.components().values():
            component.dataset = pipeline.dataset
    elif config["max_train_examples"] or config["max_eval_examples"]:
        limited = LimitedViews(pipeline.dataset, config["max_train_examples"] or None,
                               config["max_eval_examples"] or None)
        pipeline.dataset = limited
        for component in pipeline.components().values():
            component.dataset = limited
    return pipeline, prepared, report


def run_spec(config, tasks, budget, arm, rounds):
    """One scientific identity. Resume mode stays outside that identity."""
    return RunSpec(
        task_ids=tuple(tasks), seed=config["seed"], batch_size=config["batch_size"],
        max_attempts=config["max_attempts"], max_steps=config["max_steps"],
        max_tokens=config["max_tokens"], passes=config["passes"], exhaustion="repeat_admitted",
        reduction="per_example_mean",
        replay=ReplayBudget(budget, 5_000_000, 50_000_000),
        resources=ResourceBudget(16 * 1024 ** 3,
                                 (4 if config["protocol"] == "pilot" else 16) * 1024 ** 3,
                                 40 * 1024 ** 3),
        checkpoint_every=10 if config["protocol"] == "pilot" else 1, resume=config["resume"],
        identity={
            "protocol": config["protocol"], "arm": arm, "budget": budget, "rounds": rounds,
            "shuffle_training": config.get("shuffle_training", False),
            "metric": config["metric"],
            "adaptations": ["causal-lm-token-mean", "configured-auxiliary-passes",
                            "empty-subset-untrained-initialization"],
            "cumulative_protocol": "retrain_from_common_initial_weights" if arm == "cumulative" else "",
            "evaluation_split": "validation" if config["protocol"] == "pilot" else "test",
            "validation_used_for_tuning": config["protocol"] == "pilot",
            "train_per_class": config.get("train_per_class") if config["protocol"] == "pilot" else None,
            "eval_per_class": config.get("eval_per_class") if config["protocol"] == "pilot" else None,
            "max_train_examples": config["max_train_examples"] or None,
            "max_eval_examples": config["max_eval_examples"] or None,
        },
        evaluation_spec={"batch_size": config["batch_size"],
                         "return_predictions": config["protocol"] == "pilot",
                         "generation": {"max_new_tokens": 16 if config["protocol"] == "pilot" else 8,
                                        "do_sample": False, "seed": config["seed"]}},
    )


def attempt_for(parent, config_hash):
    """Reuse an unfinished matching attempt. Completed, rejected, and foreign attempts stay on disk."""
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = parent / f"attempt-{index}"
        marker = path / "experiment.json"
        if not path.exists():
            return path, False
        saved = read_json(marker) if marker.exists() else {}
        if saved.get("config_hash") == config_hash and not saved.get("rejected"):
            status = read_json(path / "run.json").get("status") if (path / "run.json").exists() else "running"
            return path, status == "completed"
        index += 1
        if index > 1000:
            raise RuntimeError(f"Too many attempts in {parent}")


def run_arm(config, arm, budget, rounds, model, dataset, raw, tasks):
    """Run one arm into a new-or-resumed directory and return its summary row."""
    spec = run_spec(config, tasks, budget, arm, rounds)
    digest = json.dumps({"spec": spec.scientific(), "arm": arm, "rounds": rounds,
                         "configuration": {key: value for key, value in config.items()
                                           if key not in {"output", "resume", "initial_weights"}},
                         "dataset_limits": [config["max_train_examples"], config["max_eval_examples"]]},
                        sort_keys=True, default=str)
    parent = ROOT / config["output"] / config["protocol"] / arm / f"seed-{config['seed']}" / f"budget-{budget}" / f"rounds-{rounds}"
    output, completed = attempt_for(parent, digest)
    row = {"arm": arm, "budget": budget, "rounds": rounds,
           "output": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)}
    if completed:
        row["skipped"] = "completed attempt preserved"
        row.update({key: value for key, value in summarize_attempt(output, tasks, config["metric"]).items()
                    if key not in row})
        return row
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment.json", {"config_hash": digest, "config": config, "arm": arm,
                                            "budget": budget, "rounds": rounds})
    model.load(config["initial_weights"], "frozen")
    pipeline, _, _ = make_pipeline(arm, config, rounds, model, dataset, raw)
    if config["protocol"] == "pilot":
        coverage = {}
        for task in tasks:
            coverage[task] = {}
            for split, view in (("train", pipeline.dataset.training_view(task)),
                                ("validation", pipeline.dataset.evaluation_view(task))):
                counts = Counter(record.class_id for record in view)
                if set(counts) != set(CLINC_DOMAINS[task]):
                    raise ValueError(f"Pilot lost intent coverage in {task}/{split}")
                coverage[task][split] = {"examples": len(view), "classes": dict(counts),
                                         "ids": [record.sample_id for record in view]}
        write_json(output / "coverage.json", coverage)
    started = time.perf_counter()
    try:
        result = pipeline.run(RunContext(model, pipeline.dataset, spec, output))
    except ValueError as error:
        if "different scientific run" not in str(error):
            raise
        rejected = read_json(output / "experiment.json")
        rejected["rejected"] = True
        write_json(output / "experiment.json", rejected)
        write_json(output / "resume-rejected.json", {"error": str(error)})
        output, _ = attempt_for(parent, digest)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "experiment.json", {"config_hash": digest, "config": config,
                                                "restarted_because": str(error)})
        model.load(config["initial_weights"], "frozen")
        pipeline, _, _ = make_pipeline(arm, config, rounds, model, dataset, raw)
        result = pipeline.run(RunContext(model, pipeline.dataset, replace(spec, resume="never"), output))
    row["total_runtime_seconds"] = time.perf_counter() - started
    row["run_status"] = result.status
    summary = summarize_attempt(output, tasks, config["metric"])
    summary["total_runtime_seconds"] = row["total_runtime_seconds"]
    write_json(output / "summary.json", summary)
    row.update(summary)
    return row


def main(argv=None):
    """Resolve the config, run the requested arms, and write comparison.json."""
    config = resolved_config(parse_args(sys.argv[1:] if argv is None else argv))
    try:
        model, dataset, raw, tasks = load_real_dataset(config)
        initial = (ROOT / config["output"] / config["protocol"] /
                   "shared-initial")
        if not (initial / "metadata.json").is_file():
            model.activate()
            initial.mkdir(parents=True, exist_ok=True)
            model.save(initial, "frozen")
            model.offload()
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency {error.name}. Real-model runs need the declared Torch extras:\n"
            "python -m pip install \"transformers>=5,<6\" safetensors huggingface_hub pyarrow") from error
    config["initial_weights"] = str(initial)
    rows = []
    once = {"new_only", "cumulative"}
    ran_once = set()
    for budget in config["budgets"]:
        for rounds in config["rounds"]:
            for arm in config["arms"]:
                if arm in once and arm in ran_once:
                    continue
                print(f"arm {arm} budget {budget} rounds {rounds}", flush=True)
                rows.append(run_arm(config, arm, budget, rounds, model, dataset, raw, tasks))
                ran_once.add(arm)
    destination = ROOT / config["output"] / config["protocol"] / "comparison"
    write_comparison(destination, rows)
    print(destination.with_suffix(".md"), flush=True)
    return rows


if __name__ == "__main__":
    main()
