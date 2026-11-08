"""Phase records, atomic model exports, and JSON/CSV/Markdown reports."""

from pathlib import Path
from dataclasses import asdict, is_dataclass
import csv
import json
import shutil
import statistics
import time

from ..utils.state import Checkpoints, checksums, read_json, write_json


def accuracies_of(metrics):
    """chrF, ROUGE-L, and right-language share, the scores this plan reports."""
    found = {}
    for task, scores in metrics.items():
        if isinstance(scores, dict):
            kept = {name: scores[name] for name in ("rouge_l", "chrf", "right_language") if name in scores}
            if kept:
                found[task] = kept
    return found


def arm_record(name, output, checkpoint, result, manifest, pipeline, **fields):
    events = read_json(Path(output) / "events.json")
    metrics = {row["task"]: row["metrics"] for row in result.metrics}
    steps = result.counters.get("steps", 0)
    rows = (manifest.get("spec", {}).get("identity") or {}).get("training_rows", 0)
    exposed = sum(event.get("origins", []).count("current") for event in events if event["kind"] == "learner_update")
    costs = {"update_seconds": 0.0, "analysis_seconds": {}, "forward_pass_units": 0, "tokens_per_language": {}}
    for event in events:
        if event["kind"] == "execution_cost" and event.get("role") == "learner":
            costs["update_seconds"] += event["wall_seconds"]
        elif event["kind"] == "analysis_cost":
            measured = event.get("measured", {})
            seconds = measured.get("synchronized_wall_seconds", measured.get("device_active_seconds", 0))
            category = event["category"]
            costs["analysis_seconds"][category] = costs["analysis_seconds"].get(category, 0) + seconds
            costs["forward_pass_units"] += event["forward_pass_units"]
        elif event["kind"] == "revision_exposure":
            costs["forward_pass_units"] += event["forward_pass_units"]
            for language, tokens in event["tokens_per_language"].items():
                costs["tokens_per_language"][language] = costs["tokens_per_language"].get(language, 0) + tokens
    record = {"name": name, "seed": manifest["spec"]["seed"], "attempt": str(output), "checkpoint": str(checkpoint),
              "status": result.status, "steps": steps, "metrics": metrics,
              "ranking": list(pipeline.policy.get("ranking") or []), "wall_clock_seconds": manifest.get("wall_clock_seconds"),
              "filtered": list(pipeline.policy.get("filtered") or []), "retired": list(pipeline.policy.get("retired") or []),
              "retired_share": len(pipeline.policy.get("retired") or []) / max(1, len(pipeline.policy.get("ever_active") or [])),
              "epochs": {"requested": manifest.get("spec", {}).get("passes", 0) if rows else 0,
                         "completed": exposed / rows if rows else 0, "steps": steps},
              "time": {"wall_clock_seconds": manifest.get("wall_clock_seconds"),
                       "gpu_hours": manifest.get("gpu_hours"), "peak_memory_mb": manifest.get("peak_memory_mb")},
              "costs": costs,
              "update_seconds": costs["update_seconds"], "trainable_parameters": pipeline.model.info.trainable_parameters,
              "storage_bytes": sum(file.stat().st_size for file in Path(output).rglob("*") if file.is_file()), **fields}
    record["accuracies"] = accuracies_of(record["metrics"])
    plan = getattr(pipeline, "adapter_plan", None)
    record["adapter_plan"] = asdict(plan) if is_dataclass(plan) else None
    return record


def finished_arm(directory, signature=None):
    """Reuse only a completed, matching arm; incomplete attempts resume numerically."""
    directory = Path(directory)
    record = directory / "arm.json"
    manifest = directory / "run.json"
    if record.is_file() and manifest.is_file() and read_json(manifest).get("status") == "completed":
        value = read_json(record)
        if signature is not None and value.get("signature") != signature:
            raise ValueError(f"Arm settings changed: {directory}; choose a new attempt")
        return value
    return None


def remember_arm(directory, record):
    write_json(Path(directory) / "arm.json", record)
    return record


def publish_final(phase_dir, name, source):
    """Publish an independently loadable adapter through the shared atomic artifact writer."""
    source = Path(source)
    if not (source / "metadata.json").is_file():
        return None
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Final model name must be one path component")
    destination = Path(phase_dir) / "final" / name
    source_files = checksums(source)
    marker = destination / "complete.json"
    if marker.is_file() and read_json(marker)["files"] != source_files:
        raise ValueError(f"Final model {name} does not match its source")
    manager = Checkpoints(phase_dir, "phase-exports")
    size = sum(file.stat().st_size for file in source.rglob("*") if file.is_file())
    manager.pin_artifact(destination, size + 65536,
                         lambda stage: shutil.copytree(source, stage, dirs_exist_ok=True))
    return str(destination)


def ensure_final(phase_dir, record):
    copied = publish_final(phase_dir, record["name"], record.get("checkpoint", ""))
    if copied and record.get("final_model") != copied:
        record = {**record, "final_model": copied}
        remember_arm(record["attempt"], record)
    return record


def finish_arm(phase_dir, directory, record):
    copied = publish_final(phase_dir, record["name"], record.get("checkpoint", ""))
    if copied:
        record["final_model"] = copied
    return remember_arm(directory, record)


class PhaseClock:
    """Accumulate the phase driver's wall time across resumes."""

    def __init__(self, directory):
        self.path = Path(directory) / "progress.json"
        self.started = time.perf_counter()
        self.body = read_json(self.path) if self.path.is_file() else {}
        self.stored = float(self.body.get("phase_seconds") or 0)

    def seconds(self):
        return self.stored + (time.perf_counter() - self.started)

    def update(self, **fields):
        self.body.update(fields)
        self.body["phase_seconds"] = self.seconds()
        write_json(self.path, self.body)

def run_summary(record):
    summary = {key: record.get(key) for key in (
        "name", "seed", "status", "steps", "epochs", "accuracies", "time", "costs", "storage_bytes",
        "trainable_parameters", "retired_share", "adapter_plan", "checkpoint", "final_model", "final_models", "metric_std")}
    for key in ("k_E", "k_G"):
        if key in record:
            summary[key] = record[key]
    if record.get("seed_runs"):
        summary["seeds"] = [run_summary(run) for run in record["seed_runs"]]
    return summary


def annotate(details, clock, runs, final_model):
    """Add the phase clock, per-arm epochs and scores, and the comparison model."""
    details = dict(details)
    details["time"] = {"phase_seconds": clock.seconds(),
                       "arms": {run["name"]: run.get("time") for run in runs}}
    details["runs"] = [run_summary(run) for run in runs]
    details["final_model"] = final_model
    return details


def save_report(directory, phase, details, context):
    """Publish tables first and the completed JSON report last."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
    saved = read_json(path) if path.is_file() else None
    if saved:
        details, context = saved["details"], saved["context"]
    def flat(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                yield from flat(child, f"{key}/{name}".strip("/"))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from flat(child, f"{key}/{index}")
        else:
            yield key, value
    rows = [("phase", key, value) for key, value in flat({key: value for key, value in details.items() if key != "runs"})]
    reported = []
    for run in details.get("runs", ()):
        reported.append(run)
        reported.extend(run.get("seeds", ()))
    for run in reported:
        rows.extend((run["name"], key, value) for key, value in flat({key: value for key, value in run.items() if key != "seeds"}))
    with (directory / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "metric", "value"))
        writer.writerows(rows)
    lines = [f"# Phase {phase}", "", "| run | metric | value |", "|---|---|---:|"]
    lines.extend(f"| {name} | {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |" for name, key, value in rows)
    if details.get("reason"):
        lines += ["", str(details["reason"])]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if saved is None:
        progress = read_json(directory / "progress.json")
        write_json(path, {"phase": phase, "status": "completed", "identity": progress["identity"],
                          "details": details, "context": context})
    return context


def compare_arms(left, right, split="test"):
    """Paired article bootstrap after averaging the declared seeds, never selecting on test."""
    from ..utils.numerics import chrf, paired_bootstrap, rouge_l
    def values(arm, language, metric):
        runs = arm.get("seed_runs", [arm])
        by_seed = {}
        for run in runs:
            path = Path(run["attempt"]) / "predictions" / f"{language}-{split}-plain.json"
            rows = read_json(path)
            by_seed[run["seed"]] = {row["id"]: metric(row["prediction"], row["target"]) for row in rows}
        identifiers = set(next(iter(by_seed.values())))
        if any(set(rows) != identifiers for rows in by_seed.values()):
            raise ValueError("Paired seed predictions have different article IDs")
        return by_seed, sorted(identifiers)
    result = {}
    for language in ("english", "hindi"):
        for name, metric in (("chrf", chrf), ("rouge_l", lambda a, b: 100 * rouge_l(a, b))):
            a, ids = values(left, language, metric)
            b, other = values(right, language, metric)
            if ids != other or set(a) != set(b):
                raise ValueError("Comparisons require matched seeds and article IDs")
            averaged_a = [statistics.mean(seed[i] for seed in a.values()) for i in ids]
            averaged_b = [statistics.mean(seed[i] for seed in b.values()) for i in ids]
            summary = paired_bootstrap(averaged_a, averaged_b, 1000, 0)
            summary["counts"] = abs(summary["delta"]) >= 1 and (summary["low"] > 0 or summary["high"] < 0)
            summary["per_seed"] = {str(seed): statistics.mean(a[seed][i] - b[seed][i] for i in ids) for seed in a}
            deltas = list(summary["per_seed"].values())
            summary["seed_std"] = statistics.stdev(deltas) if len(deltas) > 1 else None
            result[f"{language}/{name}"] = summary
    return {"left": left["name"], "right": right["name"], "split": split, "metrics": result}
