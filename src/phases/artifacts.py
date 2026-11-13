from pathlib import Path
from dataclasses import asdict, is_dataclass
import csv
import time

from ..utils.state import read_json, write_json


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
              "scores": list(pipeline.policy.get("scores") or []),
              "update_seconds": costs["update_seconds"], "trainable_parameters": pipeline.model.info.trainable_parameters,
              "storage_bytes": sum(file.stat().st_size for file in Path(output).rglob("*") if file.is_file()), **fields}
    record["accuracies"] = accuracies_of(record["metrics"])
    plan = getattr(pipeline, "adapter_plan", None)
    record["adapter_plan"] = asdict(plan) if is_dataclass(plan) else None
    return record


def remember_arm(directory, record):
    write_json(Path(directory) / "arm.json", record)
    return record


def finish_arm(directory, record):
    record["final_model"] = record["checkpoint"] or None
    return remember_arm(directory, record)


class PhaseClock:
    def __init__(self):
        self.started = time.perf_counter()

    def seconds(self):
        return time.perf_counter() - self.started


def run_summary(record):
    summary = {key: record.get(key) for key in (
        "name", "seed", "status", "steps", "epochs", "accuracies", "scores", "time", "costs", "storage_bytes",
        "trainable_parameters", "retired_share", "adapter_plan", "checkpoint", "final_model")}
    for key in ("k_E", "k_G"):
        if key in record:
            summary[key] = record[key]
    return summary


def annotate(details, clock, runs, final_model):
    details = dict(details)
    details["time"] = {"phase_seconds": clock.seconds(), "arms": {run["name"]: run.get("time") for run in runs}}
    details["runs"] = [run_summary(run) for run in runs]
    details["final_model"] = final_model
    return details


def save_report(directory, phase, details, context):
    """Publish tables first and the completed JSON report last."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
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
    for run in details.get("runs", ()):
        rows.extend((run["name"], key, value) for key, value in flat(run))
    with (directory / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "metric", "value"))
        writer.writerows(rows)
    lines = [f"# Phase {phase}", "", "| run | metric | value |", "|---|---|---:|"]
    lines.extend(f"| {name} | {key} | {str(value).replace('|', '/').replace(chr(10), ' ')} |" for name, key, value in rows)
    if details.get("reason"):
        lines += ["", str(details["reason"])]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(path, {"phase": phase, "status": "completed", "details": details, "context": context})
    return context


def compare_arms(left, right, split="test"):
    """Compare matched articles from each arm's single run."""
    from ..utils.numerics import chrf, paired_bootstrap, rouge_l
    def values(arm, language, metric):
        path = Path(arm["attempt"]) / "predictions" / f"{language}-{split}-plain.json"
        rows = read_json(path)
        scores = {row["id"]: metric(row["prediction"], row["target"]) for row in rows}
        if len(scores) != len(rows):
            raise ValueError("Comparisons require unique article IDs")
        return scores
    result = {}
    for language in ("english", "hindi"):
        for name, metric in (("chrf", chrf), ("rouge_l", lambda a, b: 100 * rouge_l(a, b))):
            a, b = values(left, language, metric), values(right, language, metric)
            if not a or set(a) != set(b):
                raise ValueError("Comparisons require matched, nonempty article IDs")
            ids = sorted(a)
            summary = paired_bootstrap([a[i] for i in ids], [b[i] for i in ids], 1000, 0)
            summary["counts"] = abs(summary["delta"]) >= 1 and (summary["low"] > 0 or summary["high"] < 0)
            result[f"{language}/{name}"] = summary
    return {"left": left["name"], "right": right["name"], "split": split,
            "uncertainty": "article_bootstrap", "metrics": result}
