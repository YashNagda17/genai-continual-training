"""One serving timing pass on fixed test articles, without warm-up."""

from pathlib import Path
import statistics
import time

import torch

from ..utils.state import read_json, write_json
from .settings import generation_spec


def _finished(path, *, checkpoint_reuse=False):
    saved = read_json(path)
    return (saved.get("status") == "completed" and saved.get("resident") is True
            and (not checkpoint_reuse or saved.get("checkpoint_reuse") is True))


def load_model(factory, checkpoint, directory, merged=False):
    model = factory()
    try:
        model.bind_runtime(directory, None)
        if checkpoint:
            model.load(checkpoint, read_json(Path(checkpoint) / "metadata.json")["state_kind"])
        if not merged:
            return model
        copy = model.merged_copy()
    except BaseException:
        model.release()
        raise
    model.release()
    return copy


def measure(path, dataset, rows, settings, choose, *, checkpoint_reuse=False):
    saved = read_json(path) if path.is_file() else {"requests": [], "status": "running", "resident": True}
    reusable = not checkpoint_reuse or saved.get("checkpoint_reuse") is True
    if saved.get("status") == "completed" and saved.get("resident") and reusable:
        return saved
    if not saved.get("resident") or not reusable:
        saved = {"requests": [], "status": "running", "resident": True}
    saved["checkpoint_reuse"] = checkpoint_reuse
    spec = generation_spec(settings)
    total = sum(len(records) for records in rows.values())
    print(f"serving {path.stem} {len(saved['requests'])}/{total} articles", flush=True)
    for language, records in rows.items():
        done = sum(row["language"] == language for row in saved["requests"])
        for record in records[done:]:
            started = time.perf_counter()
            model = choose(language)
            model.activate()
            result = model.predict(dataset.collate_sources((record,)), spec)
            saved["requests"].append({"id": record.sample_id, "language": language,
                "seconds": time.perf_counter() - started, "length": len(result["token_ids"][0]),
                "peak_device_bytes": result["cost"].get("peak_device_allocated_bytes", 0)})
            write_json(path, saved)
            done_now = len(saved["requests"])
            if total and (done_now == total or done_now * 20 // total != (done_now - 1) * 20 // total):
                print(f"serving {path.stem} {done_now}/{total} articles", flush=True)
        if done < len(records) and hasattr(model, "release_evaluation_memory"):
            model.release_evaluation_memory()
    saved["summary"] = {}
    for language in rows:
        requests = [row for row in saved["requests"] if row["language"] == language]
        timings = torch.tensor([row["seconds"] for row in requests], dtype=torch.float64)
        saved["summary"][language] = {"articles": len(rows[language]), "repeats": 1, "warmup": False, "batch_size": 1,
            "median_seconds": statistics.median(timings.tolist()), "p95_seconds": float(torch.quantile(timings, 0.95)),
            "mean_summary_tokens": statistics.mean(row["length"] for row in requests),
            "peak_device_bytes": max(row["peak_device_bytes"] or 0 for row in requests)}
    saved["status"] = "completed"
    write_json(path, saved)
    return saved


def benchmark(directory, dataset, factory, settings, checkpoints, reference=None):
    directory.mkdir(parents=True, exist_ok=True)
    rows = {language: tuple(dataset.views.get(f"{language}/test", ()))[:200] or tuple(
                row for row in dataset.views["test"] if row.task_id == language)[:200]
            for language in ("english", "hindi")}
    manifest = {"sources": checkpoints, "reference": reference, "generation": generation_spec(settings),
                "repeats": 1, "warmup": False,
                "articles": {language: [row.sample_id for row in records] for language, records in rows.items()}}
    path = directory / "manifest.json"
    if path.is_file() and read_json(path) != manifest:
        raise ValueError("Serving inputs changed; use a new attempt")
    write_json(path, manifest)
    results = {}
    for name, checkpoint in checkpoints.items():
        for merged in ((True,) if checkpoint is None else (False, True)):
            key = name + ("/merged" if merged else "/unmerged")
            path = directory / (key.replace("/", "-") + ".json")
            if path.is_file() and _finished(path):
                results[key] = read_json(path)["summary"]
                continue
            model = load_model(factory, checkpoint, directory, merged)
            model.activate()
            try:
                results[key] = measure(path, dataset, rows, settings, lambda language: model)["summary"]
            finally:
                model.release()
    if reference:
        for merged in (False, True):
            key = "two_adapter/" + ("merged" if merged else "swap")
            path = directory / (key.replace("/", "-") + ".json")
            if path.is_file() and _finished(path, checkpoint_reuse=not merged):
                results[key] = read_json(path)["summary"]
                continue
            models = {}
            try:
                if merged:
                    for language, checkpoint in reference.items():
                        models[language] = load_model(factory, checkpoint, directory, True)
                        models[language].activate()
                    choose = lambda language: models[language]
                else:
                    models["swap"] = load_model(factory, None, directory)
                    models["swap"].activate()
                    active_checkpoint = None
                    checkpoints_by_language = {language: Path(checkpoint).resolve()
                                               for language, checkpoint in reference.items()}
                    kinds = {checkpoint: read_json(checkpoint / "metadata.json")["state_kind"]
                             for checkpoint in set(checkpoints_by_language.values())}
                    def choose(language):
                        nonlocal active_checkpoint
                        checkpoint = checkpoints_by_language[language]
                        if checkpoint != active_checkpoint:
                            models["swap"].load(checkpoint, kinds[checkpoint])
                            active_checkpoint = checkpoint
                        return models["swap"]
                results[key] = measure(path, dataset, rows, settings, choose,
                                       checkpoint_reuse=not merged)["summary"]
            finally:
                for model in models.values():
                    model.release()
    return results
