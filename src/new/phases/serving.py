"""The serving bill: one timed request per fixed test article at batch size 1, without warm-up."""

import statistics
import time

import torch

from ..datasets.dataset_instruction import LANGUAGES
from ..utils.numerics import peak_mb, read_json, resolve_device, synchronize, write_json
from .support import new_model


def load(settings, checkpoint, merged):
    model = new_model(settings)
    model.load(checkpoint) if checkpoint else model.build()
    if merged:
        model.merge()
    return model


def measure(dataset, settings, rows, choose):
    summary, device = {}, resolve_device(settings.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for language, records in rows.items():
        times, lengths = [], []
        for record in records:
            synchronize(device)
            start = time.perf_counter()
            _, ids = choose(language).generate(dataset.collate([record], labels=False), settings.max_new_tokens,
                                               settings.no_repeat)
            synchronize(device)
            times.append(time.perf_counter() - start)
            lengths.append(len(ids[0]))
        summary[language] = {"articles": len(records), "median_seconds": statistics.median(times),
                             "p95_seconds": float(torch.quantile(torch.tensor(times, dtype=torch.float64), 0.95)),
                             "mean_summary_tokens": statistics.mean(lengths), "peak_device_mb": peak_mb(device)}
    return summary


def benchmark(directory, dataset, settings, checkpoints, reference=None):
    """Merged and unmerged adapters; with `reference`, the two-adapter route that swaps per request."""
    path = directory / "serving.json"
    if path.is_file():
        return read_json(path)
    rows = {language: dataset.view(f"{language}/test")[:settings.serving_rows] for language in LANGUAGES}
    results = {}
    for name, checkpoint in checkpoints.items():
        for merged in ((True,) if checkpoint is None else (False, True)):
            model = load(settings, checkpoint, merged)
            try:
                results[f"{name}/{'merged' if merged else 'unmerged'}"] = measure(dataset, settings, rows, lambda _: model)
            finally:
                model.release()
    if reference:
        models = {language: load(settings, checkpoint, True) for language, checkpoint in reference.items()}
        results["two_adapter/merged"] = measure(dataset, settings, rows, models.get)
        for model in models.values():
            model.release()
        swap = new_model(settings)

        def choose(language):
            swap.load(reference[language])  # Only the adapter weights are copied between requests.
            return swap
        results["two_adapter/swap"] = measure(dataset, settings, rows, choose)
        swap.release()
    write_json(path, results)
    return results
