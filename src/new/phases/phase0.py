"""Phase 0: one update, readable summaries, truncation, and tokens per character."""

import time

from ..utils.numerics import peak_mb, synchronize, write_json
from .support import new_model, phase


def summary_word_coverage(records, tokenizer, limit):
    """Reference-summary words found in the first `limit` article tokens, against the full article."""
    totals = {}
    for record in records:
        article = set(record.text.casefold().split())
        kept = set(tokenizer.decode(tokenizer.encode(record.text)[:limit]).casefold().split())
        bucket = totals.setdefault(record.language, {"rows": 0, "full": 0, "kept": 0})
        words = record.summary.casefold().split()
        bucket["rows"] += 1
        bucket["full"] += sum(word in article for word in words)
        bucket["kept"] += sum(word in article and word in kept for word in words)
    for bucket in totals.values():
        bucket["coverage"] = bucket["kept"] / bucket["full"] if bucket["full"] else None
    return totals


@phase(0)
def run(directory, dataset, settings, context):
    model = new_model(settings)
    try:
        model.build()
        english = dataset.view("english/train")
        synchronize(model.device)
        start = time.perf_counter()
        model.train_step(dataset.collate(english[:settings.batch_size]))
        synchronize(model.device)
        seconds = time.perf_counter() - start
        texts, _ = model.generate(dataset.collate(english[:20], labels=False), settings.max_new_tokens, settings.no_repeat)
        summaries = [{"id": r.id, "target": r.summary, "text": t} for r, t in zip(english, texts)]
        write_json(directory / "summaries.json", summaries)
        model.save(directory / "final" / "batch")
        coverage = summary_word_coverage(english + dataset.view("hindi/train"), model.tokenizer,
                                         model.settings.max_source_length)
        details = {"trained": True, "update_seconds": seconds, "peak_memory_mb": peak_mb(model.device),
                   "summary": summaries[0], "summary_count": len(summaries), "readability_review": "pending",
                   "tokens_per_character": dataset.stats, "truncation": coverage}
    finally:
        model.release()
    return details, {"phase0": details}
