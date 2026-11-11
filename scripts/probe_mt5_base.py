"""Probe eslamxm/mt5-base-finetuned-english on the prepared XL-Sum slice.

Generation, the chunked vocabulary loss, and LoRA come from the seq2seq executor.
The Hindi pass is Hindi-only: the base stays frozen, and no English rows are replayed.

From the repository root:

  python scripts/probe_mt5_base.py --eval-only
  python scripts/probe_mt5_base.py
"""

import os

# The XPU driver reads this when its context is created. It has to be set before Torch.
os.environ.setdefault("EventHostSynchronizeLinuxUserFenceKmdWait", "1")

from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset_instruction import SummarizationDataset
from src.models.model_seq2seq import Seq2SeqLMModel, Seq2SeqSettings
from src.utils.contracts import Objective, TrainLimits, TrainingBatch
from src.utils.numerics import chrf, rouge_l, script_share

MODEL_ID = "eslamxm/mt5-base-finetuned-english"
PREPARED = ROOT / "results/code/summarization-passes5/prepared"
GENERATION = {"max_new_tokens": 64, "no_repeat_ngram_size": 3, "stop_on_eos": True}


class PublishedMT5Base(Seq2SeqLMModel):
    """The published English XL-Sum checkpoint. LoRA is attached by the executor."""

    model_id = MODEL_ID
    model_type = "mt5"
    untie_head = True


def load_dataset(prepared):
    dataset = SummarizationDataset()
    dataset.load({"path": str(prepared)})
    return dataset


def take(dataset, key, count, max_source, max_target):
    rows = dataset.views[key]
    if len(rows) < count:
        raise ValueError(f"{key} has {len(rows)} rows, needed {count}")
    chosen = []
    for record in rows[:count]:
        source = record.inputs["ids"][:max_source]
        target = record.target["ids"][:max_target]
        if len(source) != len(record.inputs["ids"]) or len(target) != len(record.target["ids"]):
            record = replace(record, inputs={**record.inputs, "ids": source},
                             target={**record.target, "ids": target})
        chosen.append(record)
    return chosen


def score_rows(model, dataset, records, batch_size):
    scored = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        generated = model.predict(dataset.collate_sources(chunk), GENERATION)
        for record, text in zip(chunk, generated["texts"]):
            text = text.strip()
            scored.append({
                "id": record.sample_id, "prediction": text, "target": record.target["text"],
                "article": record.inputs["text"], "chrf": chrf(text, record.target["text"]),
                "rouge_l": 100 * rouge_l(text, record.target["text"]),
                "devanagari_share": script_share(text, "devanagari"),
            })
    return scored


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def score_both(model, dataset, rows, batch_size):
    scored = {}
    for language in ("english", "hindi"):
        scored[language] = score_rows(model, dataset, rows[language], batch_size)
        print(f"  {language:8}  chrF {mean(scored[language], 'chrf'):6.2f}   "
              f"ROUGE-L {mean(scored[language], 'rouge_l'):6.2f}   "
              f"Devanagari letters {100 * mean(scored[language], 'devanagari_share'):5.1f}%")
    return scored


def train_hindi(model, dataset, records, batch_size, epochs):
    """Repeat the Hindi slice for the requested epochs. The executor owns the AdamW step."""

    def batches():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            yield TrainingBatch(dataset.collate(chunk), origins=tuple("hindi" for _ in chunk))

    steps = (len(records) + batch_size - 1) // batch_size
    total = steps * epochs
    reports = model.train(
        epochs, batch_factory=batches, objective=Objective("per_example_mean"), hooks=None,
        limits=TrainLimits(total, total, total * batch_size * 512))
    losses = []
    seconds = 0.0
    for index, report in enumerate(reports, start=1):
        seconds += report.cost.get("synchronized_wall_seconds", report.cost.get("wall_seconds", 0.0))
        if not report.success:
            raise RuntimeError(f"Hindi step {index} skipped: {report.skip_reason}")
        loss = float(report.loss_sums["supervised"]) / float(report.loss_counts["supervised"])
        losses.append(loss)
        epoch = (index - 1) // steps + 1
        print(f"  epoch {epoch}/{epochs}  step {(index - 1) % steps + 1:4d}/{steps}  loss {loss:.4f}")
    return {"steps": len(reports), "epochs": epochs, "examples": len(records), "batch_size": batch_size,
            "loss_first": losses[0], "loss_last": losses[-1], "seconds": seconds}


def show_examples(title, scored, count):
    print(f"\n{title}")
    for language in ("english", "hindi"):
        for row in scored[language][:count]:
            print(f"\n[{language}] {row['id']}  chrF {row['chrf']:.1f}")
            print(f"  article:    {row['article'][:240].replace(chr(10), ' ')}")
            print(f"  reference:  {row['target'][:240].replace(chr(10), ' ')}")
            print(f"  predicted:  {row['prediction'][:240].replace(chr(10), ' ')}")


def write_report(path, payload):
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    before, after = payload["before"], payload.get("after")
    lines = [
        "# mT5-base XL-Sum probe",
        "",
        f"Model: `{payload['model']}`",
        f"Parameters: {payload['parameters']:,} total, {payload['trainable']:,} trainable",
        f"Articles per language: {payload['eval_n']}. Prepared prompt prefix: `{payload['prefix']}`",
        "",
        "The published card reports English XL-Sum ROUGE-1 31.7 and ROUGE-L 26.43 on its own",
        "evaluation setup. The numbers below use this project's chrF and whitespace ROUGE-L",
        "on the prepared validation slice.",
        "",
        "| | English chrF | English ROUGE-L | Hindi chrF | Hindi ROUGE-L |",
        "|---|---:|---:|---:|---:|",
        f"| Before Hindi | {before['english']['chrf']:.2f} | {before['english']['rouge_l']:.2f} | "
        f"{before['hindi']['chrf']:.2f} | {before['hindi']['rouge_l']:.2f} |",
    ]
    if after:
        train = payload["hindi_run"]
        lines.append(
            f"| After {train['examples']} Hindi examples | {after['english']['chrf']:.2f} | "
            f"{after['english']['rouge_l']:.2f} | {after['hindi']['chrf']:.2f} | {after['hindi']['rouge_l']:.2f} |")
        lines.extend(["",
                      f"English chrF change: {after['english']['chrf'] - before['english']['chrf']:+.2f}. "
                      f"Hindi chrF change: {after['hindi']['chrf'] - before['hindi']['chrf']:+.2f}.",
                      f"Hindi loss {train['loss_first']:.3f} to {train['loss_last']:.3f} "
                      f"in {train['seconds'] / 60:.1f} minutes, {train['epochs']} epochs, "
                      f"batch {train['batch_size']}. No English replay."])
    (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=ROOT / "results/code/mt5-base-probe")
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--train-rows", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--max-source", type=int, default=256)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.prepared)
    prefix = dataset.manifest["preprocessing"]["prefix"]
    rows = {
        "english": take(dataset, "english/validation", args.eval_n, args.max_source, args.max_target),
        "hindi": take(dataset, "hindi/validation", args.eval_n, args.max_source, args.max_target),
    }
    if not args.eval_only:
        rows["hindi_train"] = take(dataset, "hindi/train", args.train_rows, args.max_source, args.max_target)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    model = PublishedMT5Base(Seq2SeqSettings(
        rank=args.rank, alpha=args.alpha, learning_rate=args.lr, max_batch_size=args.batch_size,
        logit_chunk_size=32, max_source_length=args.max_source, max_target_length=args.max_target,
    ), device=args.device, cache_dir=hub)
    model.model_id = args.model
    model.bind_dataset(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    model.bind_runtime(args.output, None)
    model.activate()
    info = model.info
    print(f"Device {model.device}, float32. {info.total_parameters:,} parameters, "
          f"{info.trainable_parameters:,} trainable. Chunked loss, batch {args.batch_size}.")

    print(f"\nBefore Hindi, {args.eval_n} validation articles per language")
    before = score_both(model, dataset, rows, args.batch_size)
    show_examples("Sample summaries before Hindi", before, args.show)
    training = None
    after = None
    if not args.eval_only:
        print(f"\nHindi-only LoRA: {args.train_rows} examples, {args.epochs} epochs, "
              f"batch {args.batch_size}, lr {args.lr}")
        training = train_hindi(model, dataset, rows["hindi_train"], args.batch_size, args.epochs)
        print("\nAfter Hindi")
        after = score_both(model, dataset, rows, args.batch_size)
        show_examples("Sample summaries after Hindi", after, args.show)
        print(f"\nEnglish chrF change {mean(after['english'], 'chrf') - mean(before['english'], 'chrf'):+.2f}. "
              f"Hindi chrF change {mean(after['hindi'], 'chrf') - mean(before['hindi'], 'chrf'):+.2f}.")
        model.save(args.output / "hindi", "hindi")

    def pack(scored):
        return {language: {"chrf": mean(rows_for_language, "chrf"),
                           "rouge_l": mean(rows_for_language, "rouge_l"),
                           "devanagari_share": mean(rows_for_language, "devanagari_share"),
                           "rows": rows_for_language}
                for language, rows_for_language in scored.items()}

    total, trainable = model.parameter_counts()
    write_report(args.output, {
        "model": args.model, "parameters": total, "trainable": trainable, "prefix": prefix,
        "eval_n": args.eval_n, "device": str(model.device), "dtype": "float32",
        "before": pack(before), "after": None if after is None else pack(after), "hindi_run": training,
    })
    print(f"\nWrote {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
