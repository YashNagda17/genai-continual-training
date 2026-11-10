"""Bounded CUDA checks on prepared data, separate from experiment phases."""

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import uuid

# These checks use existing local assets only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from .datasets.dataset_instruction import SummarizationDataset
from .models.model_mt5 import MT5SmallModel
from .models.model_seq2seq import Seq2SeqSettings
from .utils.contracts import Objective, TrainingBatch, TrainLimits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("training", "generation"), default="training")
    parser.add_argument("--prepared", default="results/code/summarization-passes5/prepared")
    parser.add_argument("--checkpoint", help="Required for generation; training uses a fresh disposable adapter")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--examples", type=int, default=2, help="Generation articles (1 to 4)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not 1 <= args.batches <= 10 or not 1 <= args.examples <= 4 or not 1 <= args.batch_size <= 32:
        parser.error("Use 1-10 batches, 1-4 generation articles, and batch size 1-32")
    if args.mode == "generation" and not args.checkpoint:
        parser.error("Generation requires --checkpoint pointing to a saved English adapter")
    if args.mode == "training" and args.checkpoint:
        parser.error("Training checks use a fresh adapter; do not supply --checkpoint")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This check requires CUDA and will not silently use CPU")
    dataset = SummarizationDataset()
    dataset.load({"path": args.prepared})
    prep = dataset.manifest["preprocessing"]
    if prep["identity"]["model"] != MT5SmallModel.model_id:
        parser.error("Prepared data must belong to mT5-small")
    output = Path(args.output or f"results/performance/tiny-check-{uuid.uuid4().hex[:8]}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("Choose a new output file")
    model = MT5SmallModel(Seq2SeqSettings(max_source_length=prep["max_source"],
                                        max_target_length=prep["max_target"]), device=str(device))
    model.bind_dataset(dataset)
    report = {"mode": args.mode, "device": str(model.device), "gpu": torch.cuda.get_device_name(device),
              "torch": str(torch.__version__), "prepared": str(Path(args.prepared).resolve()),
              "prefix": prep.get("prefix"), "precision": "float32", "rows": [],
              "note": "Tiny diagnostic; synchronized wall times are not GPU-kernel-only times. No phase execution."}
    with tempfile.TemporaryDirectory(prefix=".tiny-check-", dir=output.parent) as temporary:
        model.bind_runtime(Path(temporary), None)
        try:
            started = time.perf_counter()
            if args.checkpoint:
                checkpoint = Path(args.checkpoint)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                if metadata["model"] != MT5SmallModel.model_id:
                    parser.error("Checkpoint must belong to mT5-small")
                model.load(checkpoint, metadata["state_kind"])
                report["checkpoint"] = str(checkpoint.resolve())
            model.activate()
            torch.cuda.synchronize(device)
            report["model_load_seconds"] = time.perf_counter() - started
            print(f"{args.mode} check | {report['gpu']} | {model.device} | fp32", flush=True)
            if args.mode == "training":
                rows = dataset.training_view("english")
                if len(rows) < args.batches * args.batch_size:
                    parser.error("Not enough English training rows for the requested batches")
                for index in range(args.batches):
                    started = time.perf_counter()
                    batch = dataset.collate(rows[index * args.batch_size:(index + 1) * args.batch_size])
                    item = TrainingBatch(batch, ("current",) * len(batch.records))
                    step, = model.train(1, batch_factory=lambda: (item,), objective=Objective("per_example_mean"),
                                        hooks=None, limits=TrainLimits(1, 1, batch.valid_tokens))
                    row = {"batch": index + 1, "examples": len(batch.records), "success": step.success,
                           "update_seconds": step.cost["synchronized_wall_seconds"],
                           "wall_seconds": time.perf_counter() - started,
                           "peak_mib": step.cost["peak_device_allocated_bytes"] / 2**20}
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    if not step.success:
                        raise RuntimeError(step.skip_reason)
                warm = report["rows"][1:] or report["rows"]
                report["warm_mean_update_seconds"] = statistics.mean(row["update_seconds"] for row in warm)
                print(f"After startup: {report['warm_mean_update_seconds']:.3f} seconds/update", flush=True)
            else:
                rows = dataset.views["english/validation"][:args.examples]
                if len(rows) != args.examples:
                    parser.error("Not enough English validation articles")
                started = time.perf_counter()
                model._script_masks()
                torch.cuda.synchronize(device)
                report["cold_script_masks_seconds"] = time.perf_counter() - started
                # Warm decoder/sentinel setup before comparing the two paths.
                model.predict(dataset.collate_sources(rows[:1]), {"max_new_tokens": 1})
                for index, record in enumerate(rows):
                    results = {}
                    modes = (False, True) if index % 2 == 0 else (True, False)
                    for early in modes:
                        result = model.predict(dataset.collate_sources((record,)),
                            {"max_new_tokens": 64, "no_repeat_ngram_size": 3, "stop_on_eos": early})
                        results[early] = result
                    if results[False]["token_ids"] != results[True]["token_ids"]:
                        raise RuntimeError("Output tokens changed between the two generation paths")
                    row = {"id": record.sample_id, "equal_tokens": True,
                           "fixed_seconds": results[False]["cost"]["synchronized_wall_seconds"],
                           "early_seconds": results[True]["cost"]["synchronized_wall_seconds"],
                           "fixed_steps": results[False]["cost"]["decoder_steps"],
                           "early_steps": results[True]["cost"]["decoder_steps"]}
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
        finally:
            model.release()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
