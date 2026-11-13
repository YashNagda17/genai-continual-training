"""Run the registered summarization phases with one saved report per phase."""

import os

# The Intel driver otherwise waits for the GPU by spinning one core. Read before any XPU context exists.
os.environ.setdefault("EventHostSynchronizeLinuxUserFenceKmdWait", "1")

from pathlib import Path
import argparse
from functools import partial
from dataclasses import replace

from .models.model_mbart import MBart25Summarizer
from .models.model_seq2seq import Seq2SeqSettings
from .phases.phase import PHASES
from .phases.settings import PHASE_LAYOUT, PHASE_NAMES, PhaseSettings, align_settings
from .phases.data import build_smoke_dataset, build_smoke_model, choose_ids, xlsum_directory
from .phases.support import run_session
from .utils.state import Artifacts, write_json
from .datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from .datasets.dataset_sources import load_source

# Match the course report: forgetting, replay, blocks, compression, distillation.
RUN_ORDER = tuple(PHASES)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the pilot instead of the scientific protocol")
    parser.add_argument("--protocol", choices=("smoke", "pilot", "scientific"), default="scientific")
    parser.add_argument("--output", default=".")
    parser.add_argument("--check", action="store_true",
                        help="Check all five phases on real mBART with small batches and saved outputs")
    parser.add_argument("--examples", type=int, default=None,
                        help="Training rows per language for --check or --pilot (check default: 50)")
    parser.add_argument("--device", default=None, help="CUDA/XPU auto-detection; smoke defaults to CPU")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Reduce mBART training activation memory at the cost of recomputation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training passes per language (default: 4; nondefault values require --pilot or smoke)")
    args = parser.parse_args(argv)
    if args.check and args.protocol == "smoke":
        parser.error("--check uses the real model; omit --protocol smoke")
    if args.examples is not None and (args.examples < 5 or not (args.check or args.pilot or args.protocol == "pilot")):
        parser.error("--examples must be at least 5 and requires --check or --pilot")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.epochs not in (None, PhaseSettings().english_passes) and not (args.check or args.pilot) and args.protocol == "scientific":
        parser.error("Nondefault --epochs requires --pilot or --protocol smoke")
    return args


def prepare(protocol, device, settings=None, *, epochs=None, gradient_checkpointing=False, examples=None, check=False):
    if protocol == "smoke":
        factory = partial(build_smoke_model, device=device or "cpu")
        # The tiny smoke model takes at most four rows per update and runs every conditional arm.
        settings = PhaseSettings(batch_size=2, analysis_batch=2, english_passes=1, hindi_passes=1,
                                 max_new_tokens=8, shrink_tolerance=100.0, force_conditional=True)
    else:
        settings = settings or PhaseSettings(protocol=protocol, batch_size=16, analysis_batch=16,
                        max_new_tokens=256, stability_threshold=0.5, sample_size=64)
        if check:
            settings = replace(settings, batch_size=2, analysis_batch=2, english_passes=1, hindi_passes=1,
                               max_new_tokens=32, force_conditional=True)
        if examples is not None:
            # Gradient maps need two disjoint samples after past-target filtering.
            settings = replace(settings, sample_size=min(settings.sample_size, 8, examples // 4))
        def factory():
            # mBART-large summarizer; English and Hindi share en_XX (see model_mbart).
            # The custom backward uses one gradient buffer; the chunk floor bounds vocabulary scratch memory.
            return MBart25Summarizer(Seq2SeqSettings(rank=16, alpha=32.0, max_batch_size=32, logit_chunk_size=32,
                                                    learning_rate=settings.learning_rate,
                                                    seed=settings.seed), device=device or "auto",
                                      gradient_checkpointing=gradient_checkpointing)
    if epochs is not None:
        settings = replace(settings, english_passes=epochs, hindi_passes=epochs)
    model = factory()
    try:
        if protocol == "smoke":
            dataset = build_smoke_dataset(model)
        else:
            dataset, raw = load_source(xlsum_directory(), SummarizationDataset)
            views = dataset.split(raw, {"seed": settings.seed})
            selected, subsets = {}, {}
            sizes = (("train", examples), ("validation", min(10, examples)), ("test", min(10, examples))) if examples is not None else (
                ("train", 2000), ("validation", 200), ("test", 500))
            for split, count in sizes:
                identifiers = set()
                for language in ("english", "hindi"):
                    rows = [row for row in views[split] if row.task_id == language and
                            row.inputs["text"].strip() and row.target["text"].strip()]
                    ids = choose_ids(rows, count, settings.seed)
                    subsets[f"{language}/{split}"] = ids
                    identifiers.update(ids)
                selected[split] = tuple(row for row in views[split] if row.sample_id in identifiers)
            dataset.preprocess(selected, Seq2SeqPreprocessing(
                **model.preprocessing_kwargs(), prefix=settings.task_prefix or None))
            for name, ids in subsets.items():
                dataset.expose(name, ids)
            dataset.expose("english/pool", choose_ids(dataset.training_view("english"),
                           min(500, examples) if examples is not None else 500, settings.seed + 1))
        return dataset, factory, align_settings(settings, model)
    finally:
        model.release()


def run_phases(root, dataset, model_factory, settings):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("phase[0-9]*")) or (root / "run.json").exists():
        raise FileExistsError(f"Run output already exists: {root}; choose a fresh directory")
    context, completed = {}, []
    with Artifacts(root, "phase-driver").lock(), run_session():
        for index in RUN_ORDER:
            if settings.protocol == "scientific" and index != 1 and context["phase1"]["gate"] != "continue":
                write_json(root / "run.json", {"status": "stopped_gate", "gate": context["phase1"]["gate"],
                                              "completed_phases": completed, "phase_layout": PHASE_LAYOUT})
                break
            print(f"Phase {index}/{len(RUN_ORDER)}: {PHASE_NAMES[index]}", flush=True)
            context.update(PHASES[index](root, dataset, model_factory, settings, context))
            completed.append(index)
        else:
            write_json(root / "run.json", {"status": "completed", "completed_phases": completed,
                                          "phase_layout": PHASE_LAYOUT})
    return context


def main(argv=None):
    args = parse_args(argv)
    protocol = "pilot" if args.pilot or args.check else args.protocol
    examples = args.examples if args.examples is not None else (50 if args.check else None)
    output = Path(args.output).resolve()
    if any((output / name).exists() for name in ("prepared", "run.json")) or any(output.glob("phase[0-9]*")):
        raise FileExistsError(f"Run output already exists: {output}; choose a fresh directory")
    dataset, factory, settings = prepare(protocol, args.device, epochs=args.epochs,
                                         gradient_checkpointing=args.gradient_checkpointing,
                                         examples=examples, check=args.check)
    if protocol != "smoke":
        dataset.save(output / "prepared")
    run_phases(output, dataset, factory, settings)
    print(output, flush=True)
    return output


if __name__ == "__main__":
    main()
