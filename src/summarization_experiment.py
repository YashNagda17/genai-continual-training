"""Run the registered summarization phases with one saved report per phase."""

from pathlib import Path
import argparse

from .models.model_mt5 import MT5SmallModel
from .models.model_seq2seq import Seq2SeqSettings
from .phases.phase0 import run as run_phase0
from .phases.phase1 import run as run_phase1
from .phases.phase2 import run as run_phase2
from .phases.phase3 import run as run_phase3
from .phases.phase4 import run as run_phase4
from .phases.phase5 import run as run_phase5
from .phases.phase6 import run as run_phase6
from .phases.phase7 import run as run_phase7
from .phases.settings import PhaseSettings, align_settings
from .phases.data import build_smoke_dataset, build_smoke_model, choose_ids, xlsum_directory
from .phases.support import read_context
from .utils.state import Checkpoints, read_json, write_json
from .datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from .datasets.dataset_sources import load_source

PHASES = (run_phase0, run_phase1, run_phase2, run_phase3, run_phase4, run_phase5, run_phase6, run_phase7)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the pilot instead of the scientific protocol")
    parser.add_argument("--protocol", choices=("smoke", "pilot", "scientific"), default="scientific")
    parser.add_argument("--output", default=".")
    parser.add_argument("--phase", type=int, choices=range(8))
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def selected_phases(root, phase):
    """Continue from the last phase, or run Phase 0 through Phase 7 when nothing has started."""
    if phase is not None:
        return (phase,)
    path = Path(root) / "progress.json"
    if not path.is_file():
        return tuple(range(8))
    last = read_json(path).get("last_phase")
    if not isinstance(last, int) or not 0 <= last <= 7:
        return tuple(range(8))
    start = last if not (Path(root) / f"phase{last}" / "report.json").is_file() else last + 1
    return tuple(range(start, 8))


def prepare(protocol, device, settings=None):
    if protocol == "smoke":
        factory = build_smoke_model
        settings = PhaseSettings(max_new_tokens=8, shrink_tolerance=100.0)
    else:
        settings = settings or PhaseSettings(protocol=protocol, batch_size=16, analysis_batch=8,
                        max_new_tokens=64, stability_threshold=0.5, sample_size=64)
        def factory():
            return MT5SmallModel(Seq2SeqSettings(rank=16, alpha=32.0, max_batch_size=32,
                                                seed=settings.seed), device=device or "auto")
    model = factory()
    try:
        if protocol == "smoke":
            dataset = build_smoke_dataset(model)
        else:
            dataset, raw = load_source(xlsum_directory(), SummarizationDataset)
            views = dataset.split(raw, {"seed": settings.seed})
            selected, subsets = {}, {}
            for split, count in (("train", 2000), ("validation", 200), ("test", 500)):
                identifiers = set()
                for language in ("english", "hindi"):
                    rows = [row for row in views[split] if row.task_id == language and
                            row.inputs["text"].strip() and row.target["text"].strip()]
                    ids = choose_ids(rows, count, settings.seed)
                    subsets[f"{language}/{split}"] = ids
                    identifiers.update(ids)
                selected[split] = tuple(row for row in views[split] if row.sample_id in identifiers)
            dataset.preprocess(selected, Seq2SeqPreprocessing(**model.preprocessing_kwargs()))
            for name, ids in subsets.items():
                dataset.expose(name, ids)
            dataset.expose("english/pool", choose_ids(dataset.training_view("english"), 500, settings.seed + 1))
        return dataset, factory, align_settings(settings, model)
    finally:
        model.release()


def run_phases(root, dataset, model_factory, settings, phase=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selected = selected_phases(root, phase)
    context = read_context(root, dataset, settings, before=selected[0]) if selected else {}
    with Checkpoints(root, "phase-driver").lock():
        for index in selected:
            if settings.protocol == "scientific" and index >= 2 and context.get("phase1", {}).get("gate") != "continue":
                completed = [i for i in range(8) if (root / f"phase{i}" / "report.json").is_file()]
                write_json(root / "progress.json", {"status": "stopped_gate", "gate": context.get("phase1", {}).get("gate"),
                                                   "completed_phases": completed, "last_phase": index - 1})
                break
            context.update(PHASES[index](root, dataset, model_factory, settings, context))
            completed = [i for i in range(8) if (root / f"phase{i}" / "report.json").is_file()]
            write_json(root / "progress.json", {"status": "completed" if index == selected[-1] else "running",
                                               "completed_phases": completed, "last_phase": index})
    return context


def main(argv=None):
    args = parse_args(argv)
    protocol = "pilot" if args.pilot else args.protocol
    output = Path(args.output).resolve()
    dataset, factory, settings = prepare(protocol, args.device)
    if protocol != "smoke" and not (output / "prepared" / "manifest.json").is_file():
        dataset.save(output / "prepared")
    run_phases(output, dataset, factory, settings, args.phase)
    print(output, flush=True)
    return output


if __name__ == "__main__":
    main()
