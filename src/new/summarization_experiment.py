"""Run the Idea_Refined_2 phases: mT5-small learns English summaries, then Hindi, with a fixed English replay pool.

python -m src.new.summarization_experiment --output results/code/summarization-new [--phase N] [--protocol pilot|smoke]"""

from pathlib import Path
import argparse

from .datasets.dataset_instruction import SummarizationDataset
from .datasets.dataset_sources import xlsum_directory
from .models.model_mt5 import MT5SmallModel
from .models.model_seq2seq import load_tokenizer
from .phases import phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7
from .phases.settings import protocol_settings
from .utils.numerics import write_json

PHASES = (phase0.run, phase1.run, phase2.run, phase3.run, phase4.run, phase5.run, phase6.run, phase7.run)


def prepare(settings):
    """The fixed subsets per language (2,000 / 200 / 500) and the 500-article English pool, tokenized."""
    counts = {"train": settings.train_rows, "validation": settings.validation_rows, "test": settings.test_rows}
    dataset = SummarizationDataset.xlsum(xlsum_directory(), settings.seed, counts, settings.pool_rows)
    return dataset.tokenize(load_tokenizer(MT5SmallModel.model_id), prefix=settings.task_prefix)


def run_phases(root, dataset, settings, only=None):
    """Finished phases are reused from their reports. A failed scientific gate stops the later phases."""
    root, context = Path(root), {}
    for index in range(8):
        if only is not None and index > only:
            break
        if index >= 2 and settings.protocol == "scientific" and context["phase1"]["gate"] != "continue":
            write_json(root / "progress.json", {"status": "stopped_gate", "gate": context["phase1"]["gate"],
                                                "last_phase": index - 1})
            break
        # --phase N stops after phase N; earlier phases are read from their reports, or run first.
        context.update(PHASES[index](root, dataset, settings, context))
        write_json(root / "progress.json", {"status": "running" if index < 7 else "completed", "last_phase": index})
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("scientific", "pilot", "smoke"), default="scientific")
    parser.add_argument("--output", default=".")
    parser.add_argument("--phase", type=int, choices=range(8))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, help="Training passes per language (default 4)")
    args = parser.parse_args(argv)
    settings = protocol_settings(args.protocol, args.device, args.epochs)
    run_phases(Path(args.output).resolve(), prepare(settings), settings, args.phase)
    print(Path(args.output).resolve(), flush=True)


if __name__ == "__main__":
    main()
