"""Real CLINC memorization diagnostic; validation is reported separately from training.

Run with python -m src.learning_sanity --output results/learning-sanity/attempt-1.
The output must be new. This diagnostic is not a continual-learning comparison.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from src.datasets.dataset_instruction import InstructionDataset
from src.models.model_causal_lm import CausalLMSettings
from src.models.model_smollm2 import SmolLM2Model
from src.pipelines.memory_baselines import MemoryBaselineConfig, MemoryBaselinePipeline
from src.utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec
from src.utils.state import write_json


def diagnostics(model, dataset, records, batch_size):
    """Persist actual generated strings as well as strict accuracy and target loss."""
    generation = {"max_new_tokens": 16, "do_sample": False, "seed": 17}
    metrics = model.evaluate(records, {"batch_size": batch_size, "generation": generation})
    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        generated = model.predict(dataset.collate_prompts(batch), generation)
        for record, response in zip(batch, generated["texts"]):
            predictions.append({"id": record.sample_id, "split": record.split,
                                "messages": record.inputs["messages"],
                                "target": record.target["text"], "prediction": response,
                                "exact_match": response.strip() == record.target["text"].strip()})
    return {"metrics": metrics, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(Path("src/datasets/data/clinc_oos_domains").glob("*/records.jsonl"))
    if len(sources) != 1:
        raise ValueError("Expected one pinned CLINC domain snapshot")
    rows = [json.loads(line) for line in sources[0].read_text(encoding="utf-8").splitlines()]
    selected = []
    for split, count in (("train", 4), ("validation", 2)):
        for label in ("transfer", "balance"):
            members = [row for row in rows if row["split"] == split and row["class"] == label][:count]
            if len(members) != count:
                raise ValueError(f"Missing {split} examples for {label}")
            selected.extend(members)
    source = output / "records.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    dataset = InstructionDataset(evaluation_split="validation")
    raw, _ = dataset.load({"path": source, "dataset_id": "clinc-learning-sanity"})
    settings = CausalLMSettings(mode="lora", rank=8, alpha=16, seed=17,
                               learning_rate=1e-3, max_length=128, max_batch_size=4,
                               working_bytes=1024**3, max_score_bytes=256 * 1024**2,
                               compile=False)
    model = SmolLM2Model(settings, device=args.device)
    pipeline = MemoryBaselinePipeline(MemoryBaselineConfig("new_only", 1., 0, False), model)
    _, preprocessing = pipeline.bind_instruction_dataset(dataset, raw)
    train = dataset.training_view("banking")
    validation = dataset.evaluation_view("banking")
    if len(train) != 8 or len(validation) != 4:
        raise ValueError("Preprocessing dropped diagnostic examples; inspect preprocessing")
    batch = dataset.collate(train[:4])
    for record, labels in zip(batch.records, batch.payload["labels"]):
        target = torch.tensor(record.target["ids"])
        if not torch.equal(labels[labels != -100], target):
            raise ValueError("Response supervision differs from expected target tokens")
    spec = RunSpec(task_ids=("banking",), seed=17, batch_size=4,
                   max_steps=args.steps, max_attempts=args.steps, max_tokens=1_000_000,
                   passes=args.steps, exhaustion="repeat_admitted", reduction="per_example_mean",
                   replay=ReplayBudget(0, 0, 0),
                   resources=ResourceBudget(8 * 1024**3, 4 * 1024**3, 10 * 1024**3),
                   checkpoint_every=10, resume="auto",
                   identity={"purpose": "memorization-diagnostic", "source": str(sources[0]),
                             "settings": asdict(settings), "test_used": False},
                   evaluation_spec={"batch_size": 4, "generation": {
                       "max_new_tokens": 16, "do_sample": False, "seed": 17}})
    write_json(output / "diagnostic-config.json", {
        "settings": asdict(settings), "source": str(sources[0]),
        "preprocessing": preprocessing, "train_ids": [r.sample_id for r in train],
        "validation_ids": [r.sample_id for r in validation],
        "torch": torch.__version__, "device": str(model.device),
        "note": "Training accuracy measures memorization; validation is not a test result."})
    try:
        model.activate()
        before = {"train": diagnostics(model, dataset, train, 4),
                  "validation": diagnostics(model, dataset, validation, 4)}
        write_json(output / "before.json", before)
        print("before", {k: v["metrics"] for k, v in before.items()}, flush=True)
        pipeline.run(RunContext(model, dataset, spec, output))
        model.activate()
        after = {"train": diagnostics(model, dataset, train, 4),
                 "validation": diagnostics(model, dataset, validation, 4)}
        write_json(output / "after.json", after)
        write_json(output / "gate.json", {
            "memorization_passed": after["train"]["metrics"]["exact_match"] == 1.,
            "train_nll_decreased": after["train"]["metrics"]["negative_nll"] >
                                   before["train"]["metrics"]["negative_nll"],
            "validation_exact_match": after["validation"]["metrics"]["exact_match"],
            "scientific_result": False})
        print("after", {k: v["metrics"] for k, v in after.items()}, flush=True)
    finally:
        model.offload()


if __name__ == "__main__":
    main()
