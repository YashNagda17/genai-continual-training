"""One sample run: local SmolLM2, one CLINC epoch, pipeline test, then a few inferences.

From the repository root or from src/:

    python -m src.example
    python src/example.py
    python example.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.combined_pipeline import CombinedPipeline
from src.datasets.dataset_instruction import InstructionDataset
from src.datasets.dataset_sources import ClincOOSSource, load_source
from src.models.model_causal_lm import CausalLMSettings
from src.models.model_smollm2 import SmolLM2Model
from src.pipelines.al_sar import ALSARConfig, ALSARPipeline
from src.pipelines.csrel import CSReLConfig, CSReLPipeline
from src.pipelines.oasis import OASISConfig, OASISPipeline
from src.pipelines.tsr import TSRConfig, TSRPipeline
from src.utils.contracts import ReplayBudget, ResourceBudget, RunContext, RunSpec

OUTPUT = ROOT / "results" / "sample" / "clinc-smollm2"
INFERENCE_EXAMPLES = 4


def clinc_directory():
    """Use the fetched CLINC JSONL, downloading the plus split only when it is absent."""
    root = ROOT / "src" / "datasets" / "data"
    ready = [path for path in (root / "clinc_oos").glob("*") if (path / "records.jsonl").is_file()]
    if ready:
        return max(ready, key=lambda path: (path / "manifest.json").stat().st_mtime)
    return ClincOOSSource().fetch(root)


def build():
    """Load SmolLM2, prepare official CLINC splits, and assemble the combined pipeline."""
    settings = CausalLMSettings(
        mode="lora", rank=8, alpha=16, seed=17,
        max_length=96, max_batch_size=16,
        working_bytes=4 * 1024 ** 3, max_score_bytes=4 * 1024 ** 3,
        compile=True,
    )
    model = SmolLM2Model(settings, device="auto")
    dataset, raw = load_source(clinc_directory(), InstructionDataset)
    pipeline = CombinedPipeline(
        oasis=OASISPipeline(OASISConfig(0.2, 0.0, 1.0, 1e-8, 1.0, 8, "head", "current_only")),
        tsr=TSRPipeline(TSRConfig(0.0, 0.5, 0.0, 2.0, True, "fixed-zero-B", "lora-b", 1, 32, False)),
        al_sar=ALSARPipeline(ALSARConfig(1.0, 4.0, 0.2, 4, "subset", False, True, True, "active_population")),
        csrel=CSReLPipeline(CSReLConfig("csrel_cl", 1, 1, 1, "reference-base", "subset-base", 0.7, 2, False)),
        model=model,
    )
    pipeline.bind_instruction_dataset(dataset, raw)
    source = dataset.manifest.get("source", {})
    spec = RunSpec(
        task_ids=tuple(source.get("task_order") or ("clinc",)),
        seed=17, batch_size=16, max_attempts=20000, max_steps=20000, max_tokens=5_000_000,
        passes=1, exhaustion="repeat_admitted", reduction="per_example_mean",
        replay=ReplayBudget(64, 50_000, 50_000_000),
        resources=ResourceBudget(16 * 1024 ** 3, 16 * 1024 ** 3, 40 * 1024 ** 3),
        checkpoint_every=1000,
        identity={"protocol": "clinc-sample-one-epoch", "adaptations": ["sample"]},
        evaluation_spec={"batch_size": 16, "generation": {"max_new_tokens": 16, "do_sample": False, "seed": 17}},
        resume="never",
    )
    return model, dataset, pipeline, spec


def main():
    """Delete any previous sample output, then train one fresh CLINC pass and generate four intents."""
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    model, dataset, pipeline, spec = build()
    print(f"model SmolLM2 device={model.device} tasks={spec.task_ids}", flush=True)
    result = pipeline.run(RunContext(model, dataset, spec, OUTPUT))
    print(f"status {result.status} steps={result.counters.get('steps')}", flush=True)
    for row in result.metrics:
        print(f"test task={row['task']} {row['metrics']}", flush=True)
    held = dataset.evaluation_view(spec.task_ids[0])[:INFERENCE_EXAMPLES]
    generated = pipeline.predict(dataset.collate_prompts(held), spec.evaluation_spec["generation"])
    for record, text in zip(held, generated["texts"]):
        print(f"gold {record.target['text']}\npred {text.strip()}\n", flush=True)
    return result


if __name__ == "__main__":
    main()
