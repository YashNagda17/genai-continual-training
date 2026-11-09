# Summarization experiments

The active code implements [Idea_Refined_2.md](/Idea_Refined_2.md): mT5-small learns English summarization, then Hindi, with a fixed English replay pool. Only this model family, XL-Sum, its three pipelines, and their behavior tests remain. Paper sources, reviews, literature tools, and writing workflows are preserved.

Use Python 3.11+ with the dependencies in [requirements.txt](/requirements.txt). Torch owns training, gradients and inference. Models detect CUDA/ROCm, then XPU, then CPU; explicit device requests are supported. Every role is capped at 700M total parameters.

```sh
./download_datasets.sh xlsum
./download_model_weights.sh mt5
python -m src.summarization_experiment --device cuda
python -m src.summarization_experiment --pilot --device xpu
python -m src.summarization_experiment --output results/code/summarization --phase 3 --device cuda
```

The default protocol is scientific and the default output is the current directory. With no `--phase`, the run continues from the last phase through Phase 7, or starts at Phase 0 when `progress.json` is absent. `--phase N` runs only that phase. The project description and the remaining work are in [README.md](/README.md).

Locked settings are in [the experiment contract](/design/summarization-experiments.md). English uses seed 0; Hindi arms use seeds 1, 2, and 3. Validation decides gates, winners and shrink ranks. Test scores are report-only. A failed scientific X gate stops later phases.

For local checks without downloads:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.summarization_experiment --protocol smoke --output /tmp/summarization-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest src.tests.test_summarization src.tests.test_phases -q
```

Smoke uses a tiny random mT5 on CPU. Its metrics and timings are development evidence. Routine tests do not establish real-weight/GPU behavior or scientific results.

Each phase is a small experiment definition:

| Phase | Work |
|---|---|
| 0 | One update, generated-summary artifacts, training-only truncation checks |
| 1 | English checkpoint, Hindi-only X, G5 and base/English/X serving |
| 2 | Random, loss, drift, past-summary replay, repair-and-stop |
| 3 | Two-sample C1 maps and saved prediction, C2 block plans, D, full-English reference |
| 4 | E room plans, strict parameter matching, validation winner |
| 5 | Validation shrink search for A and the winner, serving comparisons |
| 6 | Independent prefixed English checkpoint and G4 |
| 7 | Drift plus past summaries and optional frozen room; unavailable extensions are explicit |

With no `--phase`, execution continues from the last phase through Phase 7. `--phase N` runs one phase using earlier reports in the same output root. Shared code validates settings and data identity before reusing reports or finished arms. A changed protocol needs a separate output directory.

Every phase saves:

- `progress.json` and `events.jsonl`, including failures and interruptions.
- Per-arm `run.json`, recovery state, predictions, and `arm.json`; scientific Hindi arms have separate seed directories.
- `final/<arm>/`, an atomically published, independently loadable adapter snapshot.
- `report.json`, `report.csv`, and `report.md`, including actual steps/epochs, scores, cost records, seed spread, decisions and artifact paths.

The completed JSON report is published last. Missing tables can be rebuilt from it. Removing a phase report reuses completed arms; incomplete arms resume from checkpoints. Recovery retention and model residency follow [model-weights.md](/design/model-weights.md).

Serving uses one request per fixed test article at batch size 1, with no warm-up and one pass. Reports include per-request latency, median/p95 across articles, output length, and device memory. Merged/unmerged adapters and the two-adapter reference are timed separately. No timing repeats are used.

Use direct imports, for example `from src.models.model_mt5 import MT5SmallModel`. Data preprocessing receives tokenizer callbacks from the model; it does not import or select a model. Pipelines receive model/data adapters through `RunContext`. Namespace packages require no `__init__.py`.

Detailed contracts: [scope](/design/README.md), [revision](/design/revision.md), [experiments](/design/summarization-experiments.md), [models](/design/models.md), [datasets](/design/datasets.md).
