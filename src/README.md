# Summarization experiments

The active code implements [Idea_Refined_2.md](/Idea_Refined_2.md): mT5-small learns English summarization, then Hindi, with a fixed English replay pool. Only this model family, XL-Sum, its three pipelines, and their behavior tests remain. Paper sources, reviews, literature tools, and writing workflows are preserved.

Use Python 3.11+ with the dependencies in [requirements.txt](/requirements.txt). Torch owns training, gradients and inference. Models detect CUDA/ROCm, then XPU, then CPU; explicit device requests are supported. Every role is capped at 700M total parameters.

The current non-smoke runner selects `MBart25Summarizer`. Its default device is automatic, so a CUDA cloud host no longer inherits an XPU-only request. Use `--device cuda:0` explicitly on the T4 setup. `--gradient-checkpointing` enables mBART's activation-memory tradeoff; it can reduce memory use but adds recomputation, so benchmark it on the remote host. Enabling it changes the live run identity: use a separate attempt from a run started without it.

Non-smoke runs publish recovery checkpoints every 100 recorded events, with forced lifecycle phase-boundary saves. Smoke keeps a cadence of one. A restart can replay batches since the last complete checkpoint. CUDA hosts skip the local 180-second between-arm cooldown. Phase 0 and serving retain resident models, and analysis jobs leave the idle learner offloaded while switching auxiliary roles. Legacy serving swap timings without the checkpoint-reuse marker are remeasured.

Validation and test generation run after all configured epochs in each training stage. English transfers the final-epoch checkpoint; it no longer runs per-epoch validation or reloads an earlier validation winner. The new evaluation schedule enters the arm identity, so use a new attempt directory for English runs previously started with best-pass selection. Historical imported checkpoints remain unchanged.

```sh
./download_datasets.sh xlsum
./download_model_weights.sh mt5
python -m src.summarization_experiment --device cuda
python -m src.summarization_experiment --pilot --device xpu
python -m src.summarization_experiment --output results/code/summarization --phase 3 --device cuda
```

The default protocol is scientific and the default output is the current directory. With no `--phase`, the run continues from the last phase through Phase 7, or starts at Phase 0 when `progress.json` is absent. `--phase N` runs only that phase. The project description and the remaining work are in [README.md](/README.md).

For a five-epoch development comparison, use a new output folder:

```sh
python -u -m src.summarization_experiment --pilot --phase 1 --epochs 5 --device cuda:0 --output results/code/summarization-pilot-5epochs
```

This trains English for five passes, then Hindi for five passes. Each stage has
625 updates at batch size 16 on its 2,000 examples. Default real-data runs use four
passes per language. Changing epochs requires a new output folder, and later
phases using this pilot must also receive `--epochs 5` to match its saved settings.

Locked settings are in [the experiment contract](/design/summarization-experiments.md). English uses seed 0; Hindi arms use seeds 1, 2, and 3. Validation decides gates, winners and shrink ranks. Test scores are report-only. A failed scientific X gate stops later phases.

For the imported `summarization-passes5` checkpoints, prepare a report and portable
handoff without model execution:

```sh
python -u -m src.phase2_from_checkpoint --exploratory
```

The source ran **four** epochs. Its English validation chrF (20.7666) is below
lead-1 (22.0174), so the scientific gate remains `fix_english`. The user explicitly
authorized an exploratory Phase 2. The preparation writes
[Phase 1 training results](/results/code/summarization-passes5/phase1/training-report.md)
and matching JSON/CSV, verifies artifacts/prediction IDs and chrF, rebases Linux
paths in memory, and checks that past-summary replay has eligible examples.
Original weights, source progress and source arm records remain unchanged.
Serving benchmarks remain incomplete and no completed Phase 1 report is invented.

To start only the exploratory Phase 2, run:

```sh
python -u -m src.phase2_from_checkpoint --exploratory --run --device cuda:0
```

This reuses prepared subsets and the English checkpoint. All five arms run four
Hindi epochs on seed 1; there is no English retraining or Phase 1 serving.
Results go to `results/code/phase2-exploratory/phase2/report.md` (also JSON/CSV).
Repeat the exact command to resume interrupted work. Different source hashes or
settings require a new `--output` root. These results are exploratory, not a passed
scientific experiment.

For local checks without downloads:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.summarization_experiment --protocol smoke --output /tmp/summarization-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest src.tests.test_summarization src.tests.test_phases -q
```

Smoke uses a tiny random mT5 on CPU by default; `--device cuda:0` runs it on NVIDIA CUDA. Its metrics and timings are development evidence. Routine CPU tests do not establish real-weight/GPU behavior or scientific results.

For bounded performance checks, `python -u -m src.performance_check` runs three
English batches with a fresh disposable adapter on CUDA. It defaults to the
prepared data in `results/code/summarization-passes5/prepared` and uses local
model assets, without phases, validation or checkpoint
publication. It prints synchronized update/wall times and peak memory, then
saves one JSON under `results/performance/`. It never saves trained weights.
`--batches` is limited to 10; the warm mean excludes the first batch when possible.

`python -u -m src.performance_check --mode generation --examples 2 --checkpoint PATH_TO_ENGLISH_CHECKPOINT --prepared PATH_TO_MATCHING_PREPARED_DATA`
compares fixed-limit generation with all-row EOS stopping on two validation
articles. It checks identical output token IDs and records decoder steps and
timings. The checkpoint and prepared-data prefixes must match. This is an
inference-only check; the CPU tests do not establish its GPU speedup.
Generation requires actual adapter weights, not metadata alone. The checkpoint
weights for the imported English checkpoint are under
`results/code/summarization-passes5/phase1/final/english`; use the matching prepared
data under that run's `prepared` directory.

For an NVIDIA cloud host, use a CUDA-enabled PyTorch image or select a matching wheel from the [official installer](https://pytorch.org/get-started/locally/), then install [requirements.txt](/requirements.txt). The same native-Torch gather, weighted-loss, nonblocking-copy and packed-reporting optimizations apply on CUDA; no inference backend or precision change is needed.

Validate the cloud environment before downloading the full model and dataset:

```sh
nvidia-smi
python -c 'import torch; assert torch.version.cuda and torch.cuda.is_available(), "NVIDIA CUDA is required"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m unittest src.tests.test_cuda -v
python -m src.summarization_experiment --protocol smoke --device cuda:0 --phase 0 --output results/code/cuda-smoke
```

The CUDA tests exercise weighted training, token scoring, generation and dropout/checkpoint resume with tiny local weights. They explicitly skip without NVIDIA CUDA; a skip is not GPU validation. Existing device fallback remains available, so the availability assertion above matters on a cloud host. CUDA timing and speedup claims still require measurement on that host.

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
