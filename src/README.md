# Summarization experiments

mBART learns English summarization, then Hindi, with a fixed English replay pool. Research notes and paper references are archived under `src/docs/`.

Use Python 3.11+ with the dependencies in [requirements.txt](/requirements.txt). Torch owns training, gradients and inference. Models detect CUDA/ROCm, then XPU, then CPU; explicit device requests are supported. Every role is capped at 700M total parameters.

The current non-smoke runner selects `MBart25Summarizer`. Its default device is automatic, so a CUDA cloud host no longer inherits an XPU-only request. Use `--device cuda:0` explicitly on the T4 setup. `--gradient-checkpointing` enables mBART's activation-memory tradeoff; it can reduce memory use but adds recomputation, so benchmark it on the remote host. Enabling it changes the live run identity: use a separate attempt from a run started without it.

Validation and test generation run after all configured epochs in each training stage. English transfers the final-epoch checkpoint; it no longer runs per-epoch validation or reloads an earlier validation winner. The new evaluation schedule enters the arm identity, so use a new attempt directory for English runs previously started with best-pass selection. Historical imported checkpoints remain unchanged.

English/Hindi validation and test views stay on CPU. Only the active batch moves to the accelerator; loss batches are released before generation, and completed held-out sweeps release unused device cache while keeping the model resident. Merged serving reconstruction weights/buffers are staged on CPU instead of retaining duplicate full weights and dense updates on GPU.

```sh
./download_datasets.sh xlsum
./download_model_weights.sh mbart
python -m src.summarization_experiment --device cuda
python -m src.summarization_experiment --pilot --device xpu
python -m src.summarization_experiment --output results/code/summarization --device cuda
```

The default protocol is scientific and the default output is the current directory. Every invocation runs Phases 1 through 7 in numerical order. Use a fresh output directory; existing run artifacts are refused. The handlers share [phase.py](/src/phases/phase.py).

For a five-epoch development comparison, use a new output folder:

```sh
python -u -m src.summarization_experiment --pilot --epochs 5 --device cuda:0 --output results/code/summarization-pilot-5epochs
```

This trains English for five passes, then Hindi for five passes. Each stage has
625 updates at batch size 16 on its 2,000 examples. Default real-data runs use four
passes per language. Changing epochs requires a new output folder, the same pass count applies to every phase in that invocation.

Locked settings are in [the experiment contract](/design/summarization-experiments.md). Every arm runs once with fixed shared randomness. Validation decides gates, winners and shrink ranks. Test scores are report-only. A failed scientific X gate stops later phases.

For local checks without downloads:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.summarization_experiment --protocol smoke --output /tmp/summarization-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest src.tests.test_summarization src.tests.test_phases -q
```

Smoke uses a tiny random mBART on CPU by default; `--device cuda:0` runs it on NVIDIA CUDA. Its metrics and timings are development evidence. Routine CPU tests do not establish real-weight/GPU behavior or scientific results.

For an NVIDIA cloud host, use a CUDA-enabled PyTorch image or select a matching wheel from the [official installer](https://pytorch.org/get-started/locally/), then install [requirements.txt](/requirements.txt). The same native-Torch gather, weighted-loss, nonblocking-copy and packed-reporting optimizations apply on CUDA; no inference backend or precision change is needed.

Validate the cloud environment before downloading the full model and dataset:

```sh
nvidia-smi
python -c 'import torch; assert torch.version.cuda and torch.cuda.is_available(), "NVIDIA CUDA is required"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m src.summarization_experiment --protocol smoke --device cuda:0 --output results/code/cuda-smoke
```

Existing device fallback remains available, so the availability assertion above matters on a cloud host. CUDA timing and speedup claims still require measurement on that host.

The [phase overview](/README.md#what-the-project-does) describes each experiment.

Every phase writes JSON/CSV/Markdown reports. Each arm saves one final adapter at its canonical path, plus predictions, metrics, costs, and its `arm.json` record. There are no seed subdirectories or confirmation runs. The root `run.json` records the final outcome; it is never read to skip work.

No periodic model checkpoints, optimizer recovery files, progress cursors, or duplicate `final/<arm>/` copies are written. Failed runs keep their available reports and completed adapters, but a new invocation starts fresh. Model offload is temporary and applies only while the current process switches roles. See [model storage](/design/model-weights.md).

Serving uses one request per fixed test article at batch size 1, with no warm-up and one pass. Reports include per-request latency, median/p95 across articles, output length, and device memory. Merged/unmerged adapters and the two-adapter reference are timed separately. No timing repeats are used.

Use direct imports, for example `from src.models.model_mbart import MBart25Summarizer`. Data preprocessing receives tokenizer callbacks from the model; it does not import or select a model. Pipelines receive model/data adapters through `RunContext`. Namespace packages require no `__init__.py`.

Detailed contracts: [scope](/design/README.md), [revision](/design/revision.md), [experiments](/design/summarization-experiments.md), [models](/design/models.md), [datasets](/design/datasets.md).

Separate CSReL/TSR mBART comparisons (one epoch per trained model):

```sh
python -u -m src.baseline_experiment --source results/code/summarization --output results/code/mbart-baselines-1epoch --epochs 1 --device cuda:0 --gradient-checkpointing
```

This command runs independently of Phases 1-7. It reads the source English
checkpoint, prepared data, and pool, and reports chrF for CSReL, its matched
random-memory control, TSR, and ordinary random replay. JSON/CSV/Markdown and
final adapters are written under the separate output directory.
CSReL/TSR are summarization adaptations; no exact paper reproduction is claimed.
Use a completed mBART run with prepared data, its English adapter, and its pool. Historical local runs have been removed; imported reports do not contain these training inputs. Pass `--source` explicitly. Old-model artifacts are incompatible.

Phase 7 compares cached teacher KD against matched replay during the same invocation. Its paired report is `phase7/kd/report.md`, with a summary in `phase7/report.md`. Frozen extra room remains in Phase 4.
