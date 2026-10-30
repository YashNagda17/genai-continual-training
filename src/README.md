# Pipeline API

Python 3.11+ and Torch; pipeline calculations, training and inference use native Torch. Models, tokenizers, datasets, optimization settings, and experiment protocols are caller inputs. Imports do not fetch data or allocate models. Package installation is deferred; choose the matching Torch build from the [official installation guide](https://pytorch.org/get-started/locally/).

Use direct module imports such as `from src.models.model_base import ModelBase`. These directories are namespace packages and need no `__init__.py`.

## Device placement

Concrete model adapters call `super().__init__(device)` with `"auto"` (default), `"cpu"`, `"cuda:N"` or `"xpu:N"`. Automatic routing chooses available CUDA/ROCm, then Intel XPU, then CPU. An unavailable requested accelerator falls back to CPU and `resolve_device` prints that choice to stderr; explicit `"cpu"` also prints, and avoids accelerator probing. NVIDIA uses CUDA, [AMD ROCm also uses Torch's `cuda` API](https://docs.pytorch.org/docs/stable/notes/hip.html), and [Intel uses `torch.xpu`](https://docs.pytorch.org/docs/stable/xpu.html). Availability depends on the installed Torch build and drivers.

The adapter places its wrapped Torch module on `self.device` when active and uses `self.to_device(payload)` for nested tensor batches. This helper preserves dtypes and autograd. Cached weights load in float16. Before the optimizer is created they are cast to `optimizer_dtype` (float32 by default). Policy tensors use float32 on every device. Gradient calculations follow their input/model device; plain inputs to unbound helpers, one-time threshold calibration and small sampling bookkeeping use CPU. Persistent policy state stays JSON-compatible. Weighted sampling uses a local CPU Torch generator seeded from the saved policy RNG, leaving the model's global Torch RNG untouched.

## Implemented algorithms

| Class / configuration | Implementation |
|---|---|
| `OASISPipeline(OASISConfig(...))` | Exact small-batch SIREN, prior-state normalization, shared-threshold admission, raw-arrival EMA, numerical target-rate calibration; current-only and growing historical-pool profiles |
| `TSRPipeline(TSRConfig(...))` | Fixed-initialization LoRA-B probing, task softmax, batch source decisions, source-era teacher routing, supervised-plus-KL objective, uniform native replay |
| `ALSARPipeline(ALSARConfig(...))` | Greedy balanced insertion, signed similarity-aware retrieval, discounted exposure, BFC prefix freezing, periodic full refresh; independent SAR/aL switches |
| `CSReLPipeline(CSReLConfig(...))` | CL and CL-Prv reference fitting, repeated subset fitting and reducible-loss selection, equal task quotas, restricted historical reselection |
| `CombinedPipeline(oasis=..., tsr=..., al_sar=..., csrel=...)` | One learner lifecycle composing the four capabilities; CSReL alone constructs boundary replay |

Method formulas and source ambiguities are documented in [the paper references](/Papers/README.md). These are implementations of the chosen equation profiles, not validated paper reproductions. CSReL-RS and its separate KD extension are outside the selected task-end contract.

## Supplying models and data

Implement [ModelBase](/src/models/model_base.py) and [DatasetBase](/src/datasets/dataset_base.py). Every deferred operation is an abstract method, so incomplete adapters cannot be instantiated. The small adapters in [tests/fakes.py](/src/tests/fakes.py) demonstrate the interface for tests only.

Dataset adapters implement `load`, `split`, `preprocess`, `fetch`, `training_view`, `evaluation_view`, `collate`, `save`, and the `fingerprint`/`has_classes` properties. Prepare sources before creating the run context. The initial lifecycle slices immutable finite task views and calls `collate`; `fetch` is also part of the adapter API for external streaming callers. A record preserves source identity, targets, task, arrival, split, token/storage counts and optional genuine class/group labels. Record payloads must serialize to JSON; `collate` produces transient Torch tensors. Large external assets must be immutable, verifiable, and counted in the declared storage footprint.

Model adapters implement:

- `info`, `describe_role`, `fork`: metadata before allocation and isolated reproducible auxiliary initialization.
- `activate`, `offload`, `release`, `parameter_counts`: bounded residency, exact state preservation and unique parameter counts. Inactive state stays on run-owned disk, not an unbounded CPU cache. Include transient scoring/training buffers in declared peak estimates.
- `train(k, *, batch_factory, objective, hooks, limits)`: finite logical passes over the supplied update stream, returning `StepReport` records. The model alone owns the optimizer, scheduler, scaler and backend RNG.
- `score`, `evaluate`, `predict`: state-preserving scoring/evaluation and inference.
- `get_weights`, `set_weights`: independent parameter/buffer snapshots and dynamic installation into the same architecture. Installing weights resets optimizer, scheduler, scaler and training-step state.
- `save`, `load`: separate complete resume state from immutable frozen snapshots.

Scoring requests use `ScoreSpec`. `loss` returns one scalar per ordered example; `last_layer_gradients` returns one vector per example; `lora_b_gradient` returns one batch-mean vector in the specified zero-B coordinates; `logits` returns detached teacher distributions' logits at the supplied temperature. Return detached Torch tensors in `ScoreResult.values`, with exact model revision, parameter-view identity, reduction and measured cost. Scoring must preserve model buffers, gradients, training mode, optimizer and training RNG. Large production gradient views require an adapter and explicit resource validation.

For aL, invoke `hooks.plan_update({"feature_norm_sq": ...})` before backward and enforce its prefix. Report `feature_norm_sq`, `freeze_prefix`, and full `layer_fisher` on refresh steps. For SAR, report `gradient_subset` and ordered `class_gradients` restricted to that subset's unfrozen coordinates; use `None` for an empty intersection. The native subset is the paper's fixed sampled parameter subset; choosing different coordinates must be recorded as an adaptation. These statistics come from the supervised objective, even when the learner also receives KD. The owner commits successful-update observations and discards them after skipped updates.

`Objective` supplies separate current/replay means for standalone CSReL and specifies TSR's `KL(teacher || student)`, temperature, temperature-squared convention and supervised reduction. Teacher outputs are prepared before the student graph. The production adapter must implement numerical loss masking, gradient isolation, KL and freeze semantics; the synthetic tests do not establish those backend properties.

## Concrete model adapters

The three adapters share [the causal-LM executor](/src/models/model_causal_lm.py):

| Class | Pinned model family |
|---|---|
| `Qwen3Model` from `src.models.model_qwen3` | `Qwen/Qwen3-0.6B` |
| `LFM25Model` from `src.models.model_lfm25` | `LiquidAI/LFM2.5-350M` |
| `SmolLM2Model` from `src.models.model_smollm2` | `HuggingFaceTB/SmolLM2-135M-Instruct` |

Install the declared Transformers 5.x, Safetensors and Hugging Face Hub dependencies later using the intended Torch build. Imports and constructors are lazy. First metadata/preprocessing access resolves the source revision and tokenizer; first activation obtains and verifies weights. Use `revision=` to pin a known commit and `cache_dir=` to choose the immutable asset cache. All execution uses Torch, with eager attention and native Torch LoRA.

```python
from src.models.model_causal_lm import CausalLMSettings
from src.models.model_qwen3 import Qwen3Model
from src.datasets.dataset_instruction import InstructionDataset, InstructionPreprocessing

model = Qwen3Model(CausalLMSettings(
    mode="lora", rank=16, alpha=32, seed=17,
    max_length=512, max_batch_size=2,
), device="auto")
dataset = InstructionDataset()
raw, _ = dataset.load({"path": "data/instructions.jsonl", "dataset_id": "instructions"})
views = dataset.split(raw, {"validation_fraction": 0.1, "seed": 17})
dataset.preprocess(views, InstructionPreprocessing(**model.preprocessing_kwargs()))
# Supply these objects in RunContext with the explicit pipeline and RunSpec.
# The runtime injects dataset collation callbacks and validates preprocessing identity.
```

CLINC, Schuster, Bitext, and TRACE are fetched into that same JSONL schema. The pipeline prepares them with the local model's tokenizer:

```python
from src.datasets.dataset_sources import fetch_source, load_instruction_source

directory = fetch_source("schuster", "src/datasets/data")
dataset, raw = load_instruction_source(directory)
pipeline.bind_instruction_dataset(dataset, raw)
```

`download_datasets.sh` fetches `clinc_oos`, `schuster`, `bitext_customer_support`, and `trace`. Task order is `dataset.manifest["source"]["task_order"]`.

Qwen always renders with `enable_thinking=False`. Prepared targets containing explicit thinking/analysis tags are reported and dropped, and prompt tokens remain unsupervised. Standard templates are used for LFM2.5 and SmolLM2. Teacher/probe roles use the same non-thinking identity. Generation accepts conversations or `dataset.collate_prompts(records)` and returns IDs, completion texts and generated token IDs. `evaluate` returns negative token NLL and perplexity; supply `evaluation_spec["generation"]` for exact-match accuracy. For direct evaluation outside a pipeline, inject `collate=dataset.collate` and `collate_prompts=dataset.collate_prompts` when constructing the model.

The learner defaults to LoRA; `mode="full"` enables full fine-tuning. TSR probes always use deterministic zero-B LoRA, even for a full-fine-tuned learner. LFM2 maps attention and convolution projections explicitly. Layer groups include embeddings, decoder blocks and output parameters; tied weights count once. These architecture/adapter mappings are declared language-model adaptations. Caller-supplied layer costs override parameter-based estimates. Scoring implements actual per-example gradients and batch-mean LoRA-B gradients, not embedding substitutes.

`max_length`, `max_batch_size`, `max_score_bytes` and `working_bytes` bound/estimate working buffers. The latter must be profiled for the chosen architecture and sequence shape; it is not a measured hardware guarantee. Resident weights load in float16 and are upgraded to float32 before AdamW; optimizer state stays float32 and GradScaler stays disabled. Set `optimizer_dtype="float16"` to keep the loaded precision. Per-example loss scores average response tokens. Batch objectives support `per_example_mean` and `per_token_mean`. Full-vocabulary teacher logits are buffered on CPU, checked against the score-byte bound, and used in chunked exact masked KL. OASIS's tied-head gradients include the input embedding path too.

`ResourceBudget(cpu_bytes, accelerator_bytes, disk_bytes=...)` includes current/replay payloads, declared model peaks, checkpoint staging, offload snapshots and referenced immutable base bytes. The compatibility default disk cap is 10 GB; configure it for the actual run. The runtime retains the checkpoint pair and required teachers. Inactive numerical roles stay under the attempt's `.offload/` directory, rather than accumulating in CPU RAM. `release()` removes an unneeded role's private offload files. Standalone calls use `cache_dir/standalone` unless `bind_runtime(output, resources)` is supplied.

LoRA checkpoints share a checksummed immutable base and store adapters/buffers; full fine-tuning stores changed base weights too. `get_weights()` returns independent complete CPU snapshots. Installing an arbitrary snapshot conservatively switches subsequent saves to full-weight overlays, so changes to nominally frozen weights are preserved. Weight installation assigns a content-derived initialization identity and resets optimization state; resume restores it. A run started from transferred/frozen weights must resume with the same declared initial model input. Checkpoints record source/config/layout, library version, tokenizer/template and complete training RNG/optimizer/scheduler/scaler state.

Exact numerical resume also requires the same resolved Torch version and device. Frozen weights can move between compatible architectures on different devices. Replacing a learner in a bound pipeline binds the replacement to that run's dataset and storage budget; inactive private state is copied into the receiving attempt without loading model tensors.

## Calling a pipeline

After constructing concrete adapters and explicit configurations:

```python
from pathlib import Path
from src.pipelines.csrel import CSReLPipeline
from src.utils.contracts import RunContext

# model, dataset, run_spec, and csrel_config are supplied application inputs.
pipeline = CSReLPipeline(csrel_config)
result = pipeline.run(RunContext(
    model=model,
    dataset=dataset,
    spec=run_spec,
    output=Path("results/code/runs/my-run/attempt-1"),
))
```

Construct `RunSpec`, `ReplayBudget`, and `ResourceBudget` from [contracts.py](/src/utils/contracts.py). Required settings include task order, seed, batch size, attempted/successful update and token limits, passes/exhaustion, reduction, replay units, CPU/device bounds, checkpoint cadence, scientific/source/adaptation identity, and evaluation settings. Limits are per task; global counters are also returned. The initial executor uses one microbatch per update and keeps partial batches. A pass is a traversal of admitted current membership; replay-only updates remain bounded by attempts/steps/tokens.

Standalone OASIS `historical_pool` requires `ReplayBudget.examples=None` and enough explicit token/byte capacity for all observed data. `current_only` is a labeled adaptation. Exact SIREN enumerates subsets; supply small selection batches within `max_selection_batch` (at most twelve for the intended profile).

The hybrid requires OASIS `current_only`, SAR `active_population`, and one of CSReL's task-end variants. Full SAR requires genuine classes or a documented grouping; aL-only can be used in the hybrid with SAR disabled. Native aL-SAR's balanced memory writer still requires class labels. TSR probe initialization, gradient view, routing/KD temperatures, and every auxiliary fitting schedule are explicit configuration.

## Composing individual steps

Every paper class accepts `model=None, dataset=None` after its configuration. Pure calculations take explicit inputs; model-dependent steps can use the supplied shared learner. Calling `run(context)` binds the context's model and dataset. These public methods can be reused or overridden in another hybrid:

| Class | Algorithm steps |
|---|---|
| OASIS | `score_gradients`, `informativeness`, `adjusted_score`, `siren`, `relative_informativeness`, `admission_probabilities`, `sample_admission`, `update_statistics`, `expected_rate`, `calibrate_threshold`; composed by `select_current` |
| TSR | `probe_batch`, `accumulate_probe`, `task_signature`, `probe_task`, `routing_weights`, `select_teacher`, `route_replay`, `draw_branch`, `retrieve_replay`, `objective`, `teacher_outputs`, `snapshot_task`, `build_task_memory` |
| aL-SAR | `register`, `reconcile`, `update_class_totals`, `effective_frequencies`, `probabilities`, `retrieve_replay`, `decay_counts`, `increment_counts`, `update_counts`, `information_per_cost`, `freezing_scores`, `plan_update`, `update_feature_statistics`, `update_fisher`, `update_similarities`, `observe_update`, `balanced_insert` |
| CSReL | `fit_model`, `score_losses`, `reference_pool`, `train_reference`, `score_reference`, `train_subset`, `reducible_losses`, `select_top`, `select_round`, `initialize_selection`, `select_coreset`, `build_memory`, `commit`, `retrieve_replay`, `objective` |

For example, OASIS scoring and SAR exposure updates can be used with the same learner without starting either standalone lifecycle:

```python
from src.pipelines.oasis import OASISPipeline
from src.pipelines.al_sar import ALSARPipeline

admission = OASISPipeline(oasis_config, model=model, dataset=dataset)
retrieval = ALSARPipeline(sar_config, model=model, dataset=dataset)
batch = dataset.collate(current_records)
selection = admission.select_current(batch, reduction="per_example_mean")
retrieval.register(current_records)
```

`CombinedPipeline(..., model=model, dataset=dataset)` assigns the learner to all components. Its `set_model(replacement)` also updates every component. Within a running lifecycle, switch models only between completed operations. Probe, teacher, reference and subset models use isolated auxiliary roles so their fitting does not change the learner.

Use the shared object to continue with the exact numerical state, or transfer a snapshot to a different wrapper of the same architecture:

```python
next_pipeline.set_model(previous_pipeline.model)            # shared live state
next_pipeline.set_weights(previous_pipeline.get_weights())  # independent weights
path = previous_pipeline.save_weights("results/code/weights/stage-1")
next_pipeline.load_weights(path)

policy = previous_pipeline.get_policy_state()
previous_pipeline.set_policy_state(policy)
```

`get_weights()` returns independent Torch tensors, typically detached clones of the wrapped module's state dictionary. A shallow [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) alone still shares storage with live parameters. `set_weights()` and `load_weights()` preserve the receiving model object and start fresh optimizer/scheduler/scaler/step state. `get_policy_state()` copies the component policies and RNG states; it contains no model tensors, dataset, replay store or lifecycle cursor. Restore it into the matching component layout; JSON round trips are supported. For an interrupted run, use the full `save`/`load` lifecycle instead.

Paper configuration and algorithm checks are intentionally omitted during logic review. Callers supply compatible inputs and valid settings. Shared record validation stays in [contracts.py](/src/utils/contracts.py). Utilities are grouped into [numerics.py](/src/utils/numerics.py) (math and metrics), [state.py](/src/utils/state.py) (replay, weights and checkpoints), and [runtime.py](/src/utils/runtime.py) (execution, resource checks and recovery).

## Recovery and artifacts

Each `output` is one attempt. `resume="auto"` resumes a compatible attempt, `"required"` rejects a missing checkpoint, and `"never"` requires a fresh directory. Changed scientific settings, adapter/data identity, implementation source, Torch version or resolved device reject resume. Changing checkpoint cadence does not alter scientific identity.

Outputs include `run.json`, `metrics.csv`, `events.json`, checksummed recovery directories, `latest.json`, and required frozen teachers. A single writer lock protects state publication. The latest complete checkpoint plus one fallback survives during execution; completion retains one. Old replay payloads and numerical recovery files are pruned while small audit manifests remain. Required teachers survive as long as retained recovery state references them. Corrupt newest state can fall back with `recovery.json` identifying the discarded progress.

Recovery includes current cursors/admissions, branch/source choices, policy RNG, exposure/gradient statistics, reference-loss tables, active CSReL role/pass/round, learner numerical state and evaluation progress. Failed updates/writes leave the prior checkpoint intact. Unsaved work may repeat after a hard interruption; its unmeasured cost is not presented as zero.

Resource checks use declared adapter/payload bounds, parameter-count verification and actual staged artifact sizes. Hardware peak memory, asynchronous timing, optimizer/teacher isolation and external-asset accounting require concrete adapters. Wall time around delegated execution is a host measurement; adapter cost records must supply synchronized device timing and actual device/memory measurements. No speedup or training-quality claim follows from the synthetic tests.

## Validation

From the project root:

```sh
python -m unittest src.tests.test_pipelines -v
python -m compileall -q src/models src/datasets src/pipelines src/utils src/combined_pipeline.py
```

`SetupModelTests` is skipped by that command. Run it only for setup (downloaded weights, Torch build, or accelerator placement), not for algorithm or pipeline changes:

```sh
GENAI_SETUP_TESTS=1 python -m unittest src.tests.test_pipelines.SetupModelTests -v
```

The compact suite checks paper equations, signed similarities, round-dependent coreset rankings, source/teacher coherence, native versus hybrid exposure, shared learner and weight/policy handoff, two-task standalone/hybrid runs, budgets, retention and representative recovery failures. It uses a two-coordinate Torch model with autograd SGD and optional KL. Small utility checks cover mocked CUDA/ROCm/XPU routing, dtype/autograd preservation and local sampling RNG. The setup check covers one short forward of each cached model on the resolved device. Transformer gradients, actual LoRA-B semantics, production KD, pinned-author-code equivalence, and scientific experiments remain unverified by the routine suite.

The sample entry point is `python -m src.example`: local SmolLM2, one CLINC pass through `CombinedPipeline`, official-test evaluation, and a few generated intents. Each launch deletes `results/sample/clinc-smollm2` and starts over. The full experiment matrix in [main.md](/design/main.md) is still a later caller.
