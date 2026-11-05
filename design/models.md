# Models and ModelBase

Owns `src/models/`, `model_base.py`, and concrete transformer adapters. Checkpoint storage is specified in [model-weights.md](/design/model-weights.md). A new model ID using an existing adapter does not require another subclass.

Execute model operations and own optimizer state. Paper pipelines supply data order, loss composition, and freeze/probe policies. The model adapter contains no branches keyed on paper names.

`ModelBase` is a project wrapper around the backend module. Its public `train(k, ...)` means training for `k` logical passes, as requested by `code-structure.md`; it must not override a backend's `train(mode: bool)` mode-switching API. Use the wrapped module for that operation.

## Concrete causal language models

`model_causal_lm.py` implements the shared Torch executor; `model_qwen3.py`, `model_lfm25.py`, and `model_smollm2.py` provide the three requested architecture profiles. Transformers supplies model definitions and tokenizers; native Torch owns LoRA, losses, gradients, optimization and generation. Constructors are lazy. Metadata uses a meta-device model to count unique parameters and plan residency before allocating weights. Pin the resolved source revision and hash immutable base assets before accepting a checkpoint reference.

`preprocessing_kwargs()` supplies arguments for `InstructionPreprocessing`; it does not import a dataset class. `attach_instruction(dataset)` binds a dataset that was already preprocessed with those arguments and checks the local tokenizer, template, EOS, pad, and non-thinking identity. It does not import a source fetcher. CLINC, Schuster, Bitext, and TRACE all use this path. Inject `collate`/`collate_prompts` callbacks for evaluation. Qwen templates always set `enable_thinking=False`; reasoning-tagged targets are reported and dropped by preprocessing. This is recorded in tokenizer/template identity for every role. Generation accepts prompt-only batches or conversations.

LoRA is the default training mode; full fine-tuning is configurable. The fixed probe always uses zero-B LoRA with a deterministic A initialization. LFM2 maps its attention and convolution projections explicitly. These language-model/adapter layer mappings are adaptations, not native paper reproduction settings. Ordered layer groups include embeddings, decoder blocks, and final output parameters, counting tied weights once. A legal freeze prefix leaves trainable suffix parameters. Fisher is the mean squared per-example supervised parameter-gradient norm per group; final-feature statistics use the supervised gradient of the final token hidden-state tensor. SAR samples a stable, seeded coordinate subset of eligible parameters and intersects it with the unfrozen groups.

Per-example loss scores are response-token means. Batch objectives support `per_example_mean` and `per_token_mean`, with separate current/replay normalization when requested. Teacher logits cover causally shifted token positions, are temperature-scaled once and buffered on CPU; exact masked KL is evaluated in chunks. Head gradients include all uses of tied output/embedding parameters. An OASIS batch uses one decoder forward and one decoder backward; each example's vector is the output-projection outer product plus the tied embedding-row scatter when the head and embeddings share storage. Last-layer, subset, and LoRA-B gradient buffers, and per-example loss scores, stay on the compute device. Fisher layer sums stay on device for the whole refresh and move to Python once. Teacher logits stay buffered on CPU. Parameter tensors move to CPU only for checkpoint and offload serialization. Scoring preserves existing gradients, module modes, buffers and training RNG.

Configure maximum batch size, sequence length, score bytes and working-memory estimates explicitly. Cached weights load in float16 on every device. Before the optimizer is created, `upgrade_parameters` casts the resident module to `optimizer_dtype` (float32 by default) so AdamW and forwards see float32 and GradScaler stays disabled. Policy and score reductions stay float32. Layer compute costs are parameter-based estimates unless caller-supplied; reports label estimates and include the extra statistics forward. Offloading serializes complete state to run-owned disk and releases parameters and optimizer tensors. Shared-base LoRA snapshots contain exact adapter parameters and persistent buffers; full fine-tuning snapshots contain all changed weights. Resume restores optimizer, scheduler, scaler, counters and role-local CPU/device RNG. Weight transfer resets those training states and assigns a content-derived initialization identity. Frozen inputs also carry a stable initialization reference; callers resume with the same declared initial model input. Scoring revisions additionally track the successful update history, data/objective/freeze decisions and role RNG so equal step numbers alone cannot identify different weights. The existing lifecycle retains/prunes checkpoints and teachers; storage budgets include offload directories and staging.

Binding a lifecycle or replacing its learner supplies the active dataset callbacks and run storage boundary. Rebinding an inactive model copies its private offload files into the receiving attempt before releasing the previous files. Exact numerical resume includes the resolved Torch version and device in execution identity; frozen weights use architecture/tokenizer compatibility and can be transferred independently of that execution identity.

## Injected adapter API

Numerical execution uses Torch. `ModelBase(device="auto")` selects an available CUDA/ROCm device, otherwise Intel XPU, otherwise CPU; explicit `cpu`, `cuda:N` and `xpu:N` requests are supported and unavailable accelerators fall back to CPU. Adapters use the resolved `torch.device` for model tensors, batches and auxiliary roles. Training, gradients and full-distribution teacher scoring remain Torch operations.

When `CausalLMSettings.compile` is set, activation wraps the decoder `forward` with `torch.compile(..., dynamic=True)`. Learner and probe roles compile their own modules. Offload drops that wrapper. Training, scoring, evaluation and prediction use native Torch. Concrete adapters implement the existing model methods and call `super().__init__(device)` to resolve their placement. `to_device(payload)` moves nested tensor batches without changing dtypes or autograd connections. No inference-server client or model export/deployment layer is part of the current implementation.

Adapters additionally implement `get_weights()` and `set_weights(weights)`. The getter returns an independent, backend-native copy of parameters and persistent model buffers. The setter installs that copy into the existing architecture and resets optimizer/scheduler/scaler/step state. It does not reconstruct the wrapper, so every pipeline sharing that object immediately sees the update. Full numerical resume continues to use `save` / `load` with `state_kind="resume"`; weight-only file loading is equivalent to `set_weights` for training-state reset.

The initial Python implementation accepts unloaded `ModelBase` wrappers. `describe_role` returns allocation estimates before `fork` creates an isolated role. `activate`, `offload`, and `release` own backend residency; `parameter_counts` verifies the loaded unique counts. `fork(role, initialization)` must reproduce the named initialization, including training RNG and the fixed zero-B LoRA probe. It must never copy the evolving learner as an auxiliary initialization. Role descriptions include coordinate, vocabulary, and scoring-loss fingerprints.

`score` returns ordered IDs, numerical values and provenance for an explicit `ScoreSpec`. `collate` belongs to the dataset adapter. `train` consumes `TrainingBatch` objects and returns one `StepReport` per attempted update. The pipeline initially supplies one microbatch per update; larger accumulation is rejected. Before backward, the executor invokes `hooks.plan_update` with supervised feature-gradient statistics and applies its returned legal prefix. Its report supplies supervised layer Fisher traces and subset gradients, separately from the KD objective. The lifecycle owner commits observations after inspecting update success.

Teacher logits are computed before constructing the learner graph, then the teacher is released. `Objective` specifies the KL direction, temperature, temperature-squared convention, loss reduction, and separately normalized current/replay weights. All backend methods are explicit base-class obligations, with no synthetic production implementations.

`ScoreResult` declares `cpu_bytes` and `accelerator_bytes` for detached buffers separately. Teacher-output residency is checked together with the learner and retained current/replay payload; CPU-only distillation does not require an accelerator budget.

Enforce the [parameter ceiling](/design/README.md) from architecture metadata before allocation, then verify unique counts after loading. Record total and trainable counts separately.

## Interface contract

When evaluation requests `return_predictions`, the causal adapter also returns a transient `_predictions` list with sample IDs, prompts, gold strings, generated strings, and exact-match flags. The runtime removes this list before metric logging and writes it to a task-stage prediction artifact. This reuses evaluation generation without an extra forward pass.

| Operation | Required behavior |
|---|---|
| `train(k, *, batch_factory, objective, hooks, limits)` | Run the supplied finite/reproducibly bounded update stream for `k` passes; return counters and cost records. Never choose replay or reshuffle independently. |
| `evaluate(view, evaluation_spec, progress=None)` | Execute fixed inference/loss evaluation without training-state changes; return predictions and metric inputs with IDs. When `progress` is set, call it as `progress(done, total)` after each scored batch. Omitting it leaves evaluation unchanged. |
| `predict(inputs, generation_spec)` | Return generated outputs and IDs using explicit decoding settings; no gold response in the generation input. |
| `score(batch, score_spec)` | Return aligned per-example losses, logits, or declared gradient features. No optimizer/scheduler step or training RNG advancement. |
| `get_weights()` / `set_weights(weights)` | Copy parameters/buffers or install them in the same wrapper with fresh optimizer state. |
| `save(path, state_kind)` / `load(path, state_kind)` | Distinguish inference weights, frozen snapshot, and resumable training state; validate all compatibility metadata. |

`batch_factory` is supplied by the lifecycle owner and controls one pass over eligible training inputs. `limits` states optimizer-step/token limits and stopping behavior. Online single-pass runs use one pass; a step-limited sampler must have a finite stopping condition. Report actual passes, consumed examples/tokens, microbatches, and optimizer steps separately.

## Required capabilities

Advertise supported operations at construction: per-example target loss, last-layer parameter gradients, final-feature activation gradients, named adapter-parameter gradients, frozen teacher logits, ordered layer groups, and freeze-aware training. A recipe requiring an unsupported operation fails preflight; it cannot silently fall back to embeddings, full-model gradients, or random selection.

Gradient features specify parameter names/order, loss reduction, precision, model checkpoint, train/eval mode and stochastic-layer behavior, and whether gradients are taken on weights or activations. OASIS last-layer gradients, aL-SAR last-feature gradients, and TSR LoRA-B gradients are distinct requests.

The training executor exposes ordered hooks for batch preparation, pre-backward policy, and post-update observations. These hooks may use declared intermediate features to choose trainable layers and read gradients, but never call `optimizer.step()` themselves. Avoid retaining per-example graphs after the step. Resolve incompatible hooks before training starts.

## Numerical and state guardrails

- Declare the supervised loss denominator: per-token, per-example mean, or another paper-specified reduction. Aggregate loss sums and counts correctly across variable-length batches and accumulation; averaging batch means is not generally equivalent.
- A scoring/probe pass preserves learner parameters, buffers, mode, gradients, optimizer/scheduler state, and training RNG. Use an isolated RNG or save/restore it. Frozen teachers run without parameter gradients and cannot share mutable adapter tensors with the learner.
- Distillation compares teacher/student distributions on identical token positions and vocabulary. Specify KL direction, temperature, any temperature-squared scaling, and token reduction. Reject incompatible teacher tokenizers/vocabularies rather than aligning by token index alone.
- Frozen parameters receive no updates from gradients, weight decay, or momentum. Clear stale gradients when masks change. Preserve optimizer state for later unfreezing according to the declared policy.
- Freezing weights does not necessarily eliminate backward computation through those layers. Only cut a prefix graph when no upstream trainable parameter needs its gradient, including adapters and embeddings. Count the work actually performed.
- Apply a consistent freeze mask over an accumulated optimizer step. A different microbatch-level scheme requires a documented algorithm variant. Distinguish skipped updates due to nonfinite gradients from successful updates in all counters.
- Keep attempted updates separate from successful optimizer steps. Overflow skips retain the scaler update, consumed-data/RNG progress, and cost, but do not advance the scheduler or commit success-clock policy observations.
- Do not promise bitwise reproducibility across hardware/backends. Record deterministic settings and validate resume within declared numerical tolerances.
- Role loading/unloading must preserve learner parameters and optimizer state; follow [model-weights.md](/design/model-weights.md).

## Generation-aware replay additions

Used by the [generation replay recipe](/design/generation-replay.md). No branch here names a method. New capabilities: `generation`, `token_loss`, `prompt_embedding`.

| Operation | Behavior |
|---|---|
| `score` kind `token_loss` | Detached per-token NLL, flat in row-major order over supervised tokens, with reduction `"none"`. The caller splits by answer length |
| `score` kind `prompt_embedding` | One L2-normalized vector per example: mean final hidden state over attended tokens of a prompt-only batch. `_batch_identity` must accept a payload with no `labels` |
| `predict` | `token_ids` end at the first EOS, inclusive, with padding removed. `texts` is unchanged. No other caller reads `token_ids` today. Prompt-plus-prefix batches use the existing prompt-only path |
| `evaluate` | `evaluation_spec["text_metrics"]` adds named metrics over generated strings, using `numerics.py` |

Split `_losses` so a new token-level helper returns flat NLL, owners, and counts, and `_losses` sums those. Existing behavior and chunking stay the same.

`train` applies `Objective.token_mode` to a batch with `token_targets`, using flat surprise `s`, baseline `b`, weight `w`, and allowance `m`:

| `token_mode` | Loss |
|---|---|
| `all` | Existing supervised loss. `token_targets` is `None` |
| `weighted` | `sum(w * s) / count(w > 0)` |
| `hinge` | `sum(w * max(0, s - b - m)) / count(w > 0 and s - b - m > 0)` |

- **A batch with zero denominator takes no optimizer step.** Return `success=False` and `skip_reason="no_active_tokens"`, with no backward. `optimizer_update` must not be called, because AdamW's weight decay and momentum move parameters even when gradients are zero, and a stop must really stop.
- Report detached pre-update surprise in `diagnostics["token_surprise"]` as a flat list, on success and on this skip. It is not persisted.
- Include a fingerprint of `token_targets` in `revision_context`, so scoring revisions distinguish different selections.
- A truncated row's surprises must equal the full row's at kept positions. Check this per adapter, including LFM2's convolutions, because cuts rely on it.
