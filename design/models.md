# Models and ModelBase

Owns `src/models/`, `model_base.py`, and concrete transformer adapters. Checkpoint storage is specified in [model-weights.md](/design/model-weights.md). A new model ID using an existing adapter does not require another subclass.

Execute model operations and own optimizer state. Paper pipelines supply data order, loss composition, and freeze/probe policies. The model adapter contains no branches keyed on paper names.

Executor bookkeeping stays lightweight: use plain model/revision/step identifiers
and seeded Python/Torch generators, without hashing token batches or base assets.
Fresh initialization streams may differ from the earlier SHA-derived seeds;
resume still restores saved weights, optimizer state, and RNG. The outer checkpoint
manager owns checksums and retention. Both executors measure synchronized wall
time; accelerator peak allocation is recorded where supported. Preserve all
numerical operations, public classes, and dataset/model device boundaries.

`ModelBase` is a project wrapper around the backend module. Its public `train(k, ...)` means training for `k` logical passes, as requested by `code-structure.md`; it must not override a backend's `train(mode: bool)` mode-switching API. Use the wrapped module for that operation.

## Concrete causal language models

`model_causal_lm.py` implements the shared Torch executor; `model_qwen3.py`, `model_lfm25.py`, and `model_smollm2.py` provide the three requested architecture profiles. Transformers supplies model definitions and tokenizers; native Torch owns LoRA, losses, gradients, optimization and generation. Constructors are lazy. Metadata uses a meta-device model to count unique parameters and plan residency before allocating weights. A model is named by its id and the revision string it was given.

`preprocessing_kwargs()` supplies arguments for `InstructionPreprocessing`; it does not import a dataset class. `attach_instruction(dataset)` binds a dataset that was already preprocessed with those arguments and checks the local tokenizer, template, EOS, pad, and non-thinking identity. It does not import a source fetcher. CLINC, Schuster, Bitext, and TRACE all use this path. Inject `collate`/`collate_prompts` callbacks for evaluation. Qwen templates always set `enable_thinking=False`; reasoning-tagged targets are reported and dropped by preprocessing. The tokenizer and template settings are stored with the role. Generation accepts prompt-only batches or conversations.

LoRA is the default training mode; full fine-tuning is configurable. The fixed probe always uses zero-B LoRA with a deterministic A initialization. LFM2 maps its attention and convolution projections explicitly. These language-model/adapter layer mappings are adaptations, not native paper reproduction settings. Ordered layer groups include embeddings, decoder blocks, and final output parameters, counting tied weights once. A legal freeze prefix leaves trainable suffix parameters. Fisher is the mean squared per-example supervised parameter-gradient norm per group; final-feature statistics use the supervised gradient of the final token hidden-state tensor. SAR samples a stable, seeded coordinate subset of eligible parameters and intersects it with the unfrozen groups.

Per-example loss scores are response-token means. Batch objectives support `per_example_mean` and `per_token_mean`, with separate current/replay normalization when requested. Teacher logits cover causally shifted token positions, are temperature-scaled once, and stay on the compute device; exact masked KL is evaluated in chunks on that device. Head gradients include all uses of tied output/embedding parameters. An OASIS batch uses one decoder forward and one decoder backward; each example's vector is the output-projection outer product plus the tied embedding-row scatter when the head and embeddings share storage. Last-layer, subset, and LoRA-B gradient buffers, per-example loss scores, and teacher logits stay on the compute device. Fisher layer sums stay on device for the whole refresh and move to Python once. Parameter tensors move to CPU only when a checkpoint or offload file is saved or loaded. Scoring preserves existing gradients, module modes, buffers and training RNG.

Configure maximum batch size, sequence length, score bytes and working-memory estimates explicitly. Cached weights load in float16 on every device. Before the optimizer is created, `upgrade_parameters` casts the resident module to `optimizer_dtype` (float32 by default) so AdamW and forwards see float32 and GradScaler stays disabled. Policy and score reductions stay float32. Layer compute costs are parameter-based estimates unless caller-supplied; reports label estimates and include the extra statistics forward. Offloading serializes complete state to run-owned disk and releases parameters and optimizer tensors. Shared-base LoRA snapshots contain exact adapter parameters and persistent buffers; full fine-tuning snapshots contain all changed weights. Resume restores optimizer, scheduler, scaler, counters and role-local CPU/device RNG. Weight transfer resets those training states. Callers resume with the same declared initial model input. Scoring records the step count, the objective, and the freeze decision. The existing lifecycle retains/prunes checkpoints and teachers; storage budgets include offload directories and staging.

Binding a lifecycle or replacing its learner supplies the active dataset callbacks and run storage boundary. Rebinding an inactive model copies its private offload files into the receiving attempt before releasing the previous files. Exact numerical resume includes the resolved Torch version and device in execution identity; frozen weights use architecture/tokenizer compatibility and can be transferred independently of that execution identity.

## Injected adapter API

Numerical execution uses Torch. `ModelBase(device="auto")` selects an available CUDA/ROCm device, otherwise Intel XPU, otherwise CPU; explicit `cpu`, `cuda:N` and `xpu:N` requests are supported and unavailable accelerators fall back to CPU. Adapters use the resolved `torch.device` for model tensors, batches and auxiliary roles. Training, gradients and full-distribution teacher scoring remain Torch operations.

When `CausalLMSettings.compile` is set, activation wraps the decoder `forward` with `torch.compile(..., dynamic=True)`. Learner and probe roles compile their own modules. Offload drops that wrapper. Training, scoring, evaluation and prediction use native Torch. Concrete adapters implement the existing model methods and call `super().__init__(device)` to resolve their placement. `to_device(payload)` moves nested tensor batches without changing dtypes or autograd connections. No inference-server client or model export/deployment layer is part of the current implementation.

Adapters additionally implement `get_weights()` and `set_weights(weights)`. The getter returns an independent, backend-native copy of parameters and persistent model buffers. The setter installs that copy into the existing architecture and resets optimizer/scheduler/scaler/step state. It does not reconstruct the wrapper, so every pipeline sharing that object immediately sees the update. Full numerical resume continues to use `save` / `load` with `state_kind="resume"`; weight-only file loading is equivalent to `set_weights` for training-state reset.

The initial Python implementation accepts unloaded `ModelBase` wrappers. `describe_role` returns allocation estimates before `fork` creates an isolated role. `activate`, `offload`, and `release` own backend residency; `parameter_counts` verifies the loaded unique counts. `fork(role, initialization)` must reproduce the named initialization, including training RNG and the fixed zero-B LoRA probe. It must never copy the evolving learner as an auxiliary initialization. Role descriptions include the model id, the role name, and the scoring loss.

`score` returns ordered IDs, numerical values and provenance for an explicit `ScoreSpec`. `collate` belongs to the dataset adapter. `train` consumes `TrainingBatch` objects and returns one `StepReport` per attempted update. The pipeline initially supplies one microbatch per update; larger accumulation is rejected. Before backward, the executor invokes `hooks.plan_update` with supervised feature-gradient statistics and applies its returned legal prefix. Its report supplies supervised layer Fisher traces and subset gradients, separately from the KD objective. The lifecycle owner commits observations after inspecting update success.

Teacher logits are computed before constructing the learner graph, then the teacher is released. `Objective` specifies the KL direction, temperature, temperature-squared convention, loss reduction, and separately normalized current/replay weights. All backend methods are explicit base-class obligations, with no synthetic production implementations.

`ScoreResult` declares `cpu_bytes` and `accelerator_bytes` for detached buffers separately. Teacher outputs stay on the learner device with the retained current and replay payload.

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
- `revision_context` records the objective, including `token_mode`, so a weighted update is visible in the step report.
- A truncated row's surprises must equal the full row's at kept positions. Check this per adapter, including LFM2's convolutions, because cuts rely on it.

## Summarization plan additions

Used by the [summarization plan](/design/summarization-plan.md). No branch here names an arm or method.

### Encoder-decoder executor

`model_seq2seq.py` implements `Seq2SeqSettings` and `Seq2SeqLMModel(ModelBase)`, the shared encoder-decoder executor. `model_mt5.py` provides `MT5SmallModel` with `model_id = "google/mt5-small"`. As with the causal adapters, Transformers supplies the model definition and tokenizer, and native Torch owns LoRA, losses, gradients, optimization, and generation. Constructors are lazy.

| Rule | Detail |
|---|---|
| Parameter count | 300,176,768 from the pinned config, with the output head **not** tied to the embeddings. Count from the hub config, since a locally built `MT5Config` may default to tied weights |
| Precision | Load in bf16 where the device supports it, otherwise fp32. **Never fp16**: T5-family models overflow in fp16. This is an explicit exception to the float16 load rule above |
| Tokenizer | mT5's SentencePiece tokenizer. Phase 0 checks the environment can load it. If it needs `sentencepiece`, that's a dependency the plan explicitly requires |
| Training mode | LoRA only. `mode="full"` is rejected |
| LoRA targets | Every block's `q`, `k`, `v`, `o` in self-attention and decoder cross-attention, and `wi_0`, `wi_1`, `wo` in the feed-forward layers. Named `(stack, block, part, projection)`. `W_used = W + (alpha / r) B A`, with `B` starting at zero |
| Inputs | Encoder input IDs and mask, plus labels with padding masked out. The decoder input is shifted from the labels inside the model. Per-example loss is the mean NLL over the summary's tokens |
| Batches | One microbatch per update, as in the causal executor. The batch bound is `hindi_batch + replay_per_batch` |

### Adapter names

mT5-small has 8 encoder and 8 decoder blocks. Adapter **parts** are named `(stack, block, part)`:

| Part | Projections | Parameters per rank |
|---|---|---:|
| `self_attention` | `q`, `k`, `v`, `o` | 3,584 |
| `cross_attention` (decoder only) | `q`, `k`, `v`, `o` | 3,584 |
| `feed_forward` | `wi_0`, `wi_1`, `wo` | 4,608 |

That is 8,192 per rank per encoder block, 11,776 per decoder block, and 159,744 for all 16 blocks. At rank 16 on all blocks that is 2,555,904 parameters.

A **cell** is one part over a **band** of two blocks (1–2, 3–4, 5–6, 7–8). The 5 part types (encoder self-attention, encoder feed-forward, decoder self-attention, cross-attention, decoder feed-forward) times 4 bands give 20 cells. Which cells and blocks an arm trains is decided in [summarization-experiments.md](/design/summarization-experiments.md).

### Adapter plans

`AdapterPlan` is a plain-JSON record in `contracts.py`. Block keys are written as `"encoder.7"` strings, since JSON can't keep tuple or integer keys. The experiment runner builds the plan; this executor only installs it. Applying a plan never changes a prediction, because new rank starts with `B = 0`.

| Field | Meaning |
|---|---|
| `trainable` | Set of `(stack, block)` whose **shared** adapter may change. All other shared adapters are frozen |
| `new_rank` | Map `(stack, block)` -> extra zero-start rank on every projection of that block, trained at the full rate |
| `lr_scale` | Map `(stack, block)` -> learning-rate multiplier for that block's shared adapter, in `[floor, 1]` |
| `identity` | Arm name, seed, and the resulting trainable and new-parameter counts |

`apply_plan(plan: AdapterPlan)` runs before the first update of a stage, and creates a fresh optimizer.

- **Freezing:** shared adapters outside `plan.trainable` get `requires_grad = False`, so autograd computes nothing below the lowest trainable parameter. In a late-block plan the encoder backward stops after the lowest trainable encoder block. Excluding frozen adapters from the optimizer alone isn't enough, because the backward pass would still run. Keeping them `requires_grad = True` with a zeroed optimizer group is rejected: it would hide the cost difference the placement runs predict.
- **New rank:** `plan.new_rank` adds a second LoRA pair per projection of the listed blocks, with `B = 0`, a deterministic `A`, and `alpha = 2 x rank`, so its scale matches the shared adapter. The forward output is unchanged on the first step.
- **Learning-rate multipliers:** `plan.lr_scale` sets one optimizer parameter group per block for the shared adapter. New rank always uses the base rate.
- The plan's trainable blocks, new rank, learning-rate scales, and parameter counts enter `revision_context`.

### Scoring, probes, and generation

`reset_probe()` folds a loaded immutable checkpoint's effective adapters into its
frozen projections, then installs deterministic zero-B probe factors. It is valid
only on an isolated probe role. Such a role saves full weights for recovery because
its frozen base now includes the checkpoint's update. `UpdateHooks.token_targets`
may compute detached G3 weights from the current forward's flat NLL; it performs no
forward, backward, or optimizer step itself.
When that hook supplies weights, the executor honors `per_example_mean` by first
averaging weighted token losses within each row, then averaging rows. Precomputed
`TokenTargets` retain their existing flat-token reduction for compatibility.

| Operation | Behavior |
|---|---|
| `score` kind `example_loss` | Per-example mean target-token NLL, as in the causal executor |
| `score` kind `token_loss` | The existing flat per-token NLL, reused for G3's weights and baselines |
| `score` kind `probe_gradient` | On an isolated **probe** role (a saved checkpoint plus a zero-B LoRA with deterministic `A`), the mean gradient of target loss over the batch with respect to the probe's `B` factors, grouped by `"cell"` or `"block"`. Returned detached on the compute device, with the group layout. No optimizer step. The learner is untouched |
| `predict` | Greedy decoding up to `max_new_tokens`. `token_ids` end at the first EOS, inclusive. Optional `generation_spec.script_guard`, below |
| `train` | Per-example mean over the batch, or `token_mode="weighted"` with `TokenTargets` (reused from the generation replay additions), including the no-step rule on zero active tokens |

**Forking from a saved checkpoint.** A probe role may start from the saved English checkpoint, because that's an immutable artifact. The existing rule that `fork` never copies the evolving learner still holds.

**Script guard.** Once per tokenizer, mark each vocabulary token that contains any Devanagari letter and each one that contains any Latin letter, from its decoded piece. `script_guard = {"block": "devanagari"}` sets those logits to `-inf` at every step. `{"penalize": "latin", "penalty": 2.0}` subtracts the penalty instead. EOS and special tokens are never masked. The guard changes decoding only, and is part of the evaluation spec's identity.

### Shrinking and serving

The effective update of each adapted projection includes the LoRA scale. With extra rank, it is the sum of the shared and new parts:

```text
ΔW = (alpha / r) B A   (+ (alpha_new / r_new) B_new A_new)
ΔW = U S Vᵀ            (per projection, singular values s_1 >= s_2 >= ...)
keep the top k:        B_k = U_k S_k^½,   A_k = S_k^½ V_kᵀ,   written with scale 1
```

The shrunk adapter stores `B_k` and `A_k` with scale 1, so its effective update is exactly the truncated `ΔW`. At `k` equal to the full rank, outputs must match the original within numerical tolerance. Which `k` to keep is the generation search in [summarization-experiments.md](/design/summarization-experiments.md). `energy_rank` in `numerics.py` only picks the energy-rule candidate.

| Operation | Behavior |
|---|---|
| `effective_updates()` | Per projection, the `ΔW` above, in float32 |
| `shrink_adapters(k)` | Replace each projection's adapter with the top-`k` factors above. Returns new weights through `set_weights` |
| `merged_copy()` | An inference-only role whose base weights are `W + ΔW`, with no adapter modules. The learner is unchanged. Merged, extra rank costs nothing per summary. Unmerged, decoder rank runs at every writing step. Used only for serving timing and merged evaluation |
| Timing | Generation timing synchronizes the device before and after each request, so asynchronous work is counted |

### Not used by this plan

The seq2seq executor doesn't implement: Fisher traces, SAR coordinate subsets, OASIS last-layer gradients, freeze-prefix hooks, teacher logits and KD, `prompt_embedding`, chat templates, or `reserve_response`. A recipe needing them fails preflight on this executor.

**Fallback:** if the plan switches to Qwen3-0.6B, the same additions (adapter plans, probe gradients, script guard, shrinking, merged copy) are added to `CausalLMModel`, with layer bands in place of encoder and decoder blocks. Update this section first.
