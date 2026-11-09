# Models and ModelBase

Only `Seq2SeqLMModel` and `MT5SmallModel` are active. Native Torch owns numerical execution; Transformers supplies model definitions and tokenizers. Constructors are lazy. [Model weights](/design/model-weights.md) owns mandatory numerical resume, bounded retention and residency.

`ModelBase` resolves CUDA/ROCm, then XPU, then CPU, and transfers nested batches without changing dtypes or autograd. Metadata and loaded unique parameter counts must enforce the 700M ceiling for every learner, probe and serving role. Each role owns isolated optimizer, scheduler, scaler, counters and Torch RNG.

`train(k, *, batch_factory, objective, hooks, limits)` consumes the caller's finite update stream. It never chooses replay or shuffles independently. Skipped updates consume inputs and measured work but do not advance the optimizer/scheduler success clock. `UpdateHooks` supplies only detached token weights; freeze-prefix, KD, Fisher and SAR hooks are removed.

`score`, `evaluate` and `predict` preserve training state. `get_weights` returns independent parameters/buffers; `set_weights` installs weights and resets numerical training state. `save/load(..., "resume")` restores complete optimizer/RNG/counter state; frozen, English, Hindi and shrunk artifacts transfer immutable weights with fresh training state. Checkpoint compatibility and payload checks remain mandatory.

Bookkeeping uses plain model/revision/step identifiers and seeded Python/Torch generators, without hashing batches or base assets. The outer checkpoint manager owns checksums and retention. Model code does not import datasets or pipelines.

## Seq2seq execution

Used by the [summarization plan](/design/summarization-plan.md). No branch here names an arm or method.

### Encoder-decoder executor

`model_seq2seq.py` implements `Seq2SeqSettings` and `Seq2SeqLMModel(ModelBase)`, the shared encoder-decoder executor. `model_mt5.py` provides `MT5SmallModel` with `model_id = "google/mt5-small"`. Transformers supplies the model definition and tokenizer, and native Torch owns LoRA, losses, gradients, optimization, and generation. Constructors are lazy.

| Rule | Detail |
|---|---|
| Parameter count | 300,176,768 from the pinned config, with the output head **not** tied to the embeddings. Count from the hub config, since a locally built `MT5Config` may default to tied weights |
| Precision | Load and train in fp32 on every device |
| Compile | CUDA and XPU compile the training forward with `torch.compile` and dynamic shapes. CPU stays eager. A changed adapter layout compiles again. Greedy generation stays eager so each decoder step runs as accelerator kernels |
| Host wait | On XPU, the process sets `EventHostSynchronizeLinuxUserFenceKmdWait` before the device context exists. The Intel driver then waits on a kernel fence. Loss sums stay on the device until the step reads one scalar after the optimizer update |
| Tokenizer | mT5's SentencePiece tokenizer. Phase 0 checks the environment can load it. If it needs `sentencepiece`, that's a dependency the plan explicitly requires |
| Training mode | LoRA only. `mode="full"` is rejected |
| LoRA targets | Every block's `q`, `k`, `v`, `o` in self-attention and decoder cross-attention, and `wi_0`, `wi_1`, `wo` in the feed-forward layers. Named `(stack, block, part, projection)`. `W_used = W + (alpha / r) B A`, with `B` starting at zero |
| Inputs | Encoder input IDs and mask, plus labels with padding masked out. The decoder input is shifted from the labels inside the model. Per-example loss is the mean NLL over the summary's tokens |
| Batches | One microbatch per update. Support `hindi_batch + replay_per_batch`, or `2 x hindi_batch` for the full-English reference |

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
| `score` kind `example_loss` | Per-example mean target-token NLL |
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

A switch to a causal fallback requires a new adapter and an updated contract; no causal executor is retained.
