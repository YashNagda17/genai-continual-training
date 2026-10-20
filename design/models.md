# Models and ModelBase

Owns `src/models/`, `model_base.py`, and concrete transformer adapters. Checkpoint storage is specified in [model-weights.md](/design/model-weights.md). A new model ID using an existing adapter does not require another subclass.

Execute model operations and own optimizer state. Paper pipelines supply data order, loss composition, and freeze/probe policies. The model adapter contains no branches keyed on paper names.

`ModelBase` is a project wrapper around the backend module. Its public `train(k, ...)` means training for `k` logical passes, as requested by `code-structure.md`; it must not override a backend's `train(mode: bool)` mode-switching API. Use the wrapped module for that operation.

Enforce the [parameter ceiling](/design/README.md) from architecture metadata before allocation, then verify unique counts after loading. Record total and trainable counts separately.

## Interface contract

| Operation | Required behavior |
|---|---|
| `train(k, *, batch_factory, objective, hooks, limits)` | Run the supplied finite/reproducibly bounded update stream for `k` passes; return counters and cost records. Never choose replay or reshuffle independently. |
| `evaluate(view, evaluation_spec)` | Execute fixed inference/loss evaluation without training-state changes; return predictions and metric inputs with IDs. |
| `predict(inputs, generation_spec)` | Return generated outputs and IDs using explicit decoding settings; no gold response in the generation input. |
| `score(batch, score_spec)` | Return aligned per-example losses, logits, or declared gradient features. No optimizer/scheduler step or training RNG advancement. |
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
