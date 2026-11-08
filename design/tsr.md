# TSRPipeline

> **Current plan:** the [summarization plan](/design/summarization-plan.md) reuses only the fixed zero-B LoRA probe idea, for location maps ([summarization-experiments.md](/design/summarization-experiments.md)). Routing, era teachers, and KD aren't used unless a third language is added. This contract is unchanged.

Owns `src/pipelines/tsr.py` and `TSRPipeline`. Its responsibilities are **task signatures, replay-source routing, and routed distillation**. Uniform within-source retrieval belongs to native TSR; the hybrid replaces that retrieval with aL-SAR explicitly.

Source and derivation: [TSR reference](/Papers/TSR-summary.md), §4.3.2 Eq. 4, §§5.1–5.3, Algorithm 2, Appendix C.1.

## Signature contract

`g_t = mean_b grad_B L_CE(theta_0; b)` over the declared current-task training probe batches.

- `theta_0` is the same fixed initialization for every task. The evaluated signature uses the **zero-initialized LoRA B factors**, flattened in a fixed cross-layer parameter order. Both the base and LoRA initialization are identical across probes.
- Use labels/responses from the current training partition only. Record probe IDs, sample/batch weighting, loss masking/reduction, parameter layout, precision, and RNG. Probing uses forward/backward computation but no optimizer step.
- An evolving learner checkpoint, different randomly initialized adapters per task, arbitrary last-layer gradients, Fisher magnitudes, or embedding vectors are not this signature. Changes require an adaptation label.
- Keep the probe model isolated. Its gradients must not enter learner accumulation or alter learner RNG/state. Different proxy models do not share a compatible signature space automatically.

## Routing and training contract

Each native past-task record is `(g_j, M_j, theta_j)`: signature, retained data subsample, and immutable end-of-task teacher. Routing returns the coherent record, not independently chosen data and teacher.

1. Compute `w_j = softmax_j(cos(g_j, g_t) / tau)`, `tau > 0`, over eligible previous tasks.
2. At each training update choose a current batch with probability `1-alpha`, otherwise choose one source task with probability `w_j` and sample a batch uniformly from its retained data. Require `0 <= alpha <= 1`; it is a **batch-level replay probability**, not a buffer fraction or per-example loss weight.
3. For replay from task `j`, use teacher `theta_j`. For a current batch, use the teacher of `argmax_j w_j` (stable task-ID tie break).
4. Train with supervised loss plus `lambda * KL(p_teacher || p_student)` on the same batch. Specify temperature, token mask, loss reduction, and any temperature scaling explicitly; distinguish routing temperature from KD temperature.
5. At task end, append the task's signature, configured data subsample, and frozen teacher. Native records remain immutable across later tasks.

For the first task or empty eligible history, take current-only updates with no teacher loss. This is the explicit boundary behavior for Algorithm 2's otherwise empty routing set. Reject a zero/nonfinite signature for cosine routing unless a separately named fallback policy is configured. Do not silently turn it into uniform replay.

## Storage and integration boundary

- Era teachers must reconstruct the exact task-boundary model; never substitute the latest learner. [model-weights.md](/design/model-weights.md) owns teacher dependencies, adapter sharing, byte-budget enforcement, and on-demand residency.
- Cached teacher logits are not a general replacement for queryable teachers: native TSR also evaluates an older teacher on current-task batches. A cache-only scheme needs a changed distillation specification.
- A fixed number of records/examples per task grows with task count. Native TSR does not by itself provide the project's fixed global storage budget. The hybrid's CSReL allocation/reselection and task eligibility rules are explicit modifications.
- Routing assumes task identities/boundaries are supplied. Inferring tasks from an unlabeled stream is outside this implementation. Signatures from unavailable future tasks are prohibited.
- SAR's conditional example retrieval must not change TSR task weights or detach examples from their era teachers. OASIS filtering changes the trained current distribution; the hybrid nevertheless probes the permitted pre-filter current training view and records that choice.

Checkpoint current/historical signatures and provenance, probe cursor/partial sum or its restart boundary, eligible membership revision, routing weights, pending branch/source/teacher, and RNG. Recompute weights only when their signature or membership inputs change.

