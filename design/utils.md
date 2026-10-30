# Shared records, storage, and measurement

Owns `src/utils/`: typed records, replay storage, serialization, RNG, config validation, measurement, and metrics. The callable implementation uses four logical files: `contracts.py` for records, `numerics.py` for shared math and metrics, `state.py` for replay/checkpoint storage, and `runtime.py` for generic lifecycle execution. Paper formulas stay in paper classes; numerical model execution remains in the model adapter. The runtime is a mixin using the owner's hooks and does not import concrete pipelines, models or datasets.

`numerics.py` also owns Torch device selection and detached policy-tensor conversion. `resolve_device` prints to stderr when it resolves to CPU, including an explicit `cpu` request and a fallback from an unavailable CUDA or XPU device, so a silent accelerator miss is visible. Numerical policy work follows the supplied tensor/model device; unbound list inputs use CPU. Policy arithmetic stays float32. Resident weights load in float16 and are cast to the optimizer dtype before AdamW. Persistent policies, IDs and event records remain plain JSON values; transient gradients, including OASIS last-layer score buffers, stay on their compute device and are not staged through CPU before device-side use. Weighted sampling uses a local CPU Torch generator seeded from the saved component RNG, without modifying global Torch RNG state. Nested tensor transfers preserve integer token/mask types and autograd state. Execution and inference remain native Torch.

`ResourceBudget.disk_bytes` bounds run-owned checkpoint, teacher, offload and staging files, defaulting to 10 GB for compatibility with two-argument callers. Immutable external base caches are separately identified and counted in model metadata. Model boundary helpers validate compatibility, tensor buffer bounds and referenced base-asset hashes.

## Boundary records

Use dataclasses or equivalent validated records.

| Record | Required fields / semantics |
|---|---|
| `RunSpec` | Schema version, run kind, method/variant, source provenance, seed streams, ordered tasks, model/data/preprocessing references, access policy, budgets, loss/training/evaluation settings, adaptations |
| `TaskContext` | Task/phase, restricted current training view, read-only eligible replay view, learner/probe references, role-specific RNG, counters, artifact sink; no test/future-data handle |
| `EvaluationContext` | Checkpoint, held-out view, decoding/metric specification, isolated RNG, output sink; no selection callbacks or training mutation |
| `SelectionResult` | Ordered candidate IDs, selected IDs, aligned scores/probabilities when defined, score provenance, eligibility mask, selection state update, cost; selection is a subset without replacement unless specified |
| `ReplayPlan` | Eligible source IDs, normalized probabilities, batch replay probability, requested size, replacement policy, teacher per source/current batch; retrieval may repeat IDs only when declared |
| `StepReport` | Attempt index, consumed IDs/origins, valid token counts, loss sums/denominators, update success/skip reason, optimizer step, gradient/freeze diagnostics, measured cost |
| `RunResult` | Status, checkpoint/manifest references, metric rows, cost summary, completed task/phase, failure details; not a model object serialized into JSON |

Validate probabilities, IDs, dimensions, and fingerprint compatibility on these boundaries. Reject NaN/Inf. Return typed empty results for permitted no-op cases. Keep score types distinct: a CSReL score is not an OASIS probability or a TSR routing weight.

## ReplayStore

Own payload, membership, task associations, and budget validation; never choose records. Expose read-only task views, materialize an explicitly permitted selection, and atomically commit a proposed replacement. Each retained record carries canonical identity, payload, token/storage counts, and metadata needed by its active policies. Every referenced teacher or signature has a compatible task/representation identity.

Budgets specify their unit and scope: global or per-task examples, serialized bytes, tokens, and/or total auxiliary storage. Enforce every declared cap; report all units. No implicit conversion between examples and tokens. A per-task cap grows with task count and is not a global bound.

Eviction removes records from the active retrieval/cache view and updates membership. Recovery-only blobs may remain while referenced by retained checkpoints; they count toward disk/auxiliary budgets and are accessible only after restoring the complete checkpoint, never to augment active replay. Algorithm-defined class summaries may outlive records within their declared budget. Capacity zero produces empty replay; report any loss of task coverage.

## Evaluation definitions

Save a task-by-checkpoint matrix. For a specified metric transformed to higher-is-better with a recorded scale, let `A[t,j]` be performance on task `j` after training task `t`.

- Current-task learning: `A[t,t]`; retention: all `A[t,j]` for `j < t`.
- Backward transfer: mean over `j < t` of `A[t,j] - A[j,j]`.
- Forgetting: mean over `j < t` of `max_{k=j..t-1} A[k,j] - A[t,j]`. Preserve negative values if performance improves; any clamped version must have a different metric name.
- Before-task transfer: evaluate task `j` immediately before learning it and compare with the same base model's score. Evaluation is read-only and cannot feed policy decisions.
- TSR-style after-learning transfer: compare `A[j,j]` against a matched model trained only on task `j` from the declared common initialization. Label this separately from before-task transfer.

Return undefined aggregate quantities as null with a reason, including first-task forgetting. Do not average heterogeneous task scores until their scale and weighting are fixed. Keep per-task values even when aggregation is justified. Report semantic/task performance and output-format validity separately for structured generation; do not repair an invalid output silently before evaluating it.

## Cost and artifact contract

Each run writes a resolved JSON manifest and status, a long-form metrics CSV, phase-level cost records, checkpoint references, and selection/replay diagnostics under `results/code/runs/<run_id>/<attempt_id>/`. Add per-example evaluation errors where applicable; publication plots are generated as PDF by code under `src/`, with script/config references in the output manifest.

Artifact utilities implement [checkpoint transactions](/design/model-weights.md), locking, event watermarks, and reference-aware pruning. Reference immutable shared blobs rather than copying them on each save.

Record data preparation, probing, selection, reference-model training, learner training, teacher inference, memory construction, and evaluation separately. Include wall-clock seconds, peak CPU/GPU memory with units and measurement scope, parameter counts, optimizer steps, tokens, replay exposures, storage breakdown, and FLOPs when measured or estimated. Label estimates and unavailable measurements. `gpu_hours = sum(device_active_seconds) / 3600`; synchronized timing must include asynchronous device work.

Never report selection-only forward cost as total CSReL cost, or TSR's absence of optimizer steps as zero probe compute. Separate cold-start overhead from incremental update cost without hiding either. Keep failed and negative runs. Outcome (`positive`, `negative`, `null`, or unassessed) is distinct from execution status and requires a predefined comparison criterion.

Aggregate independent seeds, not batches, for uncertainty. Save seed counts, mean, sample standard deviation, CI method, and missing-run reasons; do not produce a meaningful confidence interval from a single run or silently drop failed seeds.
