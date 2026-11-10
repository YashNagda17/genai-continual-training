# Shared records, storage, and measurement

Owns `src/utils/`: `contracts.py` for boundaries, `numerics.py` for Torch math/device selection and metrics, `state.py` for replay/checkpoint storage, and `runtime.py` for lifecycle execution. Utilities import no concrete models, datasets or pipelines.

Only the summarization plan is active. Removed mechanisms include legacy selection/routing/freeze records, teacher/KD fields, continual-metric helpers and unused sampling functions. Paper formulas and historical protocols remain in [Papers/](/Papers/) and their original design references.

## Boundaries

| Record | Meaning |
|---|---|
| `Record`, `Batch` | Stable source identity/split, article/target payload, tokens/storage and aligned transient tensors |
| `ScoreSpec`, `ScoreResult` | Explicit representation/reduction, ordered IDs, model provenance and measured cost |
| `Objective`, `TrainingBatch` | Supervised reduction/token mode, origin labels and optional detached token targets |
| `TokenTargets`, `UpdateHooks` | G3 baseline/weights aligned to supervised tokens; token targets are the sole update hook |
| `AdapterPlan` | Shared trainable blocks, extra rank, learning-rate scales and arm identity |
| `StepReport` | Consumed IDs/tokens, success/skip reason, loss sums/counts, numerical diagnostics and cost |
| `ModelInfo`, `ResourceBudget`, `ReplayBudget` | Total/trainable parameters and CPU/device/disk/payload bounds |
| `RunSpec`, `RunContext`, `RunResult` | Finite ordered work, injected adapters, scientific identity and persisted outcome references |

Plain JSON stores policy, IDs, manifests and events. Torch performs training and numerical calculations. Transient gradients stay on their compute device. `resolve_device` chooses CUDA/ROCm, then XPU, then CPU and prints CPU placement. Nested tensor transfers preserve integer/boolean dtypes and autograd. mT5 uses fp32.

`TokenTargets` may hold detached tensors on CPU or the compute device. Transient `StepReport.diagnostics.token_surprise` stays on the compute device and is excluded from persisted summaries. Loss sums/counts are plain numbers when `train()` returns, after a combined transfer of pending device metrics.

`TokenTargets.all_positive` defaults to false. A producer may set it only when every aligned token weight is strictly positive by construction. The executor then uses shape metadata for active-token/row counts, avoiding a device scalar read; arbitrary targets retain zero-weight skipping. The repair producer supplies this guarantee for its `1 + beta * clamp(...)` normalized weights under the existing finite-input assumption.

Nested CPU-to-accelerator batch transfers request `non_blocking=True` and preserve dtype/autograd. Source batch tensors remain unmodified until device work completes; consumers run on the same device stream. Transfers to CPU remain blocking so host consumers see completed data. Pageable input staging may still block the host; this removes per-tensor explicit stream waits, without promising transfer/compute overlap.

`ReplayStore` owns payload/membership and budget checks; it never chooses records. Pool examples, past summaries and token baselines count toward retained storage. Recovery-only data cannot augment active training access. The Hindi access rule is enforced by pipeline boundaries.

## Recovery and artifacts

[model-weights.md](/design/model-weights.md) owns checkpoint transactions, bounded retention and role residency. Checkpoints publish complete numerical/policy state atomically, verify payload checksums and retain a usable fallback before pruning. Disk budgets include offload and staging; immutable external base caches are separately identified and counted. Run identity includes model/data/settings and execution compatibility; changing settings cannot resume a different run.

`Checkpoints.pin_artifact` publishes immutable English/Hindi/shrunk adapters and phase final exports through a staged directory and completion marker. The checkpoint manager owns integrity hashes; model batches and prepared data are not repeatedly content-hashed.

The writer lock is exclusive/non-blocking (POSIX `flock`, Windows `msvcrt`). File writes flush before atomic rename. Directory fsync is best-effort where unavailable. Windows permission retries are bounded; no copy/delete replacement substitutes for atomic publication. Obsolete-tree cleanup stays within the owning attempt, preserves retained recovery and tolerates already-removed obsolete files.

The runtime records status, counters, metrics, costs and checkpoints. A failed/interrupted attempt keeps its last complete recovery point. Learner and one analysis role run sequentially: offload the learner before acquiring an auxiliary, release that role before switching, then restore the learner. Every role is capped at 700M total parameters.

Phase reports are owned by [phases/artifacts.py](/src/phases/artifacts.py): saved arm records, per-seed summaries, status events, and JSON/CSV/Markdown publication after final snapshots. Missing tables rebuild from completed reports without model execution.

## Measurements

Metrics include unofficial whitespace ROUGE-L, character chrF, script/right-language share, paired article bootstrap, overlap, interaction, kappa and energy rank. Higher-is-better scores retain their declared scale. Validation alone decides gates, winners and shrink ranks. Test predictions support report comparisons only.

Record update time, GPU hours, peak memory, optimizer steps, tokens per language, trainable parameters, analysis categories/forward-pass units and storage. Preserve failed and skipped work. Null means unavailable; CPU measurements do not establish accelerator results.

For scientific comparisons, average each article across all three declared Hindi seeds before the fixed-seed 1,000-resample paired bootstrap. Also retain per-seed deltas and standard deviation. Never silently drop a failed seed. Serving uses one timing pass without warm-up, independently of training seed replication. Paper-level interpretation and manual ratings consume these artifacts later.
