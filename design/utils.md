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

Exclude Explorer metadata (`desktop.ini` and `Thumbs.db`, case-insensitive) from checkpoint hashes, synchronization, and payload accounting. Continue verifying every actual model/state payload. Cleanup of already obsolete trees tolerates entries concurrently removed by another actor; missing retained recovery payloads remain errors.

Atomic file and checkpoint-directory replacement retries Windows permission failures five times with bounded exponential delays, then reports persistent failures. Never replace atomic rename with a copy/delete fallback. Comparison output roots may be absolute paths outside the repository to avoid sync-folder interference; summaries retain absolute paths for such roots.

Obsolete checkpoint-tree cleanup handles Windows read-only attributes and retries transient permission errors up to three times. Targets must resolve strictly inside the owning attempt; current retained checkpoints are excluded by pruning. Persistent failures remain explicit and preserve the published recovery pointer.

Each run writes a resolved JSON manifest and status, a long-form metrics CSV, phase-level cost records, checkpoint references, and selection/replay diagnostics under `results/code/runs/<run_id>/<attempt_id>/`. Add per-example evaluation errors where applicable; publication plots are generated as PDF by code under `src/`, with script/config references in the output manifest.

Artifact utilities implement [checkpoint transactions](/design/model-weights.md), locking, event watermarks, and reference-aware pruning. The run-writer lock is exclusive and non-blocking: POSIX uses `fcntl.flock`, and Windows uses `msvcrt.locking` on one byte. Directory `fsync` is best-effort because some platforms reject directory handles. File `fsync` retries a read/write handle where a read-only handle is rejected. Teacher paths are compared with forward slashes so Windows pruning keeps the same blobs as POSIX. Peak process RSS is null when the OS has no `resource` module. Reference immutable shared blobs rather than copying them on each save.

Record data preparation, probing, selection, reference-model training, learner training, teacher inference, memory construction, and evaluation separately. Include wall-clock seconds, peak CPU/GPU memory with units and measurement scope, parameter counts, optimizer steps, tokens, replay exposures, storage breakdown, and FLOPs when measured or estimated. Label estimates and unavailable measurements. `gpu_hours = sum(device_active_seconds) / 3600`; synchronized timing must include asynchronous device work.

Never report selection-only forward cost as total CSReL cost, or TSR's absence of optimizer steps as zero probe compute. Separate cold-start overhead from incremental update cost without hiding either. Keep failed and negative runs. Outcome (`positive`, `negative`, `null`, or unassessed) is distinct from execution status and requires a predefined comparison criterion.

Aggregate independent seeds, not batches, for uncertainty. Save seed counts, mean, sample standard deviation, CI method, and missing-run reasons; do not produce a meaningful confidence interval from a single run or silently drop failed seeds.


## Learner-order ablation

The optional `--shuffle-training` comparison flag permutes the full permitted learner view separately per pass, using a deterministic seed derived from run seed, task stage, and pass. Cumulative views mix all seen tasks. Dataset arrival order, memory candidate membership, replay RNG, and auxiliary fitting order remain unchanged. Pending queue/cursor state remains checkpointed; permutations reconstruct without consuming the replay RNG. The flag is part of scientific identity. This is an explicitly labeled learner-order ablation, not an exact paper reproduction.

## Generation-aware replay additions

Used by the [generation replay recipe](/design/generation-replay.md). All are additive; existing callers see no change.

| Item | Change |
|---|---|
| `Record` | New `practice: dict | None = None`, the metadata a retained record's active policy needs: `answer_ids`, `answer_text`, `baseline`, `writer`, `writer_mode`, `generation`, `truncated`, `gold_similarity`. It holds plain JSON and is never set by dataset code. `target` keeps the gold response |
| `practice_record(record, cut=None)` | In `contracts.py`. Returns a copy whose `target` is `{"text": answer_text, "ids": answer_ids[:cut]}`, with `tokens` updated. Same `sample_id`. Returns a row with `practice=None` unchanged |
| `TokenTargets` | In `contracts.py`. `baseline` and `weight`, float32 CPU tensors, flat in row-major order over supervised tokens. Length must equal `Batch.supervised_tokens` |
| `TrainingBatch` | New `token_targets: TokenTargets | None = None` |
| `Objective` | New `token_mode: str = "all"` (`all`, `weighted`, or `hinge`) and `token_allowance: float = 0.0`. See [models.md](/design/models.md) |
| `ReplayStore.usage` | New `answer_tokens`. Bytes already follow the serialized size, so answers and baselines count automatically. `tokens` remains the gold payload and has no implicit conversion to answer tokens |
| `numerics.py` | New `token_similarity`, `answer_disagreement`, `kmeans`, `largest_remainder`, `rouge_l`, all in native Torch on the supplied device |

Runtime additions in `runtime.py`:

- `collate_prompts(records)` and `collate_prefixes(records, lengths)` apply the same train-only guard and order check as `collate`.
- `run` passes `self.token_targets(batch)` into `TrainingBatch`. The default hook in `PipelineBase` returns `None`.
- `report_summary` also drops `token_surprise`, as it does `class_gradients`.
- `_limit_reached` also stops at `spec.identity["max_task_seconds"]` when set, using `progress["seconds"]`. Using `identity` follows `shuffle_training` and leaves `RunSpec`'s fields unchanged. The value is part of scientific identity.

New separately recorded cost categories: selection sampling, selection embedding, answer writing, answer check, surprise forward, repair rollout, noise replica. Each records generations, input tokens, new tokens, and seconds. A generation count alone hides the cost of re-reading a long prompt.

## Summarization plan additions

Used by the [summarization plan](/design/summarization-plan.md). All are additive, so existing callers see no change.

**Records** in `contracts.py`:

`UpdateHooks.token_targets(batch, detached_nll)` optionally constructs token
targets from the executor's current forward. It is used by G3; the default is
`None`, preserving older callers. `SummarizationPipeline` in `pipeline_base.py`
reuses the generic lifecycle, adds split-checked score/generation helpers, and
records analysis costs. Snapshot-backed auxiliary roles load their layout before
activation. New job identities bind configuration and input artifacts once at
setup, without hashing numerical batches.

| Item | Change |
|---|---|
| `AdapterPlan` | `trainable`, `new_rank`, `lr_scale`, `identity`, as defined in [models.md](/design/models.md). Plain JSON, with block keys written as `"encoder.7"` strings, since JSON can't keep tuple or integer keys |
| `GenerationSpec.script_guard` | Optional `{"block": script}` or `{"penalize": script, "penalty": float}`. `None` keeps current behavior |
| `Record.practice` | Reused for past summaries, adding `past_baseline` and `loss_before` keys. `writer = "english_checkpoint"` |

**Metrics** in `numerics.py`, native and labelled unofficial:

| Function | Definition |
|---|---|
| `chrf(hyp, ref, n=6, beta=2)` | Character n-gram F-score from 1 to `n`, with whitespace removed, averaged over orders, on a 0–100 scale. 0 for an empty hypothesis |
| `script_share(text, script="devanagari")` | Letters in the script divided by all letters. 0 when there are no letters |
| `paired_bootstrap(a, b, resamples, seed)` | `Δ = mean(a - b)` over paired items, with the 2.5% and 97.5% quantiles of resampled means. Uses its own CPU Torch generator |
| `interaction(s_a, s_rev, s_blk, s_d)` | `s_d - s_a - (s_rev - s_a) - (s_blk - s_a)` |
| `top_overlap(rank_1, rank_2, k)` | Size of the intersection of the two top-`k` sets, divided by `k` |
| `cohen_kappa(r1, r2)` | Agreement between two raters' categorical labels, corrected for chance |
| `energy_rank(singular_values, fraction)` | Smallest `k` whose kept squared singular values, summed over all projections, reach `fraction` of the total |

`rouge_l` is reused as it is, on whitespace tokens. It handles Hindi, because Hindi separates words with spaces.

**Cost categories:** `pool_scoring`, `pool_generation`, `stop_regeneration`, `probe_pass`, `shrink_generation`, `guard_generation`, and `serving_timing`. Each records rows, input tokens, new tokens, seconds, and **forward-pass units**, where a forward over one row is 1 unit and a training row (forward plus backward) is 3. Training cost is recorded as before, plus peak GPU memory per stage.

**Evaluation:** the task matrix above still applies, with tasks `english` and `hindi`. The current plan also reports forgetting against the English checkpoint, `forget = S_en(English checkpoint) - S_en(arm)`, which is the same quantity for two tasks, along with seed aggregation as stated above.
