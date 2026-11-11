# Weight artifacts and checkpoint identity

Owns `src/models/model_weights/`. Keep immutable base-model assets here, namespaced by model ID and revision. Run-specific learner checkpoints, auxiliary models, and teachers belong under `results/code/runs/<run_id>/<attempt_id>/checkpoints/`; a configured external weight cache must preserve the same manifest contract.

## Artifact kinds

| Kind | Required contents |
|---|---|
| Base weights | Model id, revision name, model configuration, tokenizer and template settings |
| Inference export | Base reference plus trained weights/adapters and all inference preprocessing settings |
| Frozen teacher/probe reference | Weights or adapter snapshot, model id, task, and role |
| Resume checkpoint | Learner and active auxiliary weights; optimizer/scheduler/scaler for each training role; complete lifecycle state below |

Adapter-only snapshots are valid when the base is immutable and reconstructible. Inference exports cannot be accepted as exact-resume checkpoints.

Manifest paths are relative to the attempt root or resolve through content IDs/external source URIs. Machine-local cache roots are runtime mappings, not artifact identities.

## Mandatory resume contract

A checkpoint binds numerical state to the resolved run specification, code/design version, task/phase, cursor/sampler/RNG, replay manifest/payload, policy state, teacher references, counters, and evaluation history. Include pending admissions, consumed-versus-prefetched positions, and committed event ID. The model serializes numerical state; the lifecycle owner coordinates the manifest.

- Require periodic saves in steps or elapsed time, checked at safe boundaries, plus completed task/phase transitions. Include long CSReL reference/subset training.
- Save synchronized model/optimizer/scheduler/scaler/counter/policy state after a completed update attempt with no accumulated gradients, including overflow skips. Non-training phases also have safe boundaries after committed admission/probe/selection/evaluation events. On graceful interruption, reach such a boundary or preserve the prior checkpoint; hard kills may repeat unsaved work.
- Under one run-writer lock: stage artifacts, verify files/checksums, publish a completion manifest, then atomically replace the latest pointer. Failed writes leave the prior complete checkpoint usable.
- Task-boundary commits include the new replay membership, teacher references, and policy state together. During CSReL memory construction, save the active phase/round and its auxiliary state; keep the old replay version available until the replacement commits. Resume that phase rather than retraining the learner's completed task.
- Exact resume requires the same scientific configuration, data source, preprocessing settings, and code schema. Changing a model, seed, split, method, stream order, or scientific budget (such as replay capacity or training-token limit) starts a new run. Log operational changes separately: output relocation, checkpoint cadence, or additional staging-disk capacity may change without invalidating resume when content identity, numerical behavior, and algorithmic access remain unchanged. Such changes cannot relax the 700M ceiling or discard required state.
- Restore all component state before fetching data or drawing randomness. No partial loading, optimizer reinitialization, or counter reset; weight transfer starts a separate run.
- Metrics and selection logs use stable event IDs and a checkpoint watermark. Ignore or replace uncommitted trailing logical events; resumed execution must not double-count training/evaluation results. Record recovery recomputation cost separately and identify unmeasured work lost in a hard kill.
- Evaluation can resume from committed example/metric events or deterministically rerun its incomplete phase. It must not repeat completed training or require permanently retaining all historical learner checkpoints; evaluate each required checkpoint before it is eligible for pruning.

## Rolling disk retention

- Retain **latest complete recovery checkpoint plus one previous fallback**. On completion, keep final resumable state and remove an unreferenced fallback. Inference exports reuse weight blobs. Extra best/milestone exports require an explicit policy and byte budget.
- One in-progress save can temporarily coexist with the retained pair. Budget for this peak, the old/new replay versions during commit, and all required dependencies before writing. Stream serialization instead of collecting full copies of every model in RAM. If a save cannot fit, preserve the valid checkpoint and report the failure; do not destroy it to make space.
- Prune only after the replacement checkpoint is verified and published. Delete only run-owned artifacts unreferenced by **all** retained checkpoint manifests, current task records, or explicitly pinned exports. An interrupted cleanup must remain restartable. Keep small manifests, metrics, and failure records even after obsolete weights are removed.
- Recovery snapshots and algorithm-required TSR teachers are different artifact classes. The rolling pair does not authorize deleting a still-needed teacher. Store a shared immutable base once and exact adapter states per required era when the chosen training method permits it; teacher snapshots contain no optimizer state.
- Exact reconstruction is required. If the backbone changes, adapter-only storage is insufficient: budget the necessary changed weights or reject that configuration. Lossy teacher compression, dropping eras, or replacing them with the latest teacher requires an explicit algorithm adaptation.
- Charge all required teachers, auxiliary states, caches, recovery dependencies, and staging bytes to declared storage budgets. Native TSR's task records can still grow; if they do not fit, report an incompatible run instead of hiding them outside the budget. Prune an evicted hybrid teacher only after no retained recovery state or task record references it.

## Bounded RAM/VRAM residency

Keep the active learner and at most one required auxiliary/teacher model resident on the accelerator by default, subject to the byte budget; offload or serialize the learner at a safe boundary if both do not fit. Load teachers on demand and evict the previous teacher from the bounded cache. Do not preload all teacher/auxiliary weights into CPU RAM either; keep inactive artifacts on disk.

Sharing immutable base tensors across distinct adapters is allowed only when state and autograd isolation are verified. Never swap an adapter inside a live learner graph. For a tight-memory KD path, compute detached teacher outputs before the student graph, release the teacher when necessary, and bound output buffering; exact loss-preserving chunking is allowed. Charge transfers/reloads to measured cost. Probe and CSReL stages release their models when no longer needed, retaining only state required for resume or subsequent computation.

## Generation-aware replay additions

Used by the [generation replay recipe](/design/generation-replay.md).

- `PastSelfConfig.retain_copies=False` is an explicit, labeled adaptation that drops eras. A saved copy may be pruned once its answers and baselines are committed and no retained checkpoint or task record references it. Removing it takes it out of `dependencies()`; deletion still goes through reference-aware pruning, so a retained fallback keeps its files until replaced. Its removal cannot be undone within a run, so an arm that needs a copy later must retain it.
- Answers and baselines are committed with replay membership, in one recovery checkpoint. Task-end writing and refresh progress are lifecycle state with safe boundaries after each batch. A hard kill may repeat at most the unsaved batch.
- Refresh and writing run on the resident learner and add no resident model. Selection follows the existing auxiliary rule.
- Retained answers and baselines are replay payload, not recovery-only state, so they count toward the replay budget and remain readable by the active policy.

## Summarization plan additions

Used by the [summarization plan](/design/summarization-plan.md).

| Artifact | Contents | Rule |
|---|---|---|
| English checkpoint | The mT5 base reference plus the English adapter, its pass number, its validation scores, and the prefix setting | **Kept for every Hindi run of that prefix.** Stored under the prefix name. Never pruned while any attempt references it |
| Pool artifact | The 500 pool records with past summaries, baselines, and `loss_before` | Written once at the end of the English stage, next to that prefix's English checkpoint. Never regenerated by a Hindi-trained model |
| Hindi checkpoint | Final adapter after the fixed Hindi passes, plus its `AdapterPlan` | One per arm and seed. Kept until its evaluation, shrink, and serving jobs finish |
| Shrunk adapter | F's adapter at `k_G`, written with scale 1, plus `k_E`, `k_G`, and the test table | An inference export. Not an exact-resume checkpoint |
| Merged copy | Base weights with `ΔW` folded in | Built in memory for serving timing only. Not saved unless a byte budget allows it |

- **No era teachers.** The English checkpoint is read by probes and as the two-adapter reference, but no teacher is loaded per batch.
- Rolling latest-plus-one retention applies inside each Hindi run as usual. Adapter-only snapshots are enough, because the base stays frozen.
- **Residency:** the learner plus at most one more role (the probe, a merged copy, or the English checkpoint for the reference) is resident at a time.

Performance controls: non-smoke phase runs checkpoint every 100 recorded events; smoke retains a cadence of one. Policy loops use that cadence, while lifecycle phase transitions still force a complete checkpoint. Unsaved batches can be replayed from the last coherent numerical/policy snapshot. Cadence remains operational and is excluded from `RunSpec.scientific()`.

Phase 0 retains its activated model through training, generation batches and export. Analysis callers can use `release_auxiliary(restore_learner=False)` to leave the idle learner's resume snapshot offloaded between auxiliary roles. A final passing shrink candidate stays resident through export. This does not increase the number of resident models or retain weights across independent learner runs.

The unmerged serving reference reuses its current checkpoint until the requested checkpoint changes. Older swap timing files without the `checkpoint_reuse` marker are remeasured; ordinary resident serving files remain reusable. Switch costs are still included in the request that changes checkpoints.
