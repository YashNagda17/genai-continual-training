# Weight artifacts and checkpoint identity

Owns `src/models/model_weights/`. Keep immutable base-model assets here, namespaced by model ID and revision. Run-specific learner checkpoints, auxiliary models, and teachers belong under `results/code/runs/<run_id>/<attempt_id>/checkpoints/`; a configured external weight cache must preserve the same manifest contract.

## Artifact kinds

| Kind | Required contents |
|---|---|
| Base weights | Exact source revision, file checksums, model configuration, tokenizer/template references |
| Inference export | Base reference plus trained weights/adapters and all inference preprocessing settings |
| Frozen teacher/probe reference | Immutable weights or adapter snapshot, base reference, task/role, parameter-layout fingerprint |
| Resume checkpoint | Learner and active auxiliary weights; optimizer/scheduler/scaler for each training role; complete lifecycle state below |

Adapter-only snapshots are valid when the base is immutable and reconstructible. Hash the base and adapter configuration together. Inference exports cannot be accepted as exact-resume checkpoints.

Manifest paths are relative to the attempt root or resolve through content IDs/external source URIs. Machine-local cache roots are runtime mappings, not artifact identities.

## Mandatory resume contract

A checkpoint binds numerical state to the resolved run specification, code/design version, task/phase, cursor/sampler/RNG, replay manifest/payload, policy state, teacher references, counters, and evaluation history. Include pending admissions, consumed-versus-prefetched positions, and committed event ID. The model serializes numerical state; the lifecycle owner coordinates the manifest.

- Require periodic saves in steps or elapsed time, checked at safe boundaries, plus completed task/phase transitions. Include long CSReL reference/subset training.
- Save synchronized model/optimizer/scheduler/scaler/counter/policy state after a completed update attempt with no accumulated gradients, including overflow skips. Non-training phases also have safe boundaries after committed admission/probe/selection/evaluation events. On graceful interruption, reach such a boundary or preserve the prior checkpoint; hard kills may repeat unsaved work.
- Under one run-writer lock: stage artifacts, verify files/checksums, publish a completion manifest, then atomically replace the latest pointer. Failed writes leave the prior complete checkpoint usable.
- Task-boundary commits include the new replay membership, teacher references, and policy state together. During CSReL memory construction, save the active phase/round and its auxiliary state; keep the old replay version available until the replacement commits. Resume that phase rather than retraining the learner's completed task.
- Exact resume requires matching scientific configuration, data/source/preprocessing fingerprints, and compatible code/state schema. Changing a model, seed, split, method, stream order, or scientific budget (such as replay capacity or training-token limit) starts a new run. Log operational changes separately: output relocation, checkpoint cadence, or additional staging-disk capacity may change without invalidating resume when content identity, numerical behavior, and algorithmic access remain unchanged. Such changes cannot relax the 700M ceiling or discard required state.
- Restore all component state before fetching data or drawing randomness. No partial loading, optimizer reinitialization, or counter reset; weight transfer starts a separate run.
- Metrics and selection logs use stable event IDs and a checkpoint watermark. Ignore or replace uncommitted trailing logical events; resumed execution must not double-count training/evaluation results. Record recovery recomputation cost separately and identify unmeasured work lost in a hard kill.
- Evaluation can resume from committed example/metric events or deterministically rerun its incomplete phase. It must not repeat completed training or require permanently retaining all historical learner checkpoints; evaluate each required checkpoint before it is eligible for pruning.

## Rolling disk retention

- Retain **latest complete recovery checkpoint plus one previous fallback**. On completion, keep final resumable state and remove an unreferenced fallback. Inference exports reuse weight blobs. Extra best/milestone exports require an explicit policy and byte budget.
- One in-progress save can temporarily coexist with the retained pair. Budget for this peak, the old/new replay versions during commit, and all required dependencies before writing. Stream serialization instead of collecting full copies of every model in RAM. If a save cannot fit, preserve the valid checkpoint and report the failure; do not destroy it to make space.
- Prune only after the replacement checkpoint is verified and published. Delete only run-owned artifacts unreferenced by **all** retained checkpoint manifests, current task records, or explicitly pinned exports. An interrupted cleanup must remain restartable. Keep small manifests, hashes, metrics, and failure records even after obsolete weights are removed.
- Recovery snapshots and algorithm-required TSR teachers are different artifact classes. The rolling pair does not authorize deleting a still-needed teacher. Store a shared immutable base once and exact adapter states per required era when the chosen training method permits it; teacher snapshots contain no optimizer state.
- Exact reconstruction is required. If the backbone changes, adapter-only storage is insufficient: budget the necessary changed weights or reject that configuration. Lossy teacher compression, dropping eras, or replacing them with the latest teacher requires an explicit algorithm adaptation.
- Charge all required teachers, auxiliary states, caches, recovery dependencies, and staging bytes to declared storage budgets. Native TSR's task records can still grow; if they do not fit, report an incompatible run instead of hiding them outside the budget. Prune an evicted hybrid teacher only after no retained recovery state or task record references it.

## Bounded RAM/VRAM residency

Keep the active learner and at most one required auxiliary/teacher model resident on the accelerator by default, subject to the byte budget; offload or serialize the learner at a safe boundary if both do not fit. Load teachers on demand and evict the previous teacher from the bounded cache. Do not preload all teacher/auxiliary weights into CPU RAM either; keep inactive artifacts on disk.

Sharing immutable base tensors across distinct adapters is allowed only when state and autograd isolation are verified. Never swap an adapter inside a live learner graph. For a tight-memory KD path, compute detached teacher outputs before the student graph, release the teacher when necessary, and bound output buffering; exact loss-preserving chunking is allowed. Charge transfers/reloads to measured cost. Probe and CSReL stages release their models when no longer needed, retaining only state required for resume or subsequent computation.

