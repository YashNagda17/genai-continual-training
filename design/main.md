# Entry point and run matrix

Owns `src/main.py`. Parse a configuration, expand independent runs, instantiate adapters/components, call the lifecycle owner, and record status. Keep selection, training, metric formulas, and paper-specific defaults out of this file.

## Configuration contract

A resolved run specification must identify:

- Method or hybrid recipe, paper variants, adaptation labels, and design/source revisions.
- Learner, tokenizer/template, initial checkpoint, adapter/training capabilities, and auxiliary-model roles.
- Dataset revisions, processing/splits, task order/boundaries, source access policy, and evaluation metrics/decoding.
- Replay capacity and units, auxiliary/storage limits, selection/replay rates, training/update limits, losses, optimization settings, precision, and device requirements.
- Seed list, output root, positive periodic checkpoint interval, resume mode, retention limits, CPU/GPU residency and checkpoint-disk byte budgets, and the later approved experiment protocol reference for scientific runs.

Use a closed schema: unknown keys and omitted required values are errors. Validate ranges and incompatible combinations before downloads or model allocation. Include a validation/dry-run mode that resolves the matrix and required capabilities without fetching weights, loading data payloads, or training.

## Matrix semantics

Models, method recipes, task sequences, and seeds are explicit independent axes. A list of datasets forming one continual stream is **one task sequence**, not automatically an axis of separate runs. Print the expanded run count, ordered task sequence, and output identity in dry-run output. Reject duplicate scientific run identities.

Each independent run creates fresh learner/optimizer, policy state, replay, and RNG streams. Share only immutable downloaded assets or fingerprinted preprocessing. Default to sequential execution; resource-aware parallel scheduling is a later extension, not necessary architecture.

## Run identity and recovery

- Derive `run_id` from canonical scientific configuration, seed, data/model/preprocessing fingerprints, and implementation/design version; use source checksums without Git. Each execution attempt has a unique `attempt_id` and isolated directory. Operational changes follow [model-weights.md](/design/model-weights.md).
- Support `resume=auto|required|never`, defaulting to `auto`. `auto` skips a matching completed run with validated outputs, resumes a matching incomplete run from its latest valid checkpoint, and starts fresh only when no prior run exists. `required` fails without a compatible checkpoint. `never` creates a new attempt without overwriting prior artifacts.
- In `auto`/`required`, an existing incomplete run with no usable checkpoint or incompatible scientific configuration is an error, not a silent restart. Validate manifests and dependencies; a corrupt/torn newest save may fall back to the retained previous checkpoint with an explicit recovery record stating lost progress. Never merge states from different checkpoints.
- `auto`/`required` select the explicit `attempt_id` or the sole existing attempt; multiple attempts require an explicit choice. Resume reuses that directory, while `never` allocates a new one. An explicitly requested missing attempt is an error.
- Report resumed task, phase, committed optimizer step, and pending evaluation/memory work. Acquire the run-writer lock before continuing. Resolve the complete run matrix with these decisions in dry-run mode, without loading checkpoint model tensors.
- Persist a manifest before execution and terminal status afterward: completed, failed, interrupted, or rejected. Preserve failure details and the last complete checkpoint. Return nonzero if any requested run fails; continuing other independent runs requires an explicit continue-on-error setting.
- Support multiple seeds without choosing the scientific seed count in code. Aggregation selects one completed attempt per scientific `run_id`, reports missing runs, and stays outside training loops; retries never count as independent seeds.

Follow [model-weights.md](/design/model-weights.md) for the rolling latest-plus-one retention policy, completion cleanup, and teacher dependencies. Check both per-run and remaining output-root disk capacity before staging a save. Never delete another run's artifacts automatically to free space.

Intended entry point: `python -m src.main --config <path>`.
