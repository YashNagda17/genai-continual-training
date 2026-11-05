# Entry point and run matrix

Owns the run entry points. The present sample is `src/example.py`. It loads the local SmolLM2 snapshot, prepares CLINC, and runs `CombinedPipeline` for one pass over task `clinc`. Each launch deletes `results/sample/clinc-smollm2` and starts a new attempt. The sample does not resume. The lifecycle evaluates the official test split, including exact-match generation. The script then prints a few generated intents. Paper settings in that file are sample values. The later matrix still parses a closed configuration, expands independent runs, and keeps selection formulas out of the sample file. That matrix remains `src/main.py`.

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

Sample entry point: `python -m src.example`. Intended matrix entry point: `python -m src.main --config <path>`.

## CSReL comparison entry

The `pilot` protocol uses seeded equal-per-class subsets of three CLINC domains, validation evaluation, and an explicit intent-classification instruction. It reports all selected IDs and class counts after preprocessing. It is a development pilot, not a final test result. Cumulative pilot stages must have enough update/token budget for all configured passes over all seen records; reject insufficient budgets. Pilot evaluation persists generated strings separately from aggregate metrics. Selection timing tables sum reference/subset fitting and scoring components, labeled as measured components rather than full selection wall time.

`python -m src.csrel_experiment` runs new-data-only, random memory, CSReL-CL, and complete retraining from the common initial weights. A learner update collates the current batch and the replay batch into one tensor, so the model batch bound is their sum. Each arm writes its own attempt directory under the chosen output root and never deletes an existing attempt. `backend-smoke` uses a tiny local JSONL and the SmolLM2 adapter. `smoke` uses the first two CLINC domains and records any train or test prefix as truncated. `scientific` rejects those prefixes and is the only protocol whose metrics are experiment results. Validation stays in the dataset and is not read by training, selection, or the reported test matrix. This entry does not tune hyperparameters. Forgetting follows [utils.md](/design/utils.md): the mean over earlier tasks of the best earlier score minus the final score.


## Learner-order ablation

The optional `--shuffle-training` comparison flag permutes the full permitted learner view separately per pass, using a deterministic seed derived from run seed, task stage, and pass. Cumulative views mix all seen tasks. Dataset arrival order, memory candidate membership, replay RNG, and auxiliary fitting order remain unchanged. Pending queue/cursor state remains checkpointed; permutations reconstruct without consuming the replay RNG. The flag is part of scientific identity. This is an explicitly labeled learner-order ablation, not an exact paper reproduction.

## Generation replay entry

`python -m src.generation_replay_experiment` runs the first test and arm comparisons for the [generation replay recipe](/design/generation-replay.md). Its [experiment contract](/design/generation-replay-experiments.md) owns protocols, arms, and drop rules. Attempt handling matches the CSReL comparison entry.
