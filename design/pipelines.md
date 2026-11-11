# Pipelines and PipelineBase

The active pipelines are `RevisionPipeline`, `AdapterRoomPipeline` and `GenerationShrinkPipeline`, injected with model/data adapters through `RunContext`.

`PipelineBase` delegates lifecycle, resource validation and complete recovery to `PipelineRuntime`. Pipelines own policy and RNG; the model alone steps the optimizer. Checkpoints preserve pending work, cursors, replay, policy/RNG and numerical state. Failed or interrupted work keeps the last committed recovery state.

Policy `_commit()` checks the configured event cadence rather than forcing a checkpoint. Lifecycle transitions retain forced saves. `release_auxiliary(restore_learner=True)` preserves the default restoration behavior; analysis jobs may pass false while switching roles or publishing results, keeping the learner's resumable snapshot offloaded. A passing final shrink candidate may be reused for export under its existing auxiliary key.

Public operations are `run`, `evaluate`, `predict`, `save/load`, `set_model` and policy state snapshots. Weight manipulation belongs directly to the model. Removed APIs include instruction binding and pipeline weight-wrapper methods.

`next_update` returns `(records, origins, objective, hooks)` or `None`; no teacher or KD fields remain. `Objective` selects the supervised reduction and token mode; `UpdateHooks` supplies token targets only. Policy collation rejects held-out records. Evaluation and declared validation decisions use separate guarded paths. The Hindi access rule permits only current Hindi training plus the English pool, except the explicit full-English reference.

## Component operations

The [summarization plan](/design/summarization-plan.md) adds three pipelines, with the same rule that proposals never mutate shared state.

| Capability | Input -> output | Owner |
|---|---|---|
| `write_past` | Pool records -> records with past summaries and baselines | `RevisionPipeline` |
| `rank_pool`, `replay_slots` | Pool, selector -> ranking, slot IDs per batch | `RevisionPipeline` |
| `retire`, `token_targets` | Active rows, regenerated summaries -> retired IDs; replay batch -> weights | `RevisionPipeline` |
| `location_maps` | Saved checkpoint, probe rows -> change, pull, and clash maps | `AdapterRoomPipeline` |
| `adapter_plan` | Arm, maps -> `AdapterPlan` | `AdapterRoomPipeline` |
| `shrink_search` | Finished checkpoint, validation views -> `k_E`, `k_G`, shrunk weights | `GenerationShrinkPipeline` |

- `RevisionPipeline` has a standalone lifecycle with two stages: `english`, then `hindi`. The Hindi stage is a new run that loads the English checkpoint's weights.
- `AdapterRoomPipeline` and `GenerationShrinkPipeline` don't train. `location_maps` and `shrink_search` are analysis jobs. `adapter_plan` returns a record that the Hindi stage applies before training.
- The placement and shrink rules are in [summarization-experiments.md](/design/summarization-experiments.md). The model operations they call are in [models.md](/design/models.md).
- The experiment runner composes the three pipelines. They never import each other.
- These are deliberately **not** composed in `CombinedPipeline` or `GenerationReplayPipeline`. The current plan uses neither.

Method identity: X, A, the full-English reference, C1, and C2 are `ablation`. B is `adaptation`. G1, G2, G3, E, and F are `project`.
