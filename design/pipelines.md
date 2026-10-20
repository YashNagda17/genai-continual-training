# Pipelines and PipelineBase

Owns `src/pipelines/` and `pipeline_base.py`. Each paper implementation inherits `PipelineBase`, runs independently, and exposes only the component capabilities needed by [CombinedPipeline](/design/combined-pipeline.md).

## Public lifecycle

| Operation | Contract |
|---|---|
| `run(context)` | Validate capabilities and access policy; execute the configured task stream; return artifact references and final status. |
| `evaluate(evaluation_context, evaluation_spec)` | Evaluate a designated checkpoint using held-out access isolated from policy code. |
| `predict(inputs, generation_spec)` | Delegate to the learner's inference path; never select or train. |
| `save(path)` / `load(path)` | Coordinate complete pipeline state, including model references, data position, memory, policy statistics, and RNG. |

The base class owns common lifecycle validation and delegation. It does not contain a switch over paper names, a default replay algorithm, or placeholder implementations that return empty results. Unsupported component operations fail explicitly.

## Component capabilities

Use named methods and [shared records](/design/utils.md), without a plugin framework or one class per hook.

| Capability | Input -> output | Intended owner |
|---|---|---|
| `select_current` | Current training batch + score context -> `SelectionResult` | OASIS |
| `probe_task` | Current train-only probe view + fixed reference -> task signature | TSR |
| `route_replay` | Current signature + eligible task records -> `ReplayPlan` | TSR |
| `retrieve_replay` | One eligible task's retained records + count -> ordered sampled IDs | aL-SAR component in hybrid |
| `plan_update` / `observe_update` | Batch/gradient statistics -> freeze plan / updated policy state | aL-SAR |
| `build_memory` | Current train candidates + allowed retained history + allocation -> proposed memory | CSReL |

Classes expose only supported capabilities. Objectives and teacher routing travel with the batch to the model executor.

## State ownership

- A standalone pipeline owns its lifecycle, learner reference, replay store, and paper-specific state. In a hybrid, `CombinedPipeline` owns lifecycle and replay commits; paper components own only their algorithm state.
- Selection/routing returns a proposal plus diagnostics, never mutates the dataset or the shared replay store. The lifecycle owner validates and commits memory. Scoring does not mutate the learner.
- Statistical updates have a declared clock: arrival batch, training microbatch, optimizer step, or task boundary. Persist the clock and last completed event to prevent double updates after resume.
- Components expose state to the lifecycle owner's [checkpoint transaction](/design/model-weights.md); they never write independent checkpoints.
- Evaluation uses the common metric implementation. A paper may add named metrics, but cannot change shared metric definitions or use evaluation scores to route training implicitly.

## Method identity

Every run records `method_id`, algorithm variant, source PDF checksum and section/algorithm, any official-code commit used, and an adaptation list. Use distinct statuses:

- `paper_native`: original algorithm and its declared access/protocol assumptions.
- `adaptation`: an explicit port, budget/access change, or replacement of a paper component.
- `hybrid`: composition of multiple methods under the project recipe.
- `ablation`: an intentional removal or replacement with a named control.

## Failure boundaries

Reject an incompatible gradient capability, unknown task boundary, missing teacher, unavailable class/group semantics, unsupported memory budget, or nonfinite score before silently substituting behavior. Empty history at the first task and an empty probabilistic current selection are expected states with defined behavior, not errors. Unexpected dataset exhaustion is an error unless it is the configured stopping condition.
