# Pipelines and PipelineBase

Owns `src/pipelines/` and `pipeline_base.py`. Each paper implementation inherits `PipelineBase`, runs independently, and exposes its algorithm steps for composition in [CombinedPipeline](/design/combined-pipeline.md) and other hybrids.

## Public lifecycle

| Operation | Contract |
|---|---|
| `run(context)` | Validate capabilities and access policy; execute the configured task stream; return artifact references and final status. |
| `evaluate(evaluation_context, evaluation_spec)` | Evaluate a designated checkpoint using held-out access isolated from policy code. |
| `predict(inputs, generation_spec)` | Delegate to the learner's inference path; never select or train. |
| `save(path)` / `load(path)` | Coordinate complete pipeline state, including model references, data position, memory, policy statistics, and RNG. |

The base class owns common lifecycle delegation through the generic runtime. It does not contain a switch over paper names or a default replay algorithm. Paper classes implement their supported component operations.

## Initial callable implementation

For the current logic-review stage, paper classes contain named algorithm steps and thin lifecycle composition. Inline algorithm/configuration validation, custom exceptions, and warning branches are omitted; callers supply valid paper settings and compatible models. Shared record validation remains in `contracts.py`, while generic execution, resource checks and checkpoint handling live in utilities. This review-stage scope supersedes the per-paper fail-fast requirements below until those checks are reinstated.

Every pipeline supports `set_model(model)` for shared learner identity, `get_weights` / `set_weights` for detached weight transfer, `save_weights` / `load_weights` for weight files, and `get_policy_state` / `set_policy_state` for policy/RNG snapshots independent of a run. The hybrid binds the same learner object to all components. Fixed probes, references, subsets and historical teachers remain isolated roles. Weight transfer starts a new numerical training state; `save` / `load` retains the distinct full-resume semantics.

`PipelineBase` declares these composition operations and lifecycle hooks. The generic `PipelineRuntime` mixin implements execution without importing paper classes; its lifecycle owner remains the standalone or combined pipeline instance.

Paper numerical steps use Torch tensors on the shared learner's device, with CPU fallback for unbound components. Numerical results cross back to plain scalars/lists at policy, selection-record and checkpoint boundaries. Exact SIREN still includes all subset terms; signed similarities, loss differences and statistical clocks remain unchanged. Run identity records the resolved device and Torch version because backend changes need not reproduce identical floating-point decisions.

Construct a paper pipeline with its explicit configuration and optional `model`/`dataset`, then call individual steps or `run(RunContext(model, dataset, spec, output))`. Models and datasets are injected adapters; this library does not select experimental inputs or implement the later run matrix. `bind_instruction_dataset(dataset, raw, split_spec=None)` is the instruction-source integration: it keeps official splits unless `split_spec` sets `validation_fraction`, preprocesses with the bound learner's `preprocessing_kwargs`, and calls `attach_instruction`. Truncation comes from the source manifest (`drop` or `prompt_left`) unless `split_spec` overrides it. The learner must be a causal LM already constructed from its local snapshot. `RunSpec` requires finite update/pass limits, resource budgets, source/protocol identity, and a checkpoint cadence. The initial executor accepts one microbatch per update and keeps partial batches. Callers select a supported paper profile.

The shared lifecycle drives open, training, memory construction, and evaluation phases. Paper methods implement the algorithm-specific transitions; the combined pipeline composes their public capabilities. `PipelineBase` prints one stdout line each time a sample sweep crosses another 5%: training arrivals, TSR probes, CSReL fit and score passes, and held-out evaluation. The line names the phase and reports done, total, and the crossed percent. It is not checkpoint state. Event boundaries persist cursors, pending branch/admissions, policy RNG, active auxiliary state and numerical learner state. Each supplied output directory is one attempt: `auto`, `required`, and `never` operate on that directory, with `never` refusing an existing attempt.

Standalone OASIS exposes `historical_pool` (all observed records retained, memory-only sampling) and `current_only` (explicit bounded-access adaptation) profiles. Its logical arrival size and number of update opportunities are configuration, not claimed reproduction settings. Standalone aL-SAR's Greedy Balancing Sampler inserts each arriving record, then removes a uniformly chosen record from a largest class if over capacity; ties between largest classes are sampled uniformly. These choices are recorded pending pinned-code comparison.

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

## Generation-aware replay additions

The [generation replay recipe](/design/generation-replay.md) adds these capabilities, with the same rule that proposals never mutate the shared store.

| Capability | Input -> output | Intended owner |
|---|---|---|
| `build_memory` | As above, scored by answer disagreement | `AnswerSelectionPipeline` |
| `write_answers` | Newly kept rows -> rows with `practice` | `PastSelfReplayPipeline` |
| `route_replay` | As above, plus drift by topic | `PastSelfReplayPipeline` |
| `refresh` | Retained rows -> row state, drift | `RepairStopPipeline` |
| `retrieve`, `token_targets` | Topic, batch -> practice rows, aligned targets | `RepairStopPipeline` |

Method identity gains one status, `project`: a mechanism we define ourselves, with no source PDF or paper section to record. `RepairStopPipeline` uses it, and the full recipe stays `hybrid`.

`PipelineBase` gains a default `token_targets(batch)` hook returning `None`. `RepairStopPipeline` is the one component without a standalone lifecycle, because it is not a paper method and needs a writer and a router. Its independent run is the composer's `repair_only` arm.
