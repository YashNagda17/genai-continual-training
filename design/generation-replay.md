# GenerationReplayPipeline

Implementation design for [Idea_Refined.md](/Idea_Refined.md): generation-aware repair and replay. Owns the `GenerationReplayPipeline` class in `src/combined_pipeline.py`, plus the reuse map and the guardrails below, which apply to all recipe code.

Status: contract only, 2026-11-05. Nothing here is built or measured. `CombinedPipeline` stays in the same file, unchanged, as the old-hybrid comparison arm ([contract](/design/combined-pipeline.md)).

Read with: [answer-selection.md](/design/answer-selection.md) (Part 1), [past-self-replay.md](/design/past-self-replay.md) (Part 2), [repair-stop.md](/design/repair-stop.md) (Part 3), [generation-replay-experiments.md](/design/generation-replay-experiments.md) (runs).

## Design guardrails

Every design change for this recipe must follow these.

1. **Papers give ideas, not limits.** Parts 1 and 2 extend [CSReL](/Papers/CSReL-summary.md) and [TSR](/Papers/TSR-summary.md), and their code is reused. [Part 3](/design/repair-stop.md) is our own mechanism. Ideas may come from any paper, but no contract depends on reproducing one, and each mechanism is defined here in its own terms. Credit stays in the paper tables of `Idea_Refined.md`.
2. **OASIS and aL-SAR stay out.** The recipe never calls them. Their code and contracts stay as they are, and they run only in the `old_hybrid` arm.
3. **Existing assets first.** Use the datasets in `src/datasets/data/` and the model adapters in `src/models/`. A new dataset, model family, or inference backend needs an explicit request.
4. **Reuse before adding.** Use existing code as it is if it fits, then subclass it, and only then write something new. Each new item must appear in the tables below.
5. **Change the owner first.** Each concern has one [owning contract](/design/README.md). Shared additions go in that contract's "Generation-aware replay additions" section. `csrel.md`, `tsr.md`, `oasis.md`, and `al-sar.md` describe paper methods and do not change.
6. **Each mechanism has a control and a drop rule.** No arm without a question, and no mechanism without a way to [drop it](/design/generation-replay-experiments.md).
7. **Numbers are inputs.** Values in `Idea_Refined.md` were made up to show the method and are never defaults. Thresholds and budgets are required config, fixed before a scientific run.
8. **Honest claims.** Nothing counts as a result until it is measured in `results/`. Label work as `adaptation`, `hybrid`, `ablation`, or `project` ([method identity](/design/pipelines.md)). Novelty stays unverified, and a borrowed idea is never presented as new.
9. **Ideas log.** New algorithmic ideas go in [optimizations.md](/optimizations.md), using its template, not in design files.

## Code guardrails

All recipe code must follow these.

| Area | Rule |
|---|---|
| Existing behavior | `CombinedPipeline`, `CSReLPipeline`, `TSRPipeline`, `MemoryBaselinePipeline`, and the current experiments behave as before. New fields default to current behavior. Existing tests pass unchanged |
| Layout | Modules go where the tables below say. Namespace packages, relative imports as in existing code, no `__init__.py` |
| Dependencies | The three parts never import each other. Models contain no method names. Datasets know nothing of practice answers. Utilities import no concrete model, dataset, or pipeline |
| Numerics | Native Torch for training, calculations, and inference, with GPU detection and CPU fallback through `resolve_device`. No external backend or new dependency |
| Learner updates | Only `owner.execute` updates a model, and only the executor steps an optimizer. `predict` and `score` preserve weights, optimizer, modes, and training RNG |
| Data access | Prompts for sampling, checks, and repair go through runtime collation, which rejects held-out records. Test data never reaches policy code. Older raw data is never reopened |
| Roles | Auxiliary roles use `acquire_auxiliary` and `release_auxiliary`, one at a time. Every role stays under 700M parameters. Refresh and writing add no resident model |
| State | `policy` holds plain JSON. No tensors, and no integer dictionary keys, because JSON turns them into strings. Each component uses its own `self.rng`. Generation seeds come from `fingerprint`, never global RNG |
| Resume | Each long loop keeps a cursor in `policy` or `progress`, then calls `owner.record_event` and `owner.boundary()` after each batch. Phase changes use `boundary(force=True)`. Only the lifecycle owner commits memory |
| Cost | Every generation and scoring forward records a [cost category](/design/utils.md). No uncounted work |
| Style | Named algorithm steps with short docstrings, as in `csrel.py` and `tsr.py`. Dataclass configs with required fields. No plugin framework, custom exceptions, or options without an arm. Shared checks stay at existing boundaries, and a config only checks its own enum values, as `MemoryBaselineConfig` does |
| Tests | Compact, table-driven checks in `src/tests/test_generation_replay.py` on the tiny fixtures in `fakes.py`, with no downloads. See [tests.md](/design/tests.md) |
| Outputs | Run outputs go under `results/`, each arm in its own attempt directory. Never delete an existing attempt |

## Reuse map

| Idea_Refined step | Reused as it is | New on top |
|---|---|---|
| Stream of topics | `TraceSource`, `InstructionDataset`, `bind_instruction_dataset`, `LimitedViews` | `reserve_response`, `collate_prefixes` |
| Train weak and strong models | `CSReLPipeline.fit_model`, `train_reference`, `train_subset`, `acquire_auxiliary` | |
| Picking rounds and quotas | `CSReLPipeline.select_coreset`, `reducible_losses`, `build_memory`, `commit`, `numerics.quotas` | Disagreement score, group quotas |
| Task fingerprints | `TSRPipeline.probe_task`, `task_signature`, `routing_weights` logic, `numerics.cosine_tensor`, `softmax` | |
| Current or replay choice | `TSRPipeline.draw_branch`, `numerics.sample` | Drift term, turn clock, fall-through |
| Saved copies | `TSRPipeline.snapshot_task`, `Checkpoints.blob`, reference-aware `prune` | `retain_copies` |
| Writing answers, checks, rollouts | `CausalLMModel.predict`, `collate_prompts` | EOS trim, prefix batches, `token_similarity` |
| Token surprise | `CausalLMModel._losses` internals | `token_loss` score kind |
| Practice update | `PipelineRuntime.run`, `execute`, `optimizer_update`, `current_batch` | `token_mode`, `TokenTargets`, no-step on zero active tokens |
| Memory commit | `ReplayStore.proposal`, `state_dict`, byte budget, `after_memory_commit` flow | `Record.practice` |
| Resume and retention | `PipelineRuntime.save`, `load`, `boundary`, `Checkpoints.publish` | Refresh and writing cursors |
| Evaluation | `_evaluate_seen`, `continual_metrics`, `return_predictions` | `rouge_l`, `text_metrics` |
| Baselines | `MemoryBaselinePipeline` (`new_only`, `random`, `cumulative`), `CombinedPipeline` | `usual_replay` arm config |
| Reports | `performance_report`, `timing_report`, `storage_report`, `summarize_attempt`, `write_comparison`, `attempt_for` in `src/csrel_experiment.py`, imported and not copied | New cost categories |

## New code

### New modules

| Module | Contents | Base | Owns | Never does |
|---|---|---|---|---|
| `src/pipelines/answer_selection.py` | `AnswerSelectionConfig`, `AnswerSelectionPipeline`, `FixedMemorySelection` | `CSReLPipeline` | Kept question IDs, disagreement scores, groups | Write answers, read the learner, commit memory |
| `src/pipelines/past_self_replay.py` | `PastSelfConfig`, `PastSelfReplayPipeline` | `TSRPipeline` | Topic weights, saved copies, answer writing | Choose rows or tokens, apply KD, measure drift |
| `src/pipelines/repair_stop.py` | `RepairStopConfig`, `RepairStopPipeline` | `PipelineBase` | Answer check, repair test, token lists, stop rule, drift measurement | Choose topics, commit memory, step the optimizer |
| `src/generation_replay_experiment.py` | Protocols, arms, first test | None | Run configuration and reports | Contain algorithm logic |
| `src/tests/test_generation_replay.py` | Recipe checks | None | Evidence listed in [tests.md](/design/tests.md) | Require downloads |

### Additions to existing modules

| Module | Addition | Contract |
|---|---|---|
| `src/combined_pipeline.py` | `GenerationReplayConfig`, `GenerationReplayPipeline`. `CombinedPipeline` untouched | This file |
| `src/utils/contracts.py` | `Record.practice`, `practice_record`, `TokenTargets`, `TrainingBatch.token_targets`, `Objective.token_mode`, `Objective.token_allowance` | [utils](/design/utils.md) |
| `src/utils/numerics.py` | `token_similarity`, `answer_disagreement`, `kmeans`, `largest_remainder`, `rouge_l` | [utils](/design/utils.md) |
| `src/utils/runtime.py` | `collate_prompts`, `collate_prefixes`, passing `token_targets`, `max_task_seconds` | [utils](/design/utils.md) |
| `src/utils/state.py` | `ReplayStore.usage["answer_tokens"]` | [utils](/design/utils.md) |
| `src/pipelines/pipeline_base.py` | Default `token_targets` hook returning `None` | [pipelines](/design/pipelines.md) |
| `src/datasets/dataset_instruction.py` | `reserve_response`, `collate_prefixes` | [datasets](/design/datasets.md) |
| `src/models/model_causal_lm.py` | `token_loss` and `prompt_embedding` score kinds, EOS trim in `predict`, token-level helper behind `_losses`, `token_mode` in `train`, `text_metrics` in `evaluate` | [models](/design/models.md) |
| `src/tests/fakes.py` | `predict`, `token_loss`, and `collate_prompts` on the tiny fixtures | [tests](/design/tests.md) |

### Composer

```python
GenerationReplayPipeline(selection=..., routing=..., repair=...,
                         config=GenerationReplayConfig(no_replay_action="current"),
                         model=model, dataset=dataset)
```

- `selection` is anything with `build_memory` and `after_memory_commit`: `AnswerSelectionPipeline`, `FixedMemorySelection`, `CSReLPipeline`, or `MemoryBaselinePipeline(random)`.
- `components()` returns `generation_replay`, `selection`, `routing`, `repair`. `required_capabilities()` is the union of the parts. `auxiliary_roles()` is the selection roles plus TSR's fixed probe.
- Part 3 is a component, not a paper method, so it has no standalone lifecycle: its `open_task`, `next_update`, and `finish_task` raise `NotImplementedError`. It inherits `PipelineBase` only for policy, RNG, and model plumbing. Its independent run is the composer with uniform routing and random memory (`repair_only`). Parts 1 and 2 inherit standalone lifecycles from their paper classes.

## Task lifecycle

1. **Open.** `routing.probe_task(self.current, self)` computes the TSR signature on the current training view. The turn counter restarts.
2. **Train.** `next_update` below runs until a limit is reached or current data ends. One learner, one optimizer, one origin per update.
3. **Snapshot.** At task end, `routing.snapshot_task(self)` saves the exact learner as the saved copy.
4. **Select.** `selection.build_memory(self.current, self.replay, seen_tasks, self)` proposes kept IDs. Current-task candidates come from `D_t`, and older tasks reselect only from retained rows.
5. **Write.** `routing.write_answers(new_rows, self)` writes one answer and its baselines for each newly kept current-task row. Retained older rows keep their answers. Writing is resumable and happens before commit.
6. **Commit.** The returned records carry their answers, so `self.replay.proposal(records)` commits membership and answers in one step and checks the byte budget. The composer's `after_memory_commit` calls, in order: `selection.after_memory_commit()`, `routing.commit_topics(self.replay)`, and `repair.prune_rows(self.replay)`. The new task's drift starts at 0, because the learner and saved copy have the same weights at that moment.
7. **Evaluate.** Held-out views only, through the shared evaluation path.

## Update loop

```text
next_update():
  while no limit is reached:
    if no branch is pending:
      plan = routing.route_replay(signature, replay)
      branch = routing.draw_branch(plan)
      if branch is replay:
        turns += 1
        refresh_due = (turns - 1) % R == 0
      save branch and refresh_due in progress

    if refresh_due:
      repair.refresh(replay, self)        # resumable, checkpoints between batches
      routing.set_drift(repair.drift())
      clear refresh_due

    if branch is replay:
      task = routing.next_active_task(branch, plan, repair.active_tasks())
      if task exists:
        return repair.retrieve(task, replay, batch_size), replay origins, repair.objective(), None, None
      if config.no_replay_action == "skip":
        count one attempt, record the skip, clear branch, continue

    records = current_batch()
    if no records: return None
    return records, current origins, Objective(reduction), None, None
  return None
```

- **A turn counts even if nothing needs replay.** This keeps refreshes firing when every row has recovered. Without it, a row that slips later would never be found.
- `next_active_task` keeps the drawn task if it has active rows. Otherwise it redraws from the plan's weights, renormalized over tasks with active rows.
- `no_replay_action="current"` gives the update to new learning. `"skip"` performs no update, which measures time saved directly. A skip consumes an attempt, so the loop ends even at replay probability 1.
- The drawn branch uses the drift known before that refresh. New drift applies from the next draw. A resumed run reuses the saved branch and never redraws it.
- The `token_targets(batch)` hook supplies per-token baselines and weights for replay batches and returns `None` otherwise.

## Composition rules

- **No KD.** `kd_weight` is 0 and `teacher_task` is `None`. Saved copies write answers and are not loaded per batch.
- **Plain current data.** Current batches are unfiltered, in arrival order. Replay rows are uniform within a topic's active rows.
- **One memory writer.** Only `selection` chooses members. Part 2 adds answers to members and Part 3 only reads them.
- **Gold answers stay stored.** The original target stays on each kept record, so saved-copy errors can be measured. It counts toward the replay budget and is not trained on unless the writer mode is `gold`.
- **Refresh interrupts training; it is not an update.** It runs on the learner through `predict` and `score`, and writing does the same, since the learner matches the saved copy at that moment.

## Build order

Each step is usable and tested before the next. Stop at any step whose result triggers a [drop rule](/design/generation-replay-experiments.md).

| Step | Builds | Why here |
|---|---|---|
| M0 | Writer, `token_loss`, prefix rollouts, similarity, `refresh`, and the first test | Tests the main claim with no training changes, as `Idea_Refined.md` requires. If gains are noise, the repair test is not built further |
| M1 | Hinge loss, cuts, live stop, composer with uniform routing and random memory | Runs Part 3 end to end |
| M2 | Drift routing | Needs drift from M0 |
| M3 | Answer selection | Most expensive and independent of the rest, so it comes last |
| M4 | Full arm matrix and reports | Needs all of the above |

## Where this design differs from Idea_Refined.md

| Topic | Idea_Refined.md | This design | Reason |
|---|---|---|---|
| Paper basis | Each part is described through its papers | Parts 1 and 2 extend CSReL and TSR. Part 3 is our own mechanism | Project decision. Papers give ideas, not definitions |
| Disagreement | Paper-specific scoring | `1 - mean pairwise similarity` of sampled answers | Uses the same similarity as Part 3, with no embedding path |
| Answer writing | Text sampled from the saved copy | Greedy, with gold as a control | The answer check needs an answer the saved copy can reproduce |
| ScienceQA token-swap comparison arm | Listed as a comparison | Not built for now | Part 3 is judged against its own controls. Add it later if a comparison with that approach is needed |
| Candidates for Part 1 | Every question scored | Scoring capped by `candidate_limit` | 100,000 sampled answers per full TRACE task at `K=5` and 3 rounds |
| Noise in repair gains | "Repeat some tests" | Replicas use sampled decoding | Greedy repeats are identical and would report zero noise |
| Baseline storage | 7,200 bytes for 60 rows of 60 tokens | Charged at serialized JSON size | Policy state is JSON, so 2 bytes per value does not hold |
| Savings from shorter sentences | Up to 3 of 7 tokens | Reported per task against the prompt too | Prompts dominate on three of four tasks. See the [length table](/design/generation-replay-experiments.md) |
