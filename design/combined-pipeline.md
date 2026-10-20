# CombinedPipeline

Owns `src/combined_pipeline.py`. Composes paper components into one learner lifecycle; it does not call four standalone `run()` methods in sequence. Implement the same public run/evaluate/predict/save/load contract as `PipelineBase`.

## Recipe boundary

Implements [Idea.md](/Idea.md)'s **task-chunk hybrid**, with the lifecycle below.

Validate the entire recipe before model allocation. Require explicit task boundaries, retained current-task access through memory construction, compatible gradient/teacher capabilities, aL-SAR class/group semantics, a memory allocation policy, and complete resource limits. Omitted numeric settings remain required configuration, not hidden defaults.

## Task lifecycle

1. **Open task.** Expose `D_t` and committed `M_<t`; verify their identities. Reuse the last committed recovery checkpoint, or create the initial one; do not archive an extra full model for each task. TSR probes training examples at the fixed initialization before any current-task training, then computes weights over nonempty retained source tasks.
2. **Admit current data.** OASIS scores each arriving current batch using the current learner, before training on that batch. Update its online statistics once per arrival and retain the admission decision. Replay batches do not update OASIS statistics in this recipe. Subsequent training passes reuse admitted IDs; rescoring them is a separately named schedule.
3. **Schedule updates.** Draw TSR's current/replay branch once per logical optimizer update; hold the branch, source, and teacher across accumulated microbatches. A current draw scans arrivals until admitted records are available, without redrawing the branch after rejection. Keep a bounded ID queue. If the current source is exhausted, follow the configured `stop_task` or `repeat_admitted` policy; repetition requires a nonempty admitted set and does not rescore it. An all-empty stream stops with `no_current_update`; it cannot be replaced by replay or all-current training. Require a finite attempted-update cap as well as the configured step/token/pass limits, including for replay probability one and skipped optimizer steps.
4. **Retrieve and assign teacher.** A replay update first samples source task `j` using TSR weights, then aL-SAR samples only from `M_j`. Use that task's frozen teacher. A current update uses TSR's top-routed teacher. With no eligible history, use current data only and disable teacher loss. Exhaustion and replacement policies are explicit and may not alter task weights silently.
5. **Train.** Execute one learner update through `ModelBase`. aL-SAR uses supervised-loss statistics, while CE plus TSR-KD updates the learner; charge any additional gradient computation. Commit SAR counts, similarity/Fisher statistics, and cadence only on successful optimizer updates. On a skipped update, consume/log the attempted batch and RNG but discard staged SAR observations; OASIS arrival state remains committed. Stop at the attempt cap even if successful-step counts never advance.
6. **Close task.** Persist the required immutable era snapshot, using adapter-only storage when exact, and release its resident copy. CSReL selects from all permitted `D_t`, including OASIS-rejected records, and reselects older task coresets only from retained memory. Resolve task allocations and construct the next memory without granting archive access. Auxiliary CSReL training cannot modify the learner or its teacher snapshot.
7. **Commit and evaluate.** Publish new replay, signature/teacher references, and synchronized policy state as one complete recovery checkpoint with phase `evaluation_pending`. This publication is the memory commit; do not expose it earlier through `ReplayStore`. Release current-task training access, then evaluate held-out views. Mark evaluation complete before opening the next task; recovery must not repeat the completed learner/coreset phase.

Record observed, probed, admitted, trained, replayed, and retained exposure separately. The task-chunk contract grants probes and task-end CSReL the full current training partition, including records not reached by the arrival cursor; log this auxiliary access separately. The protocol must fix pass/step/token limits, partial-batch handling, and exhaustion policy before running.

Before the first update opportunity, the callable implementation keeps its already drawn branch and scans arrivals until it establishes at least one admitted current example or exhausts the stream. This also applies when replay probability is one; it gives the all-empty rule a terminating meaning without redrawing the branch. A retained current queue is consumed only on a current branch.

## Resume and residency

Expose task phase, event watermark, pending admissions, samplers/RNG, auxiliary progress, and policy statistics to the [checkpoint/residency owner](/design/model-weights.md). Uncommitted atomic selection/probe work may restart from its preceding safe state; record recomputation cost.

## Composition guardrails

- TSR sets task probabilities; aL-SAR sets conditional example probabilities. The joint probability is `p(task) * p(example | task)`. A second global retrieval over all tasks would override TSR and is forbidden for this recipe.
- The hybrid uses aL-SAR's retrieval/freezing components, not its standalone memory insertion policy. CSReL is the sole memory writer. The standalone aL-SAR pipeline keeps its own paper-defined memory policy.
- CSReL's global-budget reselection changes TSR's immutable per-task data records. Preserve the task signature and era teacher, but version the hybrid's memory membership and label this as a bounded-memory TSR adaptation.
- Exclude tasks only for declared empty replay membership, renormalize, and log coverage loss. A nonempty task with a missing/corrupt teacher or invalid signature is an error, never an exclusion. With no retained tasks, use current-only updates without KD. Teacher reclamation follows checkpoint dependency rules.
- CSReL selection loss is not TSR's training objective by default. Use the selected CSReL variant's objective. Combining CSReL-KD with TSR-KD needs a separate specification, including logit storage and loss weights.
- Hybrid SAR exposure updates use the active-population rule in [al-sar.md](/design/al-sar.md), not standalone Algorithm 1's selection-time clock.
- LoRA, frozen backbone layers, and adaptive freezing must agree on actual trainable paths. Reject a freeze plan that prevents the required gradients or claims savings while performing the same backward pass.

## Required controls for later experiments

Provide configurations for new-only, uniform replay, individual components, and the complete hybrid under the same data/order/evaluation contract. Define cumulative retraining from a fixed initialization separately from cumulative warm-start fine-tuning. Record both access and total cost; neither cumulative reference has the bounded-history restriction. Actual model/dataset choices, comparison budgets, and success margins belong to the later experiment protocol.
