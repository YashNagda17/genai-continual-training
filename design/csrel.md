# CSReLPipeline

Owns `src/pipelines/csrel.py` and `CSReLPipeline`. Its composable role is **memory construction/reselection**; its standalone role includes the paper's continual learner and replay protocol.

Source and derivation: [CSReL reference](/Papers/CSReL-summary.md), §§3.2, 4.1–4.4, Algorithm 1, Appendices E–G.

## Algorithm contract

For permitted candidate data `D`, selected subset `S`, holdout/reference model `theta_D`, and subset model `theta_S`, the core score is:

`ReL(i) = loss(i; theta_S) - loss(i; theta_D)`.

The paper's “holdout model” is trained on permitted **training candidates**, not the evaluation holdout. Maintain distinct reference, subset, and continual-learner roles.

1. Train the reference model on the variant's permitted pool. Compute reference losses aligned to candidate IDs.
2. Start `S` empty. At each selection round, initialize and train the subset model on `S`, score only unselected candidates, and add the highest-ReL records up to that round's allocation.
3. Recompute subset-model losses after each round. Reusing a single ranking for the whole coreset is a different algorithm.
4. Return selected IDs plus reference-loss provenance and cost. Do not change the continual learner or commit memory inside the selector.

At empty `S`, use the declared subset-model initialization without optimizer updates. The author confirmed this reading on 2025-06-23 in [CSReL issue 2](https://github.com/RuilinTong/CSReL-Coreset-CL/issues/2): the first coreset model is randomly initialized and has learned nothing, so the first round prefers low reference loss when subset losses are similar. The vision training loop itself is not pinned to a commit in this repository. Resolve ties by stable sample ID. Allocate integer round sizes deterministically, with the last round taking the remainder; never duplicate records to fill capacity.

The callable CL-Prv profile follows Appendix F Algorithms 3–4: each reselection reference is fitted on the candidate task's permitted pool union the committed old memory (deduplicated by observation ID). Subset fitting remains on the selected subset. Unlike CL, Prv recomputes reference losses when that permitted pool changes. Equal-task quota remainders go to ascending task IDs; unused slots are reported and not redistributed. Auxiliary training is a finite number of ordered passes with a saved pass/cursor after every update attempt.

Zero quota returns empty memory without auxiliary fitting. Quotas above available unique candidates retain all available records and report underfill; unused quota is not redistributed unless the declared allocation policy permits it.

## Variants remain separate

| Variant | Reference and memory behavior |
|---|---|
| `csrel_cl` | Reference trained on the current task; task-end memory allocated equally across seen tasks; older coresets reselected only from their retained records, reusing stored original reference losses (Appendix E). |
| `csrel_cl_prv` | Reference trained on current candidates plus permitted memory, with the paper's previous-task treatment (Eq. 8, Appendix F). Do not substitute current-task-only reference losses. |
| `csrel_rs` | Streaming reservoir update with ReL-dependent admission and its own memory model/update schedule (Appendix G). Not equivalent to task-end greedy selection. |
| KD extension | Explicit prediction/KD score combination and stored logit provenance (Eqs. 11–12). Not automatically enabled by TSR distillation. |

The hybrid uses a named task-end variant (`csrel_cl` or `csrel_cl_prv`), supplied by configuration. RS and KD are separate implementation targets only when selected by the protocol, not mandatory speculative branches.

Standalone CSReL-CL trains with separately normalized current and replay losses, `mean(current loss) + alpha * mean(replay loss)` (Appendix E, Eq. 17); omit the replay term for empty history. Do not replace this with an unweighted mean over a concatenated batch. The hybrid's learner objective is specified separately.

## State and budgets

- Persist retained IDs, quotas, reference-loss provenance, selected subset/round, candidate cursor, and active auxiliary-training state under [model-weights.md](/design/model-weights.md).
- Train/score the reference first; release its weights/optimizer after the complete loss table is persisted and recovery dependencies permit. Train subset models sequentially, retaining active state and fallback dependencies rather than every round.
- For global example capacity `K`, equal task allocation uses deterministic integer quotas whose sum does not exceed `K`. If `K` is smaller than the task count, report unrepresented tasks. A bytes/token cap additionally constrains selection and creates a labeled budget adaptation; never truncate silently after claiming the paper's allocation.
- Older raw task data cannot be reopened to refresh reference losses or improve reselection. Cached losses must remain bound to their original model/target/reduction. Do not reinterpret stale losses as current-model scores.
- All current `D_t` can be candidates under the hybrid's task-chunk policy, including data OASIS did not train on. Restricting CSReL to OASIS-admitted records is a distinct ablation.

## Language-model adaptation

The paper's native class-incremental setting does not establish this generative port. For SFT, explicitly define comparable per-example target NLL for both scoring models, including tokenizer/template, token mask, and sum-versus-mean reduction. Length normalization changes selection and must be recorded. Auxiliary architecture and training schedule are later configuration; using a small model must not collapse the two scoring roles into one loss. A short fitting schedule is an experiment setting, not a reproduction of the paper's epoch count.

## Unresolved against the author repository

- CSReL-RS and the KD score are not implemented. CL does not stand in for them.
- The published numbers are class-incremental vision results. Per-example response-token NLL on a causal LM is a labeled adaptation.
- Author issue 2 settles the empty-subset initialization. The surrounding ResNet schedule, augmentation, and class-incremental head were not line-checked against a pinned commit in this session.

## Matched baselines for the CSReL experiment

`src/pipelines/memory_baselines.py` is separate from `CSReLPipeline`. OASIS, TSR, aL-SAR, and `CombinedPipeline` do not call it.

| Method | What changes |
|---|---|
| `new_only` | Current-task batches only. Task-end memory is empty. |
| `random` | Same uniform replay, replay weight, batch size, and separate current/replay means as standalone CSReL. Memory is an equal-quota sample drawn without replacement. Selection has its own seed and does not consume the replay RNG. |
| `cumulative` | At each task, reload the frozen weights saved from the run's initial learner, then train on the concatenation of seen-task training views. Replay stays empty. This is complete-data retraining from one initialization. |

Random and CSReL comparisons share the learner, initialization, task order, seed, optimizer settings, global replay budget, and evaluation. Only the memory writer changes. Older random and CSReL candidates are retained examples only.

