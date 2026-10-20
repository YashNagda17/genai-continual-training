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

At empty `S`, use the declared subset-model initialization without optimizer updates; this is the project's explicit empty-set convention and must be compared with the pinned reference implementation before claiming reproduction. Resolve ties by stable sample ID. Allocate integer round sizes deterministically, with the last round taking the remainder; never duplicate records to fill capacity.

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

The paper's native class-incremental setting does not establish this generative port. For SFT, explicitly define comparable per-example target NLL for both scoring models, including tokenizer/template, token mask, and sum-versus-mean reduction. Length normalization changes selection and must be recorded. Auxiliary architecture and training schedule are later configuration; using a small model must not collapse the two scoring roles into one loss.

