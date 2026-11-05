# AnswerSelectionPipeline

Owns `src/pipelines/answer_selection.py`: `AnswerSelectionConfig`, `AnswerSelectionPipeline(CSReLPipeline)`, and `FixedMemorySelection`. This is Part 1 of [Idea_Refined.md](/Idea_Refined.md), "Pick the best old questions". Its role is **memory construction only**. The [recipe guardrails](/design/generation-replay.md) apply.

Base: [CSReL reference](/Papers/CSReL-summary.md) and [CSReL contract](/design/csrel.md). Replacing CSReL's loss with answer disagreement is a labeled `adaptation`, not a reproduction.

## Score

CSReL's subtraction stays. Loss on the correct answer becomes disagreement between a model's own answers:

`score(x) = disagreement(x; weak) - disagreement(x; strong)`

`disagreement(x; model) = 1 - mean_{i<j} sim(a_i, a_j)`, over `K` answers sampled from that model for question `x`. `sim` is the [token similarity](/design/repair-stop.md) used throughout the recipe. Implemented as `numerics.answer_disagreement`.

| Role | CSReL name | Training data |
|---|---|---|
| Weak | Subset model, `subset_initialization` | Rows picked in earlier rounds. Untrained at empty `S` |
| Strong | Reference model, `reference_initialization` | The variant's permitted pool (`csrel_cl`: the whole current topic) |

Both are isolated roles trained on gold targets by the inherited `fit_model`. Neither is the learner.

| Weak model's answers | Strong model's answers | Score | Keep? |
|---|---|---|---|
| Messy | Clear | High | Yes, there is something real to teach |
| Messy | Messy | Near 0 | No, the question is naturally open |
| Clear | Clear | Near 0 | No, it is already learned |

`K`, temperature, `top_p`, and `max_new_tokens` are required config. These samples are for scoring only and are never replay targets.

## Reuse of CSReL

The subclass inherits fitting, rounds, task quotas, older-task reselection, and resumable stages from `CSReLPipeline`, and overrides only:

| Method | Change |
|---|---|
| `score_losses` | Samples answers and computes disagreement instead of NLL. Table rows keep the `loss` field so `reducible_losses` works unchanged, and add `kind="answer_disagreement"`, `samples`, and the generation spec |
| `select_top` | Selects within groups, using the quota rule below. Remains stable on ties by sample ID |
| `select_coreset` | On first entry for a task, assigns groups, stores `selection["scoring_ids"]` (the `candidate_limit` draw), checkpoints, then calls the parent |
| `score_reference`, `select_round` | Filter candidates to `scoring_ids`, then call the parent. `train_reference` still receives the full pool |
| `initialize_selection` | For `weak_only` and `random_in_group`, starts at `subset_fit` with an all-zero strong table |
| `required_capabilities` | Adds `generation`, and `prompt_embedding` under `kmeans` |
| `auxiliary_roles` | Adds `("group", group_initialization)` under `kmeans` |

Use `csrel_cl` first, because `Idea_Refined.md` defines the strong model as having seen the whole topic. Under `csrel_cl`, older-task reselection reuses each retained row's stored strong score, with its provenance, and re-samples only the weak role.

## Groups

`Idea_Refined.md` keeps questions from different groups. The pilot's CSReL memory held only 3 of 15 banking intents ([audit](/results/experiments/pilot-600-audit.md)).

| `group_source` | Groups come from | Use on |
|---|---|---|
| `class` | Genuine `Record.class_id`. Never task IDs, and not ScienceQA's A–D answer letter | CLINC |
| `kmeans` | Seeded `numerics.kmeans` on prompt embeddings from the frozen `group` role: the mean final hidden state over prompt tokens, L2-normalized | MeetingBank, Py150, ScienceQA, 20Minuten |
| `none` | One group | Control |

- Each round's count splits across groups by size, using `numerics.largest_remainder`, so the total is exact. Remainder ties go to the lower group ID.
- Unused slots from a short group go to the highest-scoring remaining rows, and each such move is logged.
- Set `groups` no larger than a task's quota, or most groups round to zero.
- Store each retained row's group in `policy["groups"]` as a plain string, so older-task reselection reuses it. Release embeddings after clustering.
- k-means groups are a training-only policy decision. They are not written to `Record.group_id` or `class_id`, and they are not claimed to be question types. Check on the first run whether they track content or just template.

## Cost limit

Answers sampled per task are about `K * C * (1 + R)` for `C` scored candidates and `R` rounds.

| Candidates `C` | `K` | `R` | Sampled answers per task |
|---|---:|---:|---:|
| 600 | 5 | 3 | 12,000 |
| 5,000 (a full TRACE task) | 5 | 3 | 100,000 |

So scoring needs a cap on TRACE. `candidate_limit` draws a seeded subsample, proportional across groups, from a generator independent of replay RNG. **The strong model still trains on the whole permitted pool**; only scoring is limited. Report the limit as an adaptation. Preflight rejects a run whose estimate exceeds `max_generations`. Every sample is charged to selection cost.

## Boundaries

- Sampling uses `predict` with an explicit seed from `(run seed, task, stage, round, sample index)`, derived through `fingerprint`, so resume repeats the same draws. Each batch's `K` samples commit together, and the next boundary follows.
- Answer samples containing reasoning tags are dropped. With fewer than two samples left, disagreement is 0 and the row logs `degenerate`.
- Scores bind to role revision and generation spec. Do not compare disagreement across different specs.
- The selector returns records and does not read `Record.practice`. The composer owns answer writing and commit.

## Controls

`score_form` selects the variant. Each uses the same quotas, groups, and budget.

| `score_form` | Behavior | Compares against |
|---|---|---|
| `gap` | `disagreement(weak) - disagreement(strong)`, the proposal | |
| `weak_only` | One round with the strong table all zero and the untrained role as "weak" | Whether subtraction helps |
| `random_in_group` | Seeded uniform score with no sampling | Whether scoring beats plain grouping |

Loss-based CSReL is the existing `CSReLPipeline`, and uniform random memory is `MemoryBaselinePipeline`. Reject Part 1 if retained quality stays within 1 point of `random_in_group`.

`FixedMemorySelection` replays a saved per-stage ID manifest instead of computing selection, so Part 3 arms use identical kept questions without re-running this expensive part. It charges the source run's measured selection cost by reference and rejects any ID not in the permitted pool.

## Config

`AnswerSelectionConfig(CSReLConfig)` adds these fields, all required: `samples`, `temperature`, `top_p`, `max_new_tokens`, `score_form`, `group_source`, `groups`, `group_initialization`, `candidate_limit`, `max_generations`.
