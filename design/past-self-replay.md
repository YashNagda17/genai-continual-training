# PastSelfReplayPipeline

> **Current plan:** not used. The [summarization plan](/design/summarization-plan.md) replaces this recipe with the simpler G1–G3 rules in [revision.md](/design/revision.md). Pieces it reuses: `Record.practice`, `practice_record`, `token_mode="weighted"`, `TokenTargets`, and `rouge_l`. This contract is unchanged and applies to the TRACE recipe only.

Owns `src/pipelines/past_self_replay.py`: `PastSelfConfig` and `PastSelfReplayPipeline(TSRPipeline)`. This is Part 2 of [Idea_Refined.md](/Idea_Refined.md), "Practise using your past self". Its jobs are **topic routing with drift, saved-copy retention, and answer writing**. The [recipe guardrails](/design/generation-replay.md) apply.

Base: [TSR reference](/Papers/TSR-summary.md) and [TSR contract](/design/tsr.md) for signatures. This is an `adaptation` of TSR: it adds drift to routing, and saved copies write answers instead of matching token predictions.

## Routing

`w_j = softmax_j((cos(g_j, g_t) + b * drift_j) / tau)`

- `g_j` and `g_t` are TSR signatures, unchanged: fixed initialization, zero-B LoRA view, train-only probes.
- `drift_j` comes from Part 3 through `set_drift`. It is the mean over topic `j`'s rows and answer tokens of `surprise_now - baseline`, in nats, keeping negative values. It measures how much worse the model now predicts the saved copy's preferred answer. It is a gap on that one answer, not a divergence between the two models. The composer passes it, and this class never computes it.
- `b` and `tau` are required config. Cosine lies in `[-1, 1]` and drift has no fixed range, so `b` sets the balance. No hidden normalization.
- A topic's drift starts at 0 on commit and keeps its last measured value between refreshes.
- Weights exist only over tasks with retained rows, as in TSR. The replay probability stays batch-level.

`routing_mode` selects the control:

| `routing_mode` | Weights | Question |
|---|---|---|
| `similarity_drift` | Formula above, the proposal | |
| `similarity` | `b = 0`, plain TSR routing | Does adding drift help? |
| `shuffled_drift` | Drift values permuted across topics at each `set_drift`, using this component's RNG | Is drift better than noise? |
| `uniform` | Equal weights | Does routing help at all? |

Overrides from `TSRPipeline`: `route_replay` (the formula above), `dependencies` (skips removed copies), `retrieve_replay` and `finish_task` (the standalone path materializes answers). Additions: `set_drift`, `next_active_task`, `write_answers`, and `commit_topics(replay)`, which restricts routing to retained topics and applies the copy rule below. The standalone `after_memory_commit()` calls `commit_topics`. Inherited unchanged: `probe_task`, `draw_branch`, `snapshot_task`. Require `kd_weight = 0`.

The standalone lifecycle supports `routing_mode` `similarity` and `uniform`, using `memory_per_task` random rows with written answers. Drift needs Part 3, so drift modes run through the [composer](/design/generation-replay.md).

## Writing answers

`write_answers(records, owner)` runs at task end, after selection. Only newly kept current-task rows are written. Older rows keep their answer unchanged, including after reselection.

For each row:

1. Generate the answer from the prompt alone, through `predict` on the learner. The learner's weights equal the saved copy at this point; the row records the saved-copy reference.
2. Keep tokens up to and including the first EOS. If there is no EOS within `writer_max_new_tokens`, keep them all and set `truncated`.
3. Reject an empty answer or one containing reasoning tags: `practice` stays `None` and the row logs `writer_rejected`. It stays in memory, but is never replayed. Report the rejection count per topic.
4. Score the answer with `ScoreSpec("token_loss", "none")` to get one baseline surprise per answer token, rounded to 4 decimal places before any use, so resume cannot differ.
5. Measure `gold_similarity`, the [token similarity](/design/repair-stop.md) to the gold response. This is how "saved copy is wrong" is reported.

The result is `Record.practice` ([schema](/design/utils.md)). Progress is saved in `policy["writing"]`, keyed by sample ID, with a checkpoint after each batch. A resumed task-end reuses completed rows.

The bytes count toward `ReplayStore`'s budget through its serialized size. An over-budget proposal fails commit and is never truncated. Estimate the size at preflight, and report it beside the removed copy's measured bytes. Whether deleting copies actually saves storage is an experimental result.

## Writer mode

| `writer_mode` | Target | Answer check and repair gains | Hinge with baselines | All-token replay |
|---|---|---|---|---|
| `greedy`, the proposal | Saved copy's deterministic answer | Yes | Yes | Yes |
| `gold`, the control | Stored response, with no baseline and no drift | No | No | Yes |

Greedy is required for the proposal because Check 1 asks whether the model can still write the stored answer. The saved copy reproduces a greedy answer, so a pass or fail means something. A sampled answer would fail even on the saved copy. The cost is that replay practises one deterministic answer per question and carries no claim about matching the saved copy's full behaviour.

## Saved copies

- Snapshot exactly once per task through the inherited `snapshot_task`. Writing depends on no reload.
- `retain_copies=False` is allowed because rows and baselines are written once and reselection never adds rows. After commit, `commit_topics` sets that topic's teacher reference to `None`, so `dependencies` stops listing it. Files go through the existing reference-aware pruning, so a retained fallback checkpoint keeps what it needs ([rule](/design/model-weights.md)). This drops eras, so label it as an adaptation.
- `retain_copies=True` keeps every copy under existing budget rules. Use it for the comparison arm and for any later diagnostic that loads a copy.
- Signatures always stay, because routing needs them.

## Boundaries

- This class selects no rows and no tokens. It returns topic weights and answers.
- Prompts come from retained or permitted training records only. Generation that fills `max_length` is prevented by `reserve_response` in [preprocessing](/design/datasets.md). `writer_max_new_tokens` must not exceed it.
- Persist drift, shuffled drift, weights, and writing progress in `policy`, and pending branch in `progress`. Do not recompute weights unless signatures, membership, or drift changed.

## Config

`PastSelfConfig(TSRConfig)` adds these fields, all required: `routing_mode`, `drift_weight` (`b`), `writer_mode`, `writer_max_new_tokens`, `retain_copies`.
