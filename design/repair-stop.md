# RepairStopPipeline

Owns `src/pipelines/repair_stop.py`: `RepairStopConfig` and `RepairStopPipeline(PipelineBase)`. This is Part 3 of [Idea_Refined.md](/Idea_Refined.md), "Repair and stop". It decides **which rows and tokens to practise, and for how long**. It does not choose topics or step the optimizer. The [recipe guardrails](/design/generation-replay.md) apply.

## Our own mechanism

Repair and Stop is a project mechanism, not a paper method or an adaptation of one. Papers supplied starting ideas, credited in the "Papers behind this part" table of `Idea_Refined.md`. They are inspiration only: no definition here has to match a paper, and there is no reproduction target.

**What defines it.** These five steps together are the mechanism. Changing one of them makes it a different method, so use a new name or arm:

1. **Own past as reference.** A token counts as forgotten by comparing the model with its own saved copy on that copy's answer, not with another model or a correct-answer check.
2. **Answer check first.** A sentence gets token work only if the model can no longer write the stored answer from the question alone.
3. **Repair test.** A forgotten token ranks higher if repairing it brings back the rest of the answer.
4. **Practise until recovered.** Each token trains only until it returns to its stored baseline plus an allowance, and then stops using compute.
5. **Check again.** Recovery is judged by whether the whole answer comes back from the original question, and a sentence that breaks later returns to practice.

**What we are free to change.** The similarity measure, gain formula, weight form, allowance, refresh schedule, and limits are our design choices, and evidence decides them. To change one, update this contract and record the reason. Larger new algorithmic ideas also go in [optimizations.md](/optimizations.md).

**How it is judged.** Only against our own [controls and drop rules](/design/generation-replay-experiments.md), not against published numbers. Novelty is unverified.

It builds on TSR's saved copies and needs no extra model role.

## Definitions

For a kept row with stored answer `a[0..n-1]` and baselines `b[0..n-1]` ([writer](/design/past-self-replay.md)):

| Name | Definition |
|---|---|
| `s_k` | Current surprise: `-log p(a[k] | prompt, a[:k])` from `ScoreSpec("token_loss", "none")` |
| `d_k` | Forgetting score: `s_k - b_k` |
| `m` | `allowance`, the extra surprise tolerated |
| `sim(x, y)` | `1 - levenshtein(x, y) / max(|x|, |y|)` on token IDs, 1 when both are empty, from `numerics.token_similarity` |
| `H` | `repair_horizon`, the most tokens a rollout writes |
| `g_k` | Repair gain, below |
| `S_k` | Final score: `d_k * (1 + beta * g_k)` |
| `w_k` | Training weight: `1 + beta * g_k` |

Check against the worked example with `beta = 2`: "approved" has `d = 1.03`, `g = 0.75`, so `S = 2.575` and `w = 2.5`.

## Refresh

`refresh(replay, owner)` runs when the [composer](/design/generation-replay.md) says it is due, on the learner, over **all** retained rows with an answer. Each stage is resumable from `policy["refresh"]`, with a checkpoint boundary after each batch.

```text
stage check (if answer_check):
  generate greedily from the prompt alone, up to len(a) + 1 tokens
  passed = sim(generated, a) >= match_threshold

stage surprise (always, for every row):
  compute s_k and d_k by teacher forcing on prompt + a
  add d_k to its topic's mean drift

stage repair (failing rows, token_policy needs gains):
  suspects = top c tokens by d_k, keeping only d_k > m
  for each suspect, highest d_k first, until max_rollouts:
    reuse a cached gain if it is no older than gain_max_age refreshes
    if a[k+1:] holds nothing but EOS: g_k = 0, no rollouts
    otherwise run the repair test

stage noise (replica_fraction of tests):
  repeat tests with sampled decoding and two seeds; update the noise floor

stage finalize (failing rows):
  apply the floor, rank by S_k, pick the top `picked` tokens
  store the row's positions, weights, and cut
  passed rows have no selected tokens
```

**Repair test.** Greedily write up to `H` tokens twice, comparing with `target = a[k+1 : k+1+H]`:

- Without repair, from `prompt + a[:k]`: the output starts at position `k`, so drop its first token to get `y0`.
- With repair, from `prompt + a[:k+1]`: the output is `y1`.

`raw_gain = sim(y1, target) - sim(y0, target)`. Then `g_k = max(0, raw_gain)`, set to 0 if it is at or below the noise floor. Keep `raw_gain` in the log, including negative values. Clamping stops a negative gain from flipping the sign of `S_k`.

**Cost per refresh.** `M` checks, `M` teacher-forced forwards, up to `F * c * 2` rollouts for `F` failing rows, plus replicas. With `M=60`, `F=20`, `c=5`: 60 + 200 = 260 generations, which matches the figure in `Idea_Refined.md`. Rollouts re-read the prompt each time, so on a long-prompt topic such as MeetingBank each one costs close to a full forward. Report input tokens as well as generation counts. Reusing the prompt's key-value cache is a behaviour-preserving optimization, deferred until the first test shows that gains are worth having.

**Noise floor.** Greedy decoding gives identical repeats, so repeating it reports zero noise. For a `replica_fraction` of tests, run the with-and-without pair at `noise_temperature` under two seeds, and take `floor = quantile(|g_seed1 - g_seed2|, noise_quantile)` over pairs measured so far in this task. Below `noise_min_pairs` pairs, there is no floor yet, so all gains count as 0. A sampled replica measures how fragile a gain is to small decoding changes, not sampling error in the greedy estimate. This is a project choice, and it is reported beside the gains.

## Practice

`retrieve(task, replay, count)` samples uniformly, without replacement, from that topic's rows with active tokens, using this component's RNG. A short topic gives a smaller batch. Rows are materialized by `practice_record(record, cut)`, so the target is `a[:cut]` and token counts shrink with it. The tokens consumed, and the padded batch width, fall with no extra logic.

`token_targets(batch)` returns aligned baselines and weights, with weight 0 on unselected tokens. `objective(reduction)` gives `token_mode` and allowance. The [model executor](/design/models.md) computes:

`loss = sum_k w_k * max(0, s_k - b_k - m) / (tokens with a positive hinge)`

The token mean replaces `Objective.reduction`. Because the denominator counts only active tokens, the remaining tokens are not diluted as others recover.

**Live stop.** `observe(report, records)` reads `token_surprise` from the same training forward, before the update:

- A selected token with `s_k - b_k <= m` is inactive, and it costs no extra forward to know this.
- With `cuts`, `cut = last active position + 1`. Later tokens cannot affect earlier ones, so earlier surprises stay valid. A causal-model test must confirm this for each adapter.
- A row with no active tokens leaves the draw until the next refresh.
- The selected set is fixed between refreshes, while the hinge is live. A selected token before the cut that slips back starts training again by itself. A token cut off stays out until the next refresh.

This reproduces the worked example: 7, 7, then 3 tokens, with "Monday" stopping in round 2 and "approved" in round 3.

An all-inactive batch reports `skip_reason="no_active_tokens"` and takes no optimizer step, but its surprise still updates row state.

**Drift and reporting.** `drift()` returns the last measured mean `d` by topic. `active_tasks()` returns topics with active rows. `prune_rows(replay)` removes state for rows no longer retained and sets the new topic's drift to 0.

## State

`policy` holds refresh progress, per-row state, noise pairs, and drift. It round-trips through JSON, so integer dictionary keys would come back as strings. **Use parallel lists**:

```python
rows[sample_id] = {"passed": bool, "similarity": float, "positions": [int],
                   "weights": [float], "active": [int], "cut": int,
                   "gains": [[position, raw_gain, refresh_index]]}
```

Stored baselines and cached gains are persistent state as allowed by the [score rule](/design/README.md). A baseline binds to its answer IDs and writer. A gain binds to its row, position, and refresh index, and it expires after `gain_max_age`, since the learner keeps changing. Set `gain_max_age=0` to disable reuse.

## Configuration of the controls

`token_policy` selects the tokens. `answer_check`, `stopping`, and `cuts` switch those features off independently.

| `token_policy` | Picked tokens | Weight | Rollouts |
|---|---|---|---|
| `repair` | Top `S_k` | `w_k` | Yes |
| `top_forgetting` | Top `d_k` | 1 | No |
| `shuffled_gain` | Top `S_k`, with gains permuted across a refresh's tested tokens | `w_k` from permuted gains | Yes |
| `random` | Uniform from all answer tokens, same count | 1 | No |
| `all_tokens` | Every answer token, with hinge and cuts off | 1 | No |

- `stopping=False` replaces the hinge with `w_k * s_k` on picked tokens (`token_mode="weighted"`), and rows remain active until the next refresh.
- `answer_check=False` treats every row as failing.
- `cuts=False` leaves whole answers.
- `random` and `shuffled_gain` use this component's RNG, saved with the policy state.
- `all_tokens` sends no token targets, so the batch uses `Objective(reduction)` and plain cross entropy. Every answer-bearing row stays active. `refresh` then runs only `surprise`, and only when routing needs drift.
- `all_tokens` with `writer_mode="gold"` and everything off is usual replay. Rows have no baselines and so no drift, so `routing_mode` must be `similarity` or `uniform`. `practice_record` returns a gold row unchanged.

## Reported per refresh

Rows checked, passed, and failing; tokens tested; share of gains at 0 and above the floor; noise floor and pair count; rollouts and input tokens; cuts and tokens saved; drift by topic. Track these across refreshes:

- `recovered_free_running`: rows that failed, trained, and then passed Check 1 **from the original prompt**. This is the real test of the main claim.
- `slip_rate`: rows that passed at one refresh and failed at the next.
- `false_skip`: rows that passed Check 1 while holding a token with `d_k > m`. It drives the answer-check [drop rule](/design/generation-replay-experiments.md).

## Config

`RepairStopConfig` fields, all required: `token_policy`, `answer_check`, `stopping`, `cuts`, `refresh_every` (`R`), `match_threshold`, `allowance` (`m`), `suspects` (`c`), `picked`, `beta`, `repair_horizon` (`H`), `gain_max_age`, `max_rollouts`, `replica_fraction`, `noise_quantile`, `noise_min_pairs`, `noise_temperature`. Fix these before a scientific run, and tune `allowance` and `match_threshold` on validation data only.
