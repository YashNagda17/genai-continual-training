# Generation replay experiments

Owns `src/generation_replay_experiment.py`: the first test, arms, fairness rules, measures, and drop rules for the [generation replay recipe](/design/generation-replay.md). It contains no algorithm logic. Numeric budgets and acceptance margins are protocol inputs fixed before a scientific run, except the 1-point margin stated in [Idea_Refined.md](/Idea_Refined.md).

```sh
python -m src.generation_replay_experiment --protocol gate|smoke|pilot|scientific --arm <name> --output <dir>
```

As in [`csrel_experiment`](/design/main.md), each arm writes its own attempt directory and never deletes an existing attempt. `gate`, `smoke`, and `pilot` results are development evidence, and only `scientific` produces reported results.

## Reuse

Import `LimitedViews`, `performance_report`, `timing_report`, `storage_report`, `summarize_attempt`, `write_comparison`, and `attempt_for` from `src/csrel_experiment.py`. Do not copy them. Add only what differs:

- A TRACE loader using the existing snapshot at `src/datasets/data/trace/`, `TraceSource`, and `bind_instruction_dataset`. `load_real_dataset` builds SmolLM2 only, so this loader also takes `--model smollm2|qwen3`, using the existing adapters. `download_datasets.sh` does not change.
- `make_pipeline(arm, ...)` for the arm table below, and a `run_spec` with the recipe's identity fields.
- `smoke` follows `backend-smoke`: a tiny local JSONL with long answers and SmolLM2. CLINC is used only to check `group_source="class"`, since its one-phrase answers do not suit Part 3.

## Topics

Current data only. Measured on the local TRACE-5000 training split, 2026-11-05, in whitespace words before tokenization or truncation:

| Task | Answer p50 / p90 / p99 | Prompt p50 / p90 | Answer share of a median row | Eligible |
|---|---:|---:|---:|---|
| MeetingBank | 52 / 109 / 166 | 951 / 7,100 | 5% | Yes |
| Py150 | 6 / 13 / 24 | 117 / 975 | 5% | Yes, weakest |
| ScienceQA | 144 / 305 / 582 | 38 / 86 | 79% | Yes |
| 20Minuten | 37 / 55 / 81 | 317 / 555 | 10% | Yes, German |
| C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds | 1 / 1 / 1 | | | No, one-word answers |

Consequences:

- **The four-topic stream is forced.** `Idea_Refined.md` asks for at least 4 topics and rules out one-word answers, and exactly four remain. Use them in benchmark order: MeetingBank, Py150, ScienceQA, 20Minuten.
- **Cuts only matter on ScienceQA.** A cut shortens the answer, but the model still reads the prompt. At most about 5–10% of a median row's tokens can be cut on the other three topics. Skipping whole rows remains available everywhere. Report saved tokens per topic instead of one pooled figure.
- **MeetingBank prompts exceed any sequence limit we can afford.** Set `max_length` and `reserve_response` per protocol, and report how many prompts `prompt_left` truncates.
- **Small models may write poorly here.** A 135M model on German news or long science answers may never produce an answer worth preserving. Check this first in the gate.

Use SmolLM2-135M for development and Qwen3-0.6B for final runs, each with its own recorded settings.

## First test (`--protocol gate`)

`Idea_Refined.md` requires this before anything else is built. It needs only build step M0.

1. Train on one long-answer topic, and write answers for a random memory of kept rows.
2. Train on a second topic with no replay, to cause forgetting. Measure Check 1, `d_k`, and repair gains on the kept rows, on the learner, with no further updates.
3. Report: the distribution of `sim(generated, answer)` and whether it separates rows; forgetting scores; `gold_similarity`; and, among tokens with similar `d_k`, the share of gains above the noise floor.

| Result | Decision |
|---|---|
| Rows separate and some gains stand out above noise | Continue to M1 with `token_policy="repair"` |
| Almost all gains are zero or noise | Drop repair tests. Keep the answer check, stop rule, and cuts with `top_forgetting` |
| Rows do not separate | Stop using the answer check. Try `answer_check=False`, or revisit Part 3 |
| Saved copies write poor text | Change the model, the topic, or `max_new_tokens` before reading anything else from this run |

Fix "clearly separate" and "stand out" as numbers in the gate configuration before running it. The gate proves no retention or cost claim, since nothing is trained on the result.

## Arms

All Part 3 arms use the same kept questions through `FixedMemorySelection`, plus the same seeds, task order, and evaluation. Only the named setting changes. Defaults for the full method: `AnswerSelectionPipeline`, `similarity_drift`, `greedy`, `token_policy="repair"`, with check, stopping, and cuts on, and `no_replay_action="current"`.

| Arm | Change from the full method | Question |
|---|---|---|
| `full` | None | The proposal |
| `shuffled_gains` | `token_policy="shuffled_gain"` | Do gains find the words that matter? |
| `top_forgetting` | `token_policy="top_forgetting"` | Do gains beat forgetting scores? |
| `random_tokens` | `token_policy="random"` | Does token selection matter at all? |
| `no_stop` | `stopping=False`, `cuts=False` | Does stopping save training? |
| `stop_no_cuts` | `cuts=False` | How much do cuts add? |
| `no_check` | `answer_check=False` | What does the check add, and what does it miss? |
| `skip_updates` | `no_replay_action="skip"` | How much time is saved directly? |
| `answer_all_tokens` | `all_tokens`, everything off | Is token work better than plain replay on written answers? |
| `usual_replay` | `all_tokens`, `writer_mode="gold"`, `routing_mode="similarity"`, everything off | The main baseline |
| `similarity_only` | `routing_mode="similarity"` | Does drift help routing? |
| `shuffled_drift` | `routing_mode="shuffled_drift"` | Is drift better than noise? |
| `repair_only` | Random memory, `uniform` routing | Part 3 alone |
| `new_only`, `cumulative` | `MemoryBaselinePipeline` | Bounds: no old data and all old data |
| `old_hybrid` | `CombinedPipeline` | The previous design |

Part 1 arms (`gap`, `weak_only`, `random_in_group`, loss-based CSReL, random) compare selection directly. They share one fixed replay method and differ only in kept rows. Their selection cost is reported separately.

## Fair comparison

Compare twice, for every arm against `usual_replay` and `full`:

- **Equal updates:** the same `max_attempts` per task. Use `exhaustion="repeat_admitted"` with enough `passes` to cover the extra current updates that recovered replay releases. Otherwise an arm that gives updates back runs out of current data, and its task ends early. Preflight must reject a budget that cannot cover this.
- **Equal time:** first measure the reference arm's total task seconds, including all extra work below. Then run each arm with `spec.identity["max_task_seconds"]`. It stops at the next safe boundary after reaching the limit. Time-limited runs are hardware-dependent, so run them on one machine and report it.

## Measures

**Quality, on held-out data against reference answers.** Matching the saved copy helps training but does not prove correctness. Primary metric: ROUGE-L for MeetingBank and 20Minuten, ROUGE-L and leading-choice accuracy for ScienceQA, and token similarity for Py150. ROUGE-L is the native `numerics.rouge_l`, whitespace tokens with no stemming, so label it unofficial. Do not average across tasks until scales and weights are fixed ([rule](/design/utils.md)). Report per-topic retention and forgetting from the task matrix on a 0–100 scale, plus answer variety on held-out output.

**Cost, everything counted.** Report every [cost category](/design/utils.md), and sum them for total cost: selection samples and embeddings, weak and strong fitting, writing, checks, surprise forwards, repair rollouts, replicas, learner training, checkpoints, and evaluation separately. Also report generations, input tokens, peak memory, and storage (answers, baselines, kept copies, checkpoints). Report quality against cost as a frontier, not a single ratio.

**Mechanism.** `recovered_free_running`, `slip_rate`, `false_skip`, share of gains above the floor, noise floor, `gold_similarity`, skipped rows and updates, and tokens cut.

Use at least three seeds for scientific claims. Failed and negative runs stay in the report ([rule](/design/utils.md)). Fix settings before running, repeat some repair tests, and report the noise.

## Drop rules

From `Idea_Refined.md`, tied to what decides each one. "Just as well" means within a margin fixed in the protocol before running.

| Observation | Measured by | Decision |
|---|---|---|
| Random tokens work just as well | `random_tokens` vs `full`, retained quality | Drop the method |
| Shuffled gains work just as well, or most gains are noise | `shuffled_gains` vs `full`; share of gains above the floor | Drop repair tests |
| Old-topic quality falls more than 1 point with no clear saving | `no_stop` vs `full`, quality and total cost | Drop stopping and cuts |
| The answer check skips rows with forgotten facts | `false_skip`; `no_check` vs `full` | Drop the answer check |
| Fixed tokens do not bring back answers from the original prompt | `recovered_free_running` | Drop the main claim, and report it |
| Extra work costs more than it saves | Total cost vs `usual_replay` at matched quality | Report it, and reduce tests or drop them |
