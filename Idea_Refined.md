# Refined Idea: Generation-Aware Repair and Replay

**Project title:** Cost-Efficient Incremental Fine-Tuning through Generation-Aware Repair and Replay

**Status:** proposal, 2026-11-05. Nothing has been built or run yet. All numbers in examples are made up to show how things work. Update [Idea.md](/Idea.md) and the [design files](/design/README.md) before coding. **Course:** CS787 Generative AI. We use only LLMs under the [700M limit](/design/README.md), and no separate VAE, GAN, or diffusion model.

## The idea in simple words

An AI model keeps learning new topics, but learning something new can damage what it already knew. The usual fix is to retrain it on complete old answers. That costs a lot, because most of those words were never forgotten.

Think of a student revising for exams:

- **Usual method:** reread every old notebook, cover to cover.
- **Our method:** find the few facts you forgot that make your answers go wrong, revise only those, and stop once you can answer correctly again.

**Our main claim to test:** fixing a few important words can bring back a whole answer, so the model needs less old data and less training.

## The problem, with an example

The model learns three topics in order: meeting summaries, science questions, then Python code.

> **Question:** Summarise this meeting about the city budget.
> **Before learning Python:** "The council approved the budget on Monday."
> **After learning Python:** "The council rejected the plan last week."

One wrong word, "rejected", sent the rest of the sentence the wrong way. Usual replay would retrain all seven words. We want to find and fix "approved", then check whether the full answer comes back.

## The questions we want to answer

How much real old data does a small model need if its saved past versions can write practice answers? Which words are worth practising? Can repairing the important words keep old skills at a lower total cost than usual replay?

## Project pitch

> We investigate whether the cost of incremental fine-tuning can be reduced by replaying the generation decisions that matter. Our method identifies forgotten tokens through their effect on generated continuations, selectively rehearses them, and stops replay when earlier generation capabilities recover. We test whether this preserves old-task generation and new-task learning with less historical data and lower end-to-end training cost.

## The three parts

| Part | Simple name | Question it answers | Based on |
|---|---|---|---|
| 1 | Pick the best old questions | Which old questions do we keep? | CSReL + ADG |
| 2 | Practise using your past self | Which old topic do we practise, and with what answers? | TSR + Marek et al. |
| 3 | Repair and stop | Which words do we practise, and for how long? | Rho-1, ssToken, CFT, RGD-R, MSSR |

We drop OASIS and aL-SAR from the main design. Their code stays, so the old hybrid can still run as a comparison.

## Part 1: Pick the best old questions

We can keep only a few old questions, so we want the ones that teach the most. We train two helper models. The **weak** model has seen only the questions picked so far, and the **strong** model has seen the whole topic. Each answers a question several times, and we check how much its answers disagree.

> **Question:** "Summarise this meeting."
> **Weak model:** "They discussed roads." / "The budget was rejected." / "A new mayor was chosen."
> **Strong model:** "The council approved the budget." / "The budget was approved."

The weak model is confused, but the strong model is clear. So this question has something real to teach.

| Weak model's answers | Strong model's answers | Keep? | Example |
|---|---|---|---|
| Messy | Clear | **Yes** | "Summarise this meeting" |
| Messy | Messy | No, the question is naturally open | "Write a poem about rain" |
| Clear | Clear | No, it's already learned | "What is 2 + 2?" |

We also keep questions from different groups, such as budget, school, and election meetings. In our pilot, the old method kept only 3 of 15 banking question types.

**For the code:** `score = disagreement(weak model) - disagreement(strong model)`

**Papers behind this part:**

| Paper | What it does | What we use | What we change |
|---|---|---|---|
| [CSReL](/Papers/CSReL-summary.md) (ICLR 2025) | Keeps examples a model could learn but hasn't yet: subset model's loss minus full model's loss. Losses 3.0 and 0.4 score 2.6, so keep; 8.0 and 7.8 score 0.2, since it's hard for both. Split CIFAR-100: 17.48% vs 13.82% uniform | The weak-minus-strong subtraction and repeated picking rounds | Compare answer disagreement, not loss on the correct answer |
| [ADG](https://arxiv.org/abs/2604.10448) (arXiv 2026) | Samples 5 answers per question, and keeps questions with the most varied answers, within groups. LLaMA3-8B on Alpaca-GPT4: 50.85 vs 46.06 random | The disagreement score and question groups | Subtract the strong model's score, and use it across a stream of topics |

**How they link:** ADG alone could favour a naturally open question, like the rain poem. CSReL's subtraction cancels that out, because the strong model is messy there too.

## Part 2: Practise using your past self

After each topic, we save a copy of the model. That copy writes the practice answers for its topic. These are normal training examples, so we need no extra loss.

> **Question kept from the meeting topic:** "Summarise this meeting."
> **Saved meeting copy writes:** "The council approved the budget on Monday."
> **Current model** practises this answer while learning Python.

**Which old topic gets practised?** We look at two things:

- **Similar:** is this old topic related to the new one? Then it can help learning.
- **Forgotten (drift):** how far has the model moved from its saved copy on this topic?

| Old topic, while learning Python | Similar? | Forgotten? | Using similarity only | Using both |
|---|---|---|---:|---:|
| Science | Quite similar | A little | 90% | 55% |
| Meetings | Not similar | A lot | 10% | 45% |

Similarity alone would nearly ignore meetings, even though that skill is being lost.

**For the code:** `topic weight = softmax((similarity + b * drift) / tau)`

**Papers behind this part:**

| Paper | What it does | What we use | What we change |
|---|---|---|---|
| [TSR](/Papers/TSR-summary.md) (arXiv 2026) | Gives each task a gradient fingerprint, favours similar old tasks, and saves one model per task as its teacher. TRACE-8 on a 0.5B model: 0.475 vs 0.451 plain replay | Fingerprints, topic choice, and saved copies | Add forgetting to topic choice; saved copies write answers instead of matching token predictions |
| [Marek et al.](https://arxiv.org/abs/2605.26097) (arXiv 2026) | Shows that training on a frozen model's sampled text estimates matching its behaviour (KL) | The reason practising on samples works | Use one saved copy per topic, chosen by Part 2 |
| [OPR](https://arxiv.org/abs/2605.29495) (arXiv 2026) | The latest model answers old questions, the best answers are kept, and they're replayed as normal training. 1% budget on a 7B model: 65.45 vs 62.57 plain replay | Generated answers as normal training data | The topic's saved copy writes the answer, not the latest model |
| [SERS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html) (NeurIPS 2025) | Generates old-style questions, then uses task similarity to set regularization strength | Treat this as close prior work | Use similarity and forgetting to choose which topic to practise |

**How they link:** TSR tells us which saved copy to use, Marek et al. explain why practising its text can preserve its behaviour, and OPR and SERS report that generated replay can reduce forgetting. Adding forgetting to TSR's topic choice is our addition.

## Part 3: Repair and stop

| Problem with usual replay | Our fix |
|---|---|
| One wrong word can derail an answer, and replay can't see this | Test each suspect word by repairing it |
| It keeps practising words that have already recovered | Stop once a word recovers |
| It needs the saved copy to run on every batch | Store the saved copy's scores once |
| It spends compute on sentences that need no work | Skip recovered sentences and cut unneeded ending words |

**Words we use:**

- **Token:** a word or piece of a word.
- **Surprise:** how unexpected a token is to the model. Low chance means high surprise.
- **Baseline:** the saved copy's surprise for that token, stored once.
- **Forgetting score:** current surprise minus baseline. High means the token looks forgotten.
- **Repair gain:** how much the rest of the answer improves when this token is fixed.

**The three checks:**

| Check | Question | What happens |
|---|---|---|
| **1. Answer check** | Can the model still write the old answer by itself? | If yes, skip this sentence |
| **2. Repair test** | Which tokens bring the answer back when repaired? | Practise those, with extra weight on turning points |
| **3. Stop rule** | Has this token recovered yet? | If yes, stop training it |

### Full example

The saved copy wrote: **"The council approved the budget on Monday."**

**Step 1: Save the baselines.** We do this once: council 0.22, approved 0.36, Monday 0.51. **Step 2: Answer check.** The model now writes "The council rejected the plan last week." That's far from the old answer, so this sentence needs practice.

**Step 3: Repair test.** For each suspect token, the model writes the rest of the answer twice, without the repair and with it.

| Token | Forgetting score | Without repair | With repair | Repair gain | Final score |
|---|---:|---|---|---:|---:|
| **approved** | 1.03 | "rejected the plan last week" | "approved the budget on Monday" | 0.75 | **2.58** |
| **Monday** | 1.79 | "on Tuesday" | "on Monday" | 0, nothing comes after it | 1.79 |
| **council** | 1.67 | "The city rejected..." | "The council rejected..." | 0, the rest doesn't change | 1.67 |

`final score = forgetting score x (1 + 2 x repair gain)`

"approved" had the lowest forgetting score, but it ranks first because it steers the whole answer. If we can practise two tokens, we pick "approved" and "Monday", and "approved" trains with 2.5 times more weight.

**Step 4: Practise until recovered.** We allow 0.10 extra surprise, so the targets are 0.46 for "approved" and 0.61 for "Monday".

| Round | "approved" surprise | "Monday" surprise | What trains | Tokens used |
|---|---:|---:|---|---:|
| 1 | 1.39 | 2.30 | Both | 7 |
| 2 | 0.80 | 0.50 | "approved" only, since "Monday" recovered | 7 |
| 3 | 0.40 | Done | Nothing, since "approved" recovered | 3 |

After round 2, the sentence is cut after "approved", because later words can't affect earlier ones.

**Step 5: Check the whole answer again.** The model now writes the old answer by itself, so this sentence stops using compute. If new training breaks it later, the next check brings it back. **The real test:** does fixing tokens bring back the answer *from the original question*, without giving the model any correct words? If not, token repair isn't enough, and we report that.

### The steps, for the code

```text
WHEN A TOPIC ENDS, for each kept question:
  saved copy writes an answer; store the answer and each token's baseline
  the saved copy can be deleted if nothing else needs it

EVERY R REPLAY TURNS (a turn counts even if nothing needs replay):
  Check 1: model writes an answer; if it matches the stored one, skip the sentence
  Check 2: run the repair test on the top c forgotten tokens
           (reuse gains already measured; ignore gains within the noise level)
           final score = forgetting score x (1 + beta x repair gain)
           pick the top tokens, and cut the sentence after the last one
  Send each topic's average forgetting score to Part 2

AT EACH REPLAY UPDATE:
  Check 3: token loss = weight x max(0, surprise - baseline - allowance)
  If this topic has nothing to practise, try the next topic
  If no topic does, use this update for the current topic
```

### Papers behind this part

| Paper | What it does | What we use | What we change |
|---|---|---|---|
| [Rho-1](https://arxiv.org/abs/2404.07965) (NeurIPS 2024) | Trains on tokens whose loss is well above a reference model's | Comparing a token's loss with a reference | The reference is the saved copy, used to spot forgetting |
| [ssToken](https://arxiv.org/abs/2510.18250) (arXiv 2025) | Scores tokens against the model's own history, plus attention. 7B model: 60.48 vs 59.73 full data | Using a past self as the reference | Look for tokens that got worse, across topics |
| [CFT](https://arxiv.org/abs/2510.10974) (arXiv 2025), our closest match | Swaps a token, generates the rest, and trains on tokens where every swap breaks the final answer. 7B model: 59.2 vs 52.8 normal fine-tuning | Testing a token by generating what follows | Repair a forgotten token, compare with the past self, and need no checkable final answer |
| [Critical Tokens Matter](https://arxiv.org/abs/2411.19943) (arXiv 2024) | Re-generates from each token to find where a wrong answer went wrong | The idea that one token can decide an outcome | Test only a few forgotten tokens, with two runs each |
| [RGD-R](https://arxiv.org/abs/2411.11932) (ACL 2025 Findings) | Finds that a little correct reasoning can restore a "forgetting" model, and shares replay across tasks using that | The reason a small repair might bring back an answer | Work on single tokens, not whole tasks |
| [MSSR](https://arxiv.org/abs/2603.09892) (arXiv 2026) | Replays when memory strength falls, with a loss-triggered version as a baseline | Replay only when needed | Decide per sentence and token, with an answer check |
| [Noise study](https://arxiv.org/abs/2606.15621) (arXiv 2026) | Repeat runs of similar token tests disagree on 9–23% of estimates | Fix settings first, repeat some tests, and report the noise | Apply this to our repair test |

**How they link:** Rho-1 and ssToken give the forgetting score, CFT and Critical Tokens Matter give the repair test, RGD-R gives the reason it might work, and MSSR gives the stopping idea. None of the papers we read combines these to repair forgotten tokens across topics and stop when the answer recovers, though we've only run a few searches.

**Also useful:** [TIP](https://arxiv.org/abs/2604.14084) reports up to 47% lower peak memory from token selection, [TOSS](https://arxiv.org/abs/2603.01185) scores tokens with two reference models, and [Low-Perplexity Token Learning](https://arxiv.org/abs/2501.14315) skips risky new-topic tokens, which could be an optional extra.

## Where the savings come from

Hiding tokens from the loss saves little, because the model still reads the whole sentence. Real savings come from **skipping whole batches** when nothing needs practice, and from **shorter sentences**, such as 3 tokens instead of 7. Updates we no longer need for replay go to the current topic, so the model gets more new learning within the same budget. A separate test skips those updates, to measure time saved directly.

**Extra work must be counted too:** answer checks, repair tests, picking questions, writing answers, any saved-copy runs, and checkpoints. We keep it small by reusing repair gains, testing few tokens, and limiting how much each test writes.

**Check 1 makes Check 2 cheaper.** With 60 sentences, 5 suspect tokens each, and 2 test runs per token:

| Version | Generations each time |
|---|---:|
| Repair test on every sentence | 600 |
| Answer check first, if 20 of the 60 fail | 60 + (20 x 5 x 2) = **260** |

**Storage may fall too.** Baselines for 60 sentences of 60 tokens take 60 x 60 x 2 = 7,200 bytes, so saved copies might be deleted. Check the [storage rules](/design/model-weights.md) first.

## How we test it

We use at least 4 TRACE topics, led by MeetingBank and ScienceQA, with SmolLM2-135M for development and Qwen3-0.6B for final runs. One-word answers don't suit this method.

| Question | Compare | We need to see |
|---|---|---|
| Do repair gains find the words that matter? | Repair gains vs forgetting scores vs shuffled gains | Generated answers recover better |
| Does stopping save training? | With vs without stopping and cuts | Lower total cost at similar quality |
| Does the whole method meet the project's goal? | Our method vs usual replay vs keeping all data | Better quality for the cost, counting all extra work |

**Other versions to compare:** random tokens; highest forgetting scores with stopping; our method with no answer check; our method skipping unneeded updates; and CFT-style selection on ScienceQA, the closest paper's approach. All use the same kept questions and seeds.

**Keep it fair:** compare versions twice, once with equal updates and once with equal training time. **What we measure:** old-topic and new-topic quality, forgetting, answer variety, training time, memory, storage, and generations used. **Judging correctness:** use held-out questions and reference answers. Matching the saved copy helps training, but doesn't prove an answer is right.

**We drop a part if:**

- Random tokens work just as well: drop the method.
- Shuffled gains work just as well, or most gains are noise: drop repair tests.
- Old-topic quality falls over 1 point with no clear saving: drop stopping and cuts.
- The answer check skips sentences with forgotten facts: drop the answer check.
- Fixed tokens don't bring back answers from the original question: drop the main claim.

**Noise:** one study found repeat runs of similar token tests disagree on 9–23% of estimates. So we fix settings before running, repeat some tests to measure the noise, and report it.

## What could go wrong

| Risk | Example | What we do |
|---|---|---|
| Saved copy is wrong | It wrote "rejected", so the model practises a mistake | Compare with real stored answers |
| Answer check misses a wrong fact | "Monday" vs "Tuesday" is only one word | Use a strict check, and test it |
| Single facts get less credit | "Monday" has nothing after it, so its gain is 0 | Forgetting score still ranks it |
| Harmless rewording looks forgotten | "city" instead of "council" | Zero gain lowers its rank, though this isn't proof |
| Target is too hard | A token's baseline is so low it never recovers | Tune the allowance on validation data |
| Token slips between checks | A skipped sentence breaks after the last check | Tune `R`, and report how often this happens |
| Extra work costs more than it saves | Repair tests outweigh skipped replay | Count everything, and limit tests |
| Small models write poorly | A 135M model writes fluent wrong text | Test generation quality first |
| Paper results may not carry over | Most used models of 2B or more | Treat them as motivation only |

## Scores

These are our own estimates, not measured values: **GenAI fit** 8–8.5/10; **novelty** Part 1 6/10, Part 2 5/10, Part 3 6–7/10.

**Novelty scale:** 1–2 already published; 3–4 same method in a new setting; 5–6 known parts in a new combination that changes behaviour; 7–8 new method with no close match after a thorough search; 9–10 a new kind of problem. We ran only a few searches, so say "novelty unverified" until the [challenger stage](/skills/challenger.md) has run.

| How Part 3 differs | CFT | Critical Tokens Matter | RGD-R | **Our Part 3** |
|---|---|---|---|---|
| Compares | One model | One model | One model | **Model and its past self** |
| Action | Breaks a token | Resamples from a token | Gives part of the reasoning | **Repairs a forgotten token** |
| Judged by | Correct final answer | Correct final answer | Correct answer and reasoning | **Match with the saved answer** |
| Stops when recovered? | No | No | No | **Yes** |

## Related papers

Most were read only in part. Candidate ideas are logged in [optimizations.md](/optimizations.md).

- **Kept methods and picking by sampled answers:** [CSReL](/Papers/CSReL-summary.md), [TSR](/Papers/TSR-summary.md), [ADG](https://arxiv.org/abs/2604.10448).
- **Replay using generated text:** [Marek et al.](https://arxiv.org/abs/2605.26097), [OPR](https://arxiv.org/abs/2605.29495), [SERS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html).
- **Choosing tokens to train on:** [Rho-1](https://arxiv.org/abs/2404.07965), [ssToken](https://arxiv.org/abs/2510.18250), [TOSS](https://arxiv.org/abs/2603.01185), [TIP](https://arxiv.org/abs/2604.14084).
- **Testing tokens by generating (closest to us):** [CFT](https://arxiv.org/abs/2510.10974), [Critical Tokens Matter](https://arxiv.org/abs/2411.19943).
- **Other:** [RGD-R](https://arxiv.org/abs/2411.11932) (a little correct text can restore a "forgetting" model), [MSSR](https://arxiv.org/abs/2603.09892) (loss-triggered replay), [replay noise study](https://arxiv.org/abs/2606.15621), and [Low-Perplexity Token Learning](https://arxiv.org/abs/2501.14315) (optional extra: skip risky new-topic tokens).

## Changes needed before coding

- **Design files:** [Idea.md](/Idea.md), [README](/design/README.md), [datasets](/design/datasets.md), [utils](/design/utils.md), [combined-pipeline](/design/combined-pipeline.md), [models](/design/models.md), [model-weights](/design/model-weights.md), [main](/design/main.md), [tests](/design/tests.md).
- **Code:** `src/combined_pipeline.py`, `_losses` and `predict` in `src/models/model_causal_lm.py`, `Objective` in `src/utils/contracts.py`, `src/utils/state.py`, `src/datasets/dataset_instruction.py`, `src/csrel_experiment.py`, and `src/tests/`.

## First test, before building everything

On one trained model and one TRACE topic, check two things:

1. **Answer check:** can we clearly separate sentences that still work from those that don't?
2. **Repair gains:** among tokens with similar forgetting scores, do some gains stand out above the noise?

If almost all gains are zero or noise, drop repair tests and keep the answer check, stop rule, and cuts.
