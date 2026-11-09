# Refined Idea 5: The Final Experiment Plan

**Status:** final proposal, 2026-11-08. Nothing has been built or run. The model sizes are real. All other numbers in the examples are made up to show how things work.

**Title:** Cost-Efficient Incremental Fine-Tuning for Multilingual Summarization: a summarizer that reads its own writing to decide what to revise, what to revise with, when to stop, and how small to become.

**Rules:** LoRA only, with the base weights frozen. One model, under 700M parameters.

## Part 1: The story

A model writes good English news summaries. Hindi articles start arriving. We want the **same** model to write Hindi summaries too, **without forgetting English**, **cheaply**, and **without becoming slower** per summary.

Think of a student who writes good English summaries and starts a Hindi course:

| The student... | Experiment |
|---|---|
| Studies only Hindi | X |
| Checks whether they're just writing in the wrong alphabet | G5 |
| Re-reads random old English essays | A |
| Re-reads the essays their test scores say are slipping | B |
| Rewrites old essays and revises the ones that changed most | G1 |
| Revises from their own old essays, not the teacher's answers | G2 |
| Stops revising an essay once they can write it again | G3 |
| Changes only some chapters of their notes | C |
| Combines the best revision with the best chapters | D |
| Adds new Hindi pages only where Hindi clashes with English | E |
| Shortens the notes, then checks they can still write both languages | F |
| Writes "answer in Hindi" at the top of the page | G4 |

```text
Phase 0  Set up, and check the model can summarize
Phase 1  Teach Hindi, and see what is forgotten         X, G5
Phase 2  What to revise, and with what                 A, B, G1, G2, G3
Phase 3  Which parts of the model to change            C1, C2, D
Phase 4  Where new room goes                           E
Phase 5  How much room to keep                         F
Phase 6  Can a prompt prevent the slip                 G4
```

### The five principles we test

| | Principle | In plain words |
|---|---|---|
| **M** | Place before you enlarge, map before you sweep | Check you're training the right parts before training more parts. Look inside the model before paying for many runs |
| **N** | Revise what the model can no longer write | Spend revision on slipping examples, found and checked by the model's own writing |
| **K** | Count the whole bill | Fewer trainable parameters don't always mean cheaper. Count training, analysis, serving, and storage |
| **L** | Learn wide, remember narrow | A language may need much more room to learn than to remember |
| **P** | Check the language before the content | Part of "forgetting" may just be writing in the wrong language, which can be cheap to fix |

## Part 2: The setup

**Model: mT5-small**, an encoder-decoder. The encoder reads the article. The decoder writes the summary one token at a time, looking back at the article through cross-attention. It has 8 + 8 blocks and 300,176,768 parameters, 85% of them in the 250,112-token vocabulary. Weights stay in fp32.

**LoRA:**

```text
W_used = W + B x A        B is (n_out x r), A is (r x n_in), B starts at zero
parameters = r x (n_in + n_out)
```

| Part | Parameters per rank |
|---|---:|
| One encoder block (self-attention + feed-forward) | 8,192 |
| One decoder block (self-attention + cross-attention + feed-forward) | 11,776 |
| All 16 blocks | 159,744 |

At rank 16 on all blocks that's **2,555,904** parameters, about **10.2 MB** in fp32, while the full model is about 1.2 GB.

**Merging:** `W_new = W + B x A` before serving costs no extra time per summary. An unmerged decoder adapter costs time at every one of the 64 writing steps.

**Data: XL-Sum**, English and Hindi BBC article–summary pairs.

| Split | Per language | Used for |
|---|---:|---|
| Training | 2,000 | Training |
| English revision pool | 500, from English training | The only source of replay |
| Validation | 200 | Every decision |
| Test | 500 | Final scores only |

**Fixed before any Hindi run:**

| Setting | Value |
|---|---|
| Rank, alpha | 16, 32 |
| Input / output tokens | 256 / 64 |
| Replay budget | 2 English per 16 Hindi |
| Selector slots | Half from the top of the ranking, half random |
| Warm-up before rescoring | First 10% of Hindi steps, with random replay |
| E's slowing | `gamma = 1`, `floor = 0.25` |
| G2's filter | Drop the 20% least similar to their reference |
| G3 | `beta = 1`; retire at similarity 0.9, checked every 200 steps |
| F | 1-point tolerance; 90% energy |
| Meaningful difference | 1 ROUGE-L or 1 chrF point |
| Seeds | 3 per Hindi-stage arm, all from one English checkpoint |

## Part 3: How we measure

| Measure | Formula |
|---|---|
| ROUGE-L | Longest-common-subsequence F-score, on whitespace tokens (safe for Hindi) |
| chrF | Character n-gram F-score (safe for any script) |
| Script share | `dev(s) = Devanagari letters / all letters`; an English summary with `dev(s) > 0.5` has **slipped** |
| English forgetting | `forget = S_en(before) - S_en(after)` |
| Hindi gain | `gain = S_hi(after) - S_hi(before)` |
| Manual check | Two raters, blind to the arm, rate 30 summaries per language for faithfulness and fluency (1–3). Report agreement |

**The four bills:** update (time, peak memory, tokens), analysis (scoring, generation, and probe passes), serving (time per summary, merged and unmerged), and storage.

**Analysis cost unit:** one epoch is about 2,250 rows x 3 forward-pass units = **6,750 units**. One pass over the 500-example pool is 500 units, **about 7% of an epoch**.

**Is a difference real?** For two arms on the same 500 test articles, `Δ = mean(score_1 - score_2)`. Resample the articles 1,000 times. It counts if **Δ ≥ 1 point and the 95% interval excludes 0**.

**Pre-registration:** commit the settings and predictions P1–P10 with a date before the first Hindi run.

## Part 4: The phases

Each experiment lists its question, formula, example, what to look for, and the principle it tests.

### Phase 0: Setup and pilot

| Check | Pass if |
|---|---|
| One batch trains and one summary generates | It runs |
| After brief English training, 20 summaries are readable | Most are on topic |
| Truncation: reference-summary words in the first 256 tokens, against the full article | Nearly as many |
| Tokens per character, English and Hindi | Recorded, to report cost per language |

If it fails: train longer, change the limits, or switch to Qwen3-0.6B.

### Phase 1: Teach Hindi, and see what is forgotten

#### X: the base state

1. Train an adapter on all blocks with English, and pick the **English checkpoint** on validation.
2. The English checkpoint writes a **past summary** for each pool article. G1–G3 use these.
3. Score both languages, and time a summary.
4. From the English checkpoint, train on Hindi only.

**Example:** English drops from 30 to 22 (`forget = 8`), and Hindi rises from 8 to 25 (`gain = 17`).

| Gate result | Decision |
|---|---|
| Hindi improves and English drops | Continue |
| English barely drops | The replay story stops. Report it, then try Hindi first, or a third language |
| Hindi doesn't improve | Fix the setup |

**Free reference:** use the English checkpoint for English articles and X for Hindi ones. It never forgets, but it stores two adapters and has to switch per request. Every method must beat it on cost. **Tests K.**

#### G5: is the forgetting only the wrong language?

Block Hindi letters while X writes English:

```text
English articles:  logit(token) = -infinity    if the token contains Devanagari
Hindi articles:    logit(token) -= 2           if it contains Latin letters (names are allowed)

script share of forgetting = (S_en(X + guard) - S_en(X)) / (S_en(before) - S_en(X))
```

**Example:** English is 30 before, 22 in X with 25% slipped, and 27 in X with the guard. The script share is `(27 - 22) / (30 - 22) ≈ 60%`, so about 60% of the drop is the wrong language and 40% is lost content.

**Look for:** if the share is high, judge later methods on the content loss that remains. Report every final model with and without the guard. No training is needed. **Tests P.**

### Phase 2: What to revise, and with what

Every arm trains all blocks, with the same Hindi data, steps, and replay budget. Only the **choice** or **text** of the replayed English changes.

#### A: random revision

Each batch of 16 Hindi examples gets 2 random pool examples:

```text
loss = mean over all 18 examples of (- log p(summary | article))
```

**Look for:** less forgetting than X, with similar Hindi gain. It adds about 12% more examples. A is the baseline every selector must beat. **Tests N, K.**

#### B: revise what the loss says is forgotten

```text
l_i  = mean over summary tokens of (- log p(y_t | y_<t, article))
Δl_i = l_i(after warm-up) - l_i(English checkpoint)
```

Rank by `Δl_i`. Fill half the slots from the top, and half at random.

| Example | Before | After | Δl | Priority |
|---|---:|---:|---:|---|
| Election | 0.6 | 1.4 | **+0.8** | First |
| Science | 2.5 | 2.6 | +0.1 | Low: always hard, not forgotten |

**Look for:** B beats A by at least 1 point. Scoring costs about 7% of an epoch. **Tests N.**

#### G1: revise what the model can no longer write

Loss feeds the correct summary in word by word, but in real use the model writes on its own. So make it write, and compare:

```text
sim_i = chrF(new summary_i, past summary_i) / 100
d_i   = 1 - sim_i,  or d_i = 1 if the new summary slipped into Devanagari
```

Rank by `d_i`, with the same slot rule. The replayed text is still the reference, so only the choice differs from B.

| Article | Past | New | sim | Priority |
|---|---|---|---:|---|
| Election | "The ruling party won 54% of seats..." | "The party said the vote was..." | 0.35 | High |
| Cricket | "India beat Australia by six wickets..." | the same | 0.92 | Low |
| Science | "Scientists found water on..." | in Devanagari | — | First |

**Extra finding:** `overlap = |top100(B) ∩ top100(G1)| / 100`. Below 0.5 means **loss and real writing disagree** about what's forgotten, a result even if neither selector wins.

**Optional control:** give A extra random steps worth G1's generation time. If that ties G1, the gain was compute, not choice. **Tests N.**

#### G2: revise with the model's own past summaries

Replay `(article, past summary)` instead of `(article, reference)`, chosen at random as in A. Filter first:

```text
keep a past summary if:  dev <= 0.5,  at least 5 words,
                         and its chrF to the reference isn't in the pool's bottom 20%
```

| Reference | Past summary | Kept? |
|---|---|---|
| "Ruling party wins majority in state vote" | "The ruling party won 54% of seats in the state election" | Yes |
| "Floods hit northern towns" | "The the the weather" | No |

Past summaries come only from the English checkpoint and are never regenerated. That prevents the model from feeding on its own errors.

**Look for:** English within 1 point of A, or better, with no stored references needed. Report how many the filter drops. **Tests N, K.**

#### G3: stop revising once the model can write it again

Starting from the better of B and G1:

```text
token weight:  w_t = 1 + beta x max(0, log p_before(y_t) - log p_now(y_t)),  scaled to average 1
loss:          mean over tokens of w_t x (- log p_now(y_t))
stop rule:     every 200 steps, an example whose new summary has sim >= 0.9, in English,
               leaves replay, and the next-ranked example takes its slot
```

| Step | Election sim | Science sim | In replay |
|---:|---:|---:|---|
| 200 | 0.35 | slipped | Election, Science |
| 600 | **0.93, leaves** | 0.88 | Science, plus the next example |

The article dominates compute, so this saves replay **slots**, not much time.

**Look for:** a third or more of examples retire early, with no English loss compared with the parent arm. **Tests N, K.**

### Phase 3: Which parts of the model to change

#### C1: the cheap location map

Split the model into 20 cells: 5 parts (encoder self-attention, encoder feed-forward, decoder self-attention, cross-attention, decoder feed-forward) x 4 bands of 2 blocks. Using a zero-start probe adapter on the English checkpoint, with 64 rows per language:

```text
change map:  ‖B x A‖ per cell, in the English checkpoint and in X     (free)
pull map:    ‖g_hi(cell)‖                                             (where Hindi wants to change)
clash map:   cos(g_en(cell), g_hi(cell))                              (where Hindi pulls against English)
```

| Cell | Agreement |
|---|---:|
| Encoder self-attention, blocks 1–2 | +0.30 |
| Decoder feed-forward, blocks 7–8 | **−0.50** |

Compute it on two row samples, and use it only if they agree. That's about 6% of an epoch per sample. Then **write down a prediction** for C2.

**Look for:** stable maps, the clash concentrated in the decoder, and a correct prediction. **Tests M.**

#### C2: the trained block check

Random replay as in A. Only the chosen adapters can change:

| Configuration | Can change | Parameters |
|---|---|---:|
| All blocks (= A) | All 16 | 2,555,904 |
| Late | Encoder 7–8, decoder 7–8 | 638,976 |
| Early + late | Encoder 1 and 8, decoder 1 and 8 | 638,976 |

Late and early + late have the same size, so they test **where**. All against late tests **how much**.

**Cost prediction:** gradients in early + late must travel back to encoder block 1. Late-only can stop the backward pass after encoder block 7, and the encoder reading 256 tokens is the costly part:

```text
time(late) < time(all),     time(early + late) ≈ time(all), despite 25% of the parameters
```

**Look for:** whether late-only keeps English and still learns Hindi, and whether the timing matches. Always report both languages, since freezing more can just mean learning less. **Tests M, K.**

#### D: combine the best revision with the best blocks

| Run | Blocks | Revision |
|---|---|---|
| A | All | Random |
| Best revision | All | Best of B, G1, G2, G3 |
| Best blocks | Best from C2 | Random |
| **D** | **Best from C2** | **Best revision** |

```text
interaction = S(D) - S(A) - [S(best revision) - S(A)] - [S(best blocks) - S(A)]
```

Near 0, the effects add up. Negative, they substitute: either one alone is enough. Positive, they help each other.

**Example:** revision gives +3, blocks give +2, and D gives +3, so `interaction = 3 - 3 - 2 = -2`. They partly substitute.

Add a reference run on Hindi plus the full 2,000 English examples, to back any claim of saving. Plot English kept and Hindi gained against total update time. **Tests N, M, K.**

### Phase 4: Where new room goes

#### E: new rank where Hindi clashes

```text
agreement(block) = cos(g_en(block), g_hi(block))          (from C1)
new rank:          zero-start, only in blocks with agreement < 0, full learning rate
slowing:           shared adapter's rate = base x clip(1 + gamma x min(agreement, 0), floor, 1)
```

**Example:** decoder blocks 5–8 have agreement −0.50. They get 14 new rank, which is 14 x 4 x 11,776 = **659,456** parameters, and the shared adapter there trains at 50%.

| Arm | New rank goes | Shared adapter there | It isolates |
|---|---|---|---|
| **E1** | Where agreement < 0 | Normal rate | Our placement rule |
| **E2** | Same count of same-type blocks, at random | Normal rate | Does the map beat chance? |
| **E3** | All 16 blocks, at rank 4 (638,976) | Normal rate | Does concentrating matter? |
| **E4** | Where agreement < 0 | Slowed | Does slowing add anything? |
| *E5 (optional)* | Where agreement < 0 | Frozen | Slowing or freezing? |

The arms are equal in new parameters (within 3%) and use D's setup, so compare them with each other.

| Comparison | First wins | Tie |
|---|---|---|
| E1 against E2 | The clash map finds the right blocks | It's no better than chance |
| E1 against E3 | Concentrating helps | Placement doesn't matter. Stop E |
| E4 against E1 | Slowing adds protection | Placement alone is enough |

No extra analysis is needed: it reuses C1's map. **Tests M.**

### Phase 5: How much room to keep

#### F: shrink, tested by what the model writes

```text
ΔW = B x A = U S Vᵀ;   keep the top k:  B_k = U_k S_k^½,  A_k = S_k^½ V_kᵀ
energy rule (control):  k_E = the smallest k with (s_1² + ... + s_k²) / (all s²) >= 0.9
generation rule (ours): k_G = the smallest rank that passes the generation test
```

**Generation test:** 200 English and 200 Hindi validation summaries. A rank passes if ROUGE-L and chrF are within 1 point of the unshrunk adapter and the summaries are in the right language. Start at `k_E`, step down through 12, 8, 6, and 4, and stop at the first failure. Run it on A and on the overall winner. No training is needed.

| Rank | English | Hindi | Right language | Storage | Passes? |
|---:|---:|---:|---:|---:|---|
| 16 | 30 | 25 | 99% | 5.1 MB | |
| 10 (`k_E`) | 30 | 25 | 99% | 3.2 MB | Yes |
| 6 | 29 | 24 | 98% | 1.9 MB | **Yes: `k_G = 6`** |
| 4 | 25 | 19 | 90% | 1.3 MB | No |

| Result | Meaning |
|---|---|
| `k_G < k_E` | Reading the output finds a smaller adapter than the weights do |
| `k_G = k_E` | The energy rule is enough. Say so |
| `k_E` fails | Only the generation test caught the damage |
| `k_G` differs between A and the winner | The training recipe changes how much room Hindi needs to be remembered |

Merged serving time doesn't change. Unmerged, a smaller adapter is faster. **Tests L, K.**

### Phase 6: Can a prompt prevent the slip?

#### G4: tell the model which language to write

Prefix every input with `summarize in English:` or `summarize in Hindi:`. This needs its own English checkpoint, so it runs as a separate branch. Then run X from it.

| Model | English | Slipped |
|---|---:|---:|
| X | 22 | 25% |
| X with G5's guard | 27 | 0% |
| X, tagged | 26 | 4% |

**Look for:** if the tag recovers much of the drop, language control is learned during training, while G5 fixes it only at writing time. **Tests P.**

### Phase 7: Only if time allows

- **Reverse order:** Hindi first, then English.
- **A third language:** Marathi (Hindi's script) or Gujarati (its own script), with fixed memory and adapter size. New rank follows `r_min + (r_max - r_min) x (1 - highest similarity)`.
- **G1 and G2 together:** choose by drift, and replay the past summary.
- **G2 without references:** articles only.

## Part 5: What the results prove

### Predictions, committed before the runs

| # | Prediction | Breaks if |
|---|---|---|
| P1 | Late-only is faster and lighter than all blocks; early + late saves little | All take the same time |
| P2 | B's scoring adds under 10% to the update time | It costs as much as it saves |
| P3 | C1's maps are stable, clash in the decoder, and predict C2 | The prediction fails |
| P4 | Merged serving is as fast as the base model; unmerged is slower | No difference |
| P5 | F cuts storage and keeps both languages | A language is lost |
| P6 | One merged adapter uses half the reference's storage, at some cost in English | The reference wins every bill |
| P7 | G1 and B's top 100 overlap by less than half | They mostly agree |
| P8 | G5 recovers at least a quarter of the English drop | Almost nothing |
| P9 | G2 keeps English within 1 point of A | Clearly worse |
| P10 | G3 retires a third of examples early, with no English loss | Few retire, or English drops |

### The principle check

| Principle | Experiments | Supported if | Broken if |
|---|---|---|---|
| **M** | C1, C2, D, E | Late-only matches all blocks; C1 predicted it; E1 beats E2 and E3 | Only all blocks learn Hindi; the map fails; placement ties random |
| **N** | A, B, G1, G2, G3 | B or G1 beats random; G2 matches references; G3 frees slots | Random ties everything. That's still useful: random is cheaper |
| **K** | All, especially C2 and serving | Early + late costs about the same time as all blocks; merging removes the serving cost | Time follows parameter count everywhere |
| **L** | F | `k_G` is well below the training rank, and beats the energy rule | Shrinking loses a language, or energy matches |
| **P** | G5, G4 | The guard or the tag recovers a clear part of the drop | Neither recovers anything |

### Which story we tell

| If the results show | Headline |
|---|---|
| G1 or G2 beats its control | "A summarizer that reads its own writing revises better" |
| G5 or G4 recovers most of the drop | "Most forgetting here is the wrong language, and it's nearly free to fix" |
| Random ties every selector | "Cheap random replay is enough" |
| E1 beats E2 and E3 | "One gradient pass finds where a new language needs room" |
| `k_G` is well below `k_E` | "Learn wide, remember narrow" |
| Late-only matches all blocks | "Update the late blocks only: same quality, less cost" |
| English barely drops | "LoRA adds Hindi without forgetting English", then the reverse order |

**Safe floor:** even if every new procedure ties its control, this is a complete, pre-registered study of how revision and block choice interact, with every cost counted.

### Evaluator questions

| Question | Answer |
|---|---|
| "Isn't this ML systems?" | The model's writing decides what to revise, what text to use, when to stop, and how small to get (G1–G3, F) |
| "Isn't the drop just the wrong language?" | We measure exactly that, first, for free (G5, G4) |
| "These methods already exist" | Each part has close relatives. We claim findings: loss against writing (G1 against B), and whether revision and blocks interact (D) |
| "Won't self-replay copy mistakes?" | One round only, a filter, and a manual check (G2) |
| "Why not two adapters, or retraining on everything?" | Both are reference runs with their costs shown |
| "Is anything significant?" | Bootstrap on 500 articles, a 1-point minimum, seed spread, and pre-registration |
| "Is the gain just extra compute?" | G1's compute-matched control |
| "Can three people finish this?" | The tiers below. Tier 1 alone is a complete story |

## Part 6: Running order and tiers

**Order:**
1. Phase 0.
2. Pre-register.
3. X, then G5.
4. A, B, G1, G2.
5. C1 and its prediction, then C2.
6. G3, then D.
7. E.
8. F.
9. G4 whenever a GPU slot is free.

| Tier | Experiments | Training runs (3 seeds) |
|---|---|---:|
| **1: must** | X, A, B, G1, G2, late, early + late, D, full-English reference. G5, C1, and F need no training | **27** |
| **2: method claims** | E1–E4, G3 | **15** |
| **3: extras** | G4, G1's compute-matched control, E5 | **9** |

Tiers 1 and 2 total **42 runs**, and everything totals **51**. If mT5 support takes too long to build, Qwen3-0.6B can run the same plan, with early and late layers in place of encoder and decoder blocks.

## One-sentence pitch

We teach an English news summarizer to write Hindi with LoRA alone, letting the model's own writing decide what English to revise, what to revise it with, and when to stop. We split forgetting into wrong-language and lost-content parts, test where new room should go and how much must be kept, and count every cost along the way.
