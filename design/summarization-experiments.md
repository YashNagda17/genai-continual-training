# Summarization experiments

Owns `src/summarization_experiment.py`: phases, arms, fixed settings, winner rules, gates, measures, statistics, serving timing, manual-check sheets, and reports for [Idea_Refined_2.md](/Idea_Refined_2.md). It also owns the placement and shrink rules below. `AdapterRoomPipeline` (`src/pipelines/adapter_room.py`) carries out C1, C2, and E. `GenerationShrinkPipeline` (`src/pipelines/generation_shrink.py`) carries out F. Neither pipeline trains, and neither chooses replay; that is [revision.md](/design/revision.md). They call the model operations in [models.md](/design/models.md): adapter names, `AdapterPlan`, probe gradients, `shrink_adapters`, and `merged_copy`. The [plan guardrails](/design/summarization-plan.md) apply.

```sh
python -m src.summarization_experiment [--pilot] [--output <dir>] [--phase N]
```

For a development comparison, `--pilot --epochs 5` overrides both English and
Hindi training passes to five. The default is four passes per language.
Nondefault epoch counts require a pilot or smoke protocol and a separate output
directory; they are not the fixed scientific protocol. The chosen pass counts
enter the saved settings, run identity, step budgets and epoch reports.

`src/summarization_experiment.py` runs Phase 0 through Phase 7 in order. The default protocol is `scientific`. `--pilot` selects the pilot. The default output is the current directory. Each phase lives in `src/phases/phaseN.py` and calls the existing revision, adapter-room, and shrink pipelines. With no `--phase`, a directory that already has `progress.json` continues from that last phase through Phase 7. A directory with no progress runs Phase 0 through Phase 7. `--phase N` runs that phase only, reading earlier reports from the same output. Each attempt directory is kept. A finished `report.json` is reused and is not deleted. Only `scientific` results are reported. `pilot` and `smoke` are development evidence. The runner does not read or require `preregistration.json`.

**Resume.** A finished phase report is the boundary between phases: a later run reads it and does not repeat that phase. Inside a phase, each finished arm writes `arm.json` next to its `run.json`. A later run of that phase reads `arm.json` and does not load or retrain that arm. An arm with no `arm.json` resumes through the attempt checkpoint. Phase 0 writes `progress.json` and `final/batch` after its one batch; a later run rebuilds the report from those files. Restarting a finished attempt does not add the reload to its wall-clock time.

**Phase report.** `details` records `time` (phase seconds, and per arm the wall-clock seconds, GPU hours, and peak memory), `runs` (passes as epochs, optimizer steps, and the chrF, ROUGE-L, and right-language scores), and `final_model`. `final/<arm>/` is a copy of that arm's adapter snapshot so a later comparison can load it without replaying the attempt. It is not a second recovery checkpoint.

The shared phase wrapper owns cached-report validation, status/progress logging,
and JSON/CSV/Markdown report publication. It binds settings and prepared-data
identity before reusing a report or finished arm. Reports publish last, after
the referenced snapshots and tabular logs. Failed/interrupted work keeps its
prior checkpoints and records its error. Analysis jobs report zero training
epochs; Phase 0 reports its actual fraction of an English epoch.

Scientific and pilot runs use the fixed settings below. A pilot/smoke result is never reused
as a scientific result.

**Seeds: screen on one, confirm on three** (2026-11-10). Every scientific Hindi arm runs the
**screening seed** (seed 1) first. All arms use it, so they see the same Hindi batches in the same
order. Three arms are **confirmed** on seeds 1, 2, and 3, because the headline claims rest on them:

| Confirmed arm | When its extra seeds run |
|---|---|
| X (`hindi_only`) | Phase 1, from the start. Its spread is the seed-noise check below |
| A (`random`) | Phase 2, from the start. It is the bar every method must beat |
| The overall winner | Phase 4, after the winner rule picks it |
| The best revision | Phase 3, only if X's English validation chrF has a standard deviation above `max_seed_spread` (1 point) across seeds |

Confirmation reuses the arm's attempt directory, so seed 1 is read from its `arm.json` and only
seeds 2 and 3 train. A confirmed best revision is written back into its Phase 2 directory, and a
confirmed D into Phase 3's. Per-seed attempts and final exports stay available. Arms with one seed
are reported as "single seed, screening".

## Protocols

| Protocol | Data | Purpose |
|---|---|---|
| `smoke` | A tiny local JSONL of a few English and Hindi rows | One batch trains and one summary generates. CPU is fine |
| `pilot` | The Phase 0 checks on real XL-Sum rows | Readable summaries, truncation coverage, tokens per character, and fixing `hindi_passes` and English passes |
| `scientific` | The full splits below | Reported results only |

## Data splits

Drawn once with a fixed seed from XL-Sum's official splits. The ID lists are saved in the manifest, and every arm uses the same lists.

| Split | Per language | Source |
|---|---:|---|
| Training | 2,000 | Official train |
| English pool | 500 | A subset of the English training 2,000 |
| Validation | 200 | Official validation |
| Test | 500 | Official test |

## Fixed settings

The runner uses the values below. It does not read `preregistration.json`.

| Setting | Value |
|---|---|
| Model | `google/mt5-small`, fp32 |
| LoRA | Rank 16, alpha 32, all projections of all blocks |
| Task prompt | `summarize: ` is prepended to every English and Hindi article before tokenization |
| Limits | 256 input tokens, 64 output tokens |
| Optimizer | AdamW, learning rate `1e-3`, weight decay 0.01, gradient clip 1. Linear learning-rate warm-up over the first 5% of each stage's updates, then constant. Every stage starts a fresh optimizer |
| Passes | `english_passes = 5`, `hindi_passes = 5`. The English stage validates after each pass, saves every pass's chrF, ROUGE-L, and right-language share, and keeps the best pass by validation chrF |
| Batches | 16 Hindi rows per update (plus replay). Generation, scoring, and evaluation use 16 rows per batch (`analysis_batch = 16`). The model accepts up to 32 rows |
| Loss chunk | 32 target positions per full-vocabulary cross-entropy call (`logit_chunk_size`), with a custom first-order backward that fills one logits-gradient buffer in chunks. This bounds scratch memory and avoids repeated full-logits gradient additions. Previous Intel Arc measurements using the old 512-position loss are historical timings, not measurements of this backward implementation |
| Replay | 2 pool rows per 16 Hindi rows |
| Slots | Half from the ranking, half random |
| Replay warm-up | The first 10% of Hindi steps, with random replay |
| G2 filter | 0.2 |
| G3 | `beta = 1`, `stop_similarity = 0.9`, `refresh_steps = 200` |
| E | `gamma = 1`, `floor = 0.25`, `new_room_budget = 638,976` |
| C1 | `stability_threshold = 0.5` |
| F | Tolerance 1 point, energy 0.9, grid 12, 8, 6, 4 |
| Decoding | Greedy, up to 64 new tokens, mT5's 100 placeholder tokens (`<extra_id_N>`) never generated, no repeated trigram (`no_repeat_ngram_size = 3`). The same for every arm |
| Meaningful difference | 1 ROUGE-L or 1 chrF point |
| Seeds | English stage: seed 0. Hindi stage: screening seed 1; confirmation seeds 2 and 3 for the arms above |
| Conditional thresholds | `max_seed_spread = 1` chrF point; `g4_min_share = 0.25` |

**Why these changed (2026-11-10).** The first Phase 1 run used one pass at learning rate `2e-4`:
125 English updates. English loss was still falling steeply (22 → 6 per token), every summary began
with mT5's pretraining placeholder `<extra_id_0>`, and the English checkpoint scored 11.6 chrF on
validation against 22.0 for copying the article's first sentence. Hindi-only training then *raised*
English chrF, because the adapter was still learning the task itself. Forgetting can't be measured
before English is learned, so the learning rate, passes, decoding rules, and the lead-1 gate below
were added.

## Jobs and arms

| Job / arm | Experiment | Config |
|---|---|---|
| `english` | English checkpoint | `stage=english` |
| `english --prefix` | G4's English checkpoint | `stage=english`, dataset prefix on |
| `hindi_only` | X | `selector=none` |
| `random` | A, and C2's `all` | `selector=random` |
| `loss` | B | `selector=loss` |
| `drift` | G1 | `selector=drift` |
| `past` | G2 | `selector=random`, `replay_text=past` |
| `repair_stop` | G3 | Parent of B or G1 by the winner rule, `repair_stop=true` |
| `random_compute_matched` | G1's control, **only if G1 is the best revision** | `selector=random`, `extra_random_steps = ceil((G1's update + analysis seconds − A's) / A's seconds per update)`, on the screening seed |
| `late`, `early_late` | C2 | `selector=random`, block plan |
| `combined` | D | Best blocks plus best revision |
| `full_english` | Reference | `selector=full` |
| `room_placed`, `room_random`, `room_spread` | E1–E3 | D's setup plus room plan |
| `room_placed_slow` | E4, **only if E1 beats E2 and E3** by at least 1 English chrF point of forgetting on the screening seed | As E1, plus `lr_scale` |
| `room_placed_frozen` | E5, Phase 7 only | As above |
| `english --prefix`, `hindi_only --prefix` | G4, **only if G5's validation script share of forgetting is at least `g4_min_share`** | `selector=none` from the prefixed English checkpoint. Otherwise Phase 6 reports `skipped` with the share |
| `maps` | C1 | Analysis job on the English checkpoint and X |
| `shrink` | F | Analysis job on `random` and the overall winner |
| `serving` | Serving timing | Analysis job |
| `report` | All comparisons | No model work |

G5 isn't a job. Every evaluation reports both unguarded and guarded scores. Phase 1 reports G5's
share on test (a measure) and on validation (the G4 decision).

The `smoke` protocol sets `force_conditional`, so every conditional arm runs there and its code
path is tested. Scientific and pilot runs leave it off.

C1 and C2 are `ablation`. E and F are `project`.

## Adapter placement (C1, C2, E) and shrinking (F)

These rules decide which adapter parts can change, where new room goes, and how much room is kept. Part, cell, and block names, and the `AdapterPlan` the model installs, are in [models.md](/design/models.md).

### C1: the location maps

The `maps` job uses the **English checkpoint**, an immutable saved artifact, not the evolving learner. The change map reads saved adapters and needs no forward pass. The pull and clash maps use a separate **probe** role: the English checkpoint plus a fresh LoRA adapter on every projection with `B = 0` and a deterministic `A`. Gradients are taken with respect to the probe's `B` factors only, with no optimizer step. The learner is untouched.

Change-map projection norms accumulate by cell on the compute device, with one packed transfer for the completed map. Probe-gradient running totals are packed for one device-to-host transfer per analysis batch, split into the existing plain-JSON group lists, and committed with the cursor at the same recovery boundary. Group layout and sample-weighted means are unchanged.

Probe totals also pack CPU-to-device restoration across groups, both for each batch and for the returned mean gradients. Pull norms and cosine agreements are collected on-device and transferred together once per cell/block view. Persisted maps remain plain JSON.

The change map requests `effective_update_norms()` from the model. Shared-only adapters use low-rank Gram products instead of allocating full update matrices; extra-rank adapters use the dense sum so all cross terms are included. The map still sums each projection's Frobenius norm in its cell, within float32 tolerance.

```text
change map:  c(cell)      = ‖(alpha/r) B A‖_F summed over the cell's projections,
                            in the English checkpoint and in X
pull map:    u(cell)      = ‖g_hi(cell)‖
clash map:   a(cell)      = cos(g_en(cell), g_hi(cell))
per block:   a(block)     = cos over the concatenated probe-B gradients of that block
```

`g_en` is the mean probe-B gradient of the target loss over 64 English **pool** rows, and `g_hi` the same over 64 Hindi **training** rows. No other English data is accessible in the Hindi stage.

**Two samples.** Compute the pull and clash maps on two disjoint seeded samples of 64 + 64 rows. The maps **agree** if the Spearman correlation of the 20 clash values is at least `stability_threshold`. If they disagree, report this and don't use the clash map for E. The arm then falls back to the C2 result alone.

**Prediction.** The runner writes `c1_prediction.json` before any scientific C2 run: the configuration the maps predict will keep English best, and the one predicted to learn Hindi best. C2 scientific arms are refused until that file exists.

**Optional knock-out map.** In X, reset one cell's adapter to its English-checkpoint value, and measure the target loss on 100 validation rows per language: 20 cells x 200 rows. Generate summaries for the top three cells to confirm.

**Cost:** about 6% of an epoch per sample, recorded as `probe_pass`.

### C2: block plans

`late` and `early_late` use random replay, as in A, and no new rank. `all` is the `random` arm.

| Arm | `trainable` | Trainable parameters at rank 16 |
|---|---|---:|
| `all` (= A) | All 16 blocks | 2,555,904 |
| `late` | Encoder 7–8, decoder 7–8 | 638,976 |
| `early_late` | Encoder 1 and 8, decoder 1 and 8 | 638,976 |

Late and early + late have the same size, so they test **where**. All against late tests **how much**.

**Cost prediction (P1).** Gradients in `early_late` must travel back to encoder block 1. `late` stops the encoder backward after block 7, and the encoder reading 256 tokens is the costly part:

```text
time(late) < time(all),     time(early_late) ≈ time(all), despite 25% of the parameters
```

The backward cut is the freeze rule in [models.md](/design/models.md). Peak memory and time are recorded per arm. Always report both languages, since freezing more can just mean learning less.

### E: new room where Hindi clashes

Starts from D's shared-adapter plan and replay, kept fixed for every arm. Reuses C1's block agreement. No extra analysis pass.

```text
clashing blocks:  K = { block : a(block) < 0 }
new rank:         r_new = round(new_room_budget / sum over K of per-block cost per rank)
slowing:          lr_scale(block) = clip(1 + gamma x min(a(block), 0), floor, 1)     for block in K
```

`new_room_budget` is 638,976, the late budget. `gamma` is 1 and `floor` is 0.25. New rank uses `alpha = 2 x r_new`.

| Arm | `new_rank` | Shared adapter in the new-rank blocks |
|---|---|---|
| `room_placed` (E1) | `r_new` on every block in K | Normal rate |
| `room_random` (E2) | `r_new` on `|K|` blocks drawn at random per seed, **of the same stack and type mix as K** | Normal rate |
| `room_spread` (E3) | `round(new_room_budget / 159,744)` on all 16 blocks | Normal rate |
| `room_placed_slow` (E4) | As E1 | `lr_scale` from the rule above |
| `room_placed_frozen` (E5, optional) | As E1 | Removed from `trainable` |

**Example:** decoder blocks 5–8 have agreement −0.50, so `K` is those 4 decoder blocks. `r_new = round(638,976 / (4 x 11,776)) = 14`, which is 659,456 new parameters. The spread arm gets `round(638,976 / 159,744) = 4` on all blocks, which is 638,976. In E4, the shared adapter in blocks 5–8 trains at `1 + 1 x (-0.5) = 50%`.

**Rules:**

- New-parameter counts of all arms must be within 3% of each other, or the run is rejected before training.
- The illustrative 659,456 versus 638,976 counts differ by about 3.2%, so that
  particular rounded family fails this rule. Report the mismatch and do not silently widen the tolerance.
- E2 reports its overlap with K for each seed.
- If K is empty, E isn't run, and that's reported as a result: Hindi nowhere clashes with English.
- If the clash map failed the two-sample check, E isn't run.

Merged, the new rank costs nothing at serving time. Unmerged, decoder rank runs at every writing step. The `serving` job times both.

### F: how much room to keep

`GenerationShrinkPipeline` operates on a finished checkpoint. No training. Shrink with `shrink_adapters`, then save.

**Energy rule (the control):** one `k` shared by all projections, from `energy_rank`:

```text
k_E = the smallest k with   sum over projections of (s_1² + ... + s_k²)   >= energy_fraction x sum over projections of all s²
```

`energy_fraction` is 0.9.

**Generation test.** Use 200 English and 200 Hindi **validation** articles, with greedy decoding up to 64 tokens. A rank `k` passes if, for both languages:

```text
ROUGE-L(k) >= ROUGE-L(unshrunk) - tolerance
chrF(k)    >= chrF(unshrunk)    - tolerance
right-language share(k) >= right-language share(unshrunk) - tolerance_points
```

`tolerance` is 1 point. A summary is in the right language when `script_share` is at most 0.5 for English and above 0.5 for Hindi.

**Search:**

```text
test k_E
if it passes: step down through the grid ranks below k_E (12, 8, 6, 4), stop at the first failure;
              k_G = the smallest rank that passed
if it fails:  step up through the grid ranks above k_E, then the full rank; k_G = the first rank that passes
```

**Run on:** `random` (A) and the overall winner, chosen by the winner rule below, on the screening seed only.

**Outputs:** `k_E`, `k_G`, the full table of ranks tested with scores and storage, and the shrunk adapter saved at `k_G`. Cost is recorded as `shrink_generation`.

| Result | Reported as |
|---|---|
| `k_G < k_E` | Reading the output finds a smaller adapter than the weights do |
| `k_G = k_E` | The energy rule is enough |
| `k_E` fails | Only the generation test caught the damage |
| `k_G` differs between A and the winner | The training recipe changes how much room is needed to remember |

## Winner rules

Chosen on **validation**, never test, and on the **screening seed** for every arm, including
confirmed ones, so arms are compared on the same Hindi batches. Ties use seed 1's update time:

| Choice | Rule |
|---|---|
| G3's parent | Of `loss` and `drift`, the one with lower English validation forgetting (chrF) |
| Best revision | Of `loss`, `drift`, `past`, and `repair_stop`, the one with the lowest English validation forgetting whose Hindi validation gain is within 1 point of `random`'s. Ties go to the cheaper arm by total update time |
| Best blocks | Of `all`, `late`, and `early_late`, by the same rule |
| Overall winner | Of `combined` and the E arms that ran, by the same rule |

If no arm meets the Hindi condition, the winner is `random`, and that's reported.

## Gates

| After | Condition | Then |
|---|---|---|
| Pilot | Most of 20 summaries readable; truncation keeps nearly all summary words | Continue. Otherwise change limits or steps, or switch to Qwen3-0.6B |
| English checkpoint | English validation chrF at or below **lead-1** (the article's first sentence, split at `.`, `!`, `?`, or `।`) | `fix_english`: train English further before any Hindi arm. Checked before the X conditions |
| X | English validation forgetting at least 1 point **and** Hindi validation gain at least 1 point | Continue |
| X | English forgetting under 1 point | Stop the replay story. Report it. Run the reverse order |
| X | Hindi gain under 1 point | Fix the setup |
| X | English validation chrF standard deviation across seeds above `max_seed_spread` | Confirm the best revision on all seeds |
| Best revision | It is G1 (`drift`) | Run `random_compute_matched` |
| C1 | Two-sample agreement below `stability_threshold` | Don't run E |
| E | The clash map gives no block below zero | Don't run E. Report it |
| E | E1 doesn't beat both E2 and E3 by 1 point of forgetting | Don't run E4. Report "placement doesn't matter" |
| G5 | Validation script share of forgetting below `g4_min_share`, or undefined | Don't run G4 |

## Measures

Computed for every Hindi-stage arm and seed, on the 500 test articles per language, and again on validation for decisions:

| Measure | Definition |
|---|---|
| ROUGE-L, chrF | `numerics.rouge_l` on whitespace tokens, `numerics.chrf` on characters. Both labelled unofficial |
| Forgetting, gain | `forget = S_en(English checkpoint) - S_en(arm)`, `gain = S_hi(arm) - S_hi(English checkpoint)` |
| Slip rate | Share of English summaries with `script_share > 0.5` |
| Script share of forgetting (G5) | `(S_en(arm + guard) - S_en(arm)) / (S_en(English checkpoint) - S_en(arm))`. Null when the denominator is at most 0 |
| Overlap (G1) | `top_overlap(rank_B, rank_G1, 100)` for the same seed |
| Interaction (D) | `interaction(S(A), S(best revision), S(best blocks), S(D))` |
| Retired share (G3) | Retired rows divided by the rows that were ever active |
| Filter drops (G2) | Count and share |

## Cost and serving

**The four bills,** recorded on every job:

| Bill | Recorded |
|---|---|
| Update | Wall-clock seconds, GPU-seconds, peak GPU memory, tokens per language, optimizer steps, trainable parameters |
| Analysis | Seconds and forward-pass units for `pool_scoring`, `pool_generation`, `stop_regeneration`, `probe_pass`, `shrink_generation`, `guard_generation` |
| Serving | The `serving` job, below |
| Storage | Bytes per adapter and checkpoint, the pool with its practice data, and the English-checkpoint artifact |

**Forward-pass units:** a forward pass over one row is 1 unit, and a training row (forward plus backward) is 3. An epoch's size is `rows x 3`.

**The `serving` job** times greedy generation on the same 200 test articles per language, at batch size 1, with no warm-up and one pass (user override, 2026-11-08). The model stays loaded for every article in one timing file. A request's time is that summary, with the device synchronized before and after it. A file from a run that loaded the model once per article is measured again. It reports the median and 95th percentile across article requests, plus mean summary length. These timing passes are separate from the three Hindi training seeds. It covers:

| Model | Merged | Unmerged |
|---|---|---|
| Base mT5-small | — | — |
| English checkpoint | Yes | Yes |
| Best blocks, E1, E3, overall winner | Yes | Yes |
| The shrunk winner from F | Yes | Yes |
| Two-adapter reference (English checkpoint for English, X for Hindi) | Two merged copies: report memory | Unmerged swap per request: report time |

Batch size 32 is optional.

## Statistics

For two arms on the same articles:

1. Average each article's score over the seeds **both** arms ran: all three for two confirmed arms, the screening seed otherwise. The comparison records those seeds.
2. `Δ = mean over articles of (score_1 - score_2)`.
3. `paired_bootstrap` with 1,000 resamples of articles and a fixed seed gives the 95% interval.
4. A difference **counts** if `|Δ| >= 1` point and the interval excludes 0. Otherwise it's reported as a tie.
5. Also report the per-seed `Δ` and, with more than one shared seed, its standard deviation across seeds.

Never pick seeds or drop failed seeds silently. A single-seed comparison's interval covers article
noise only, and is labelled "screening".

## Manual check

The `report` job exports blind sheets: 30 test articles per language, fixed in the manifest, for X, A, the best revision, and the overall winner. Rows are shuffled, arm names are replaced by codes, and the key is kept in a separate file. Two raters score faithfulness and fluency (1–3) and mark the language. The report computes `cohen_kappa` per question, then unblinds.

## Reports

The `report` job reads completed attempts only, one per `run_id`, and writes:

- **Main table:** per arm, English forgetting, Hindi gain, slip rate, guarded scores, and the four bills, each with seed spread.
- **Comparisons:** every pair named in the plan's "look for" rules, as `Δ` with its interval and a counts-or-tie verdict.
- **Prediction check:** P1–P10, each marked supported, broken, or not run, from the pre-registered rule.
- **Principle check:** M, N, K, L, and P, each with the comparisons that support or break it, as in the plan.
- **Headline candidates:** the plan's "which story we tell" rows whose conditions hold.

The phase support module owns the active report summaries. The legacy CSReL
experiment entry point has been removed. Test scores remain report-only; gates,
winner selection, and shrinking use validation.

## Tiers

| Group | Jobs | Hindi training runs |
|---|---|---:|
| Screening (seed 1) | `hindi_only`, `random`, `loss`, `drift`, `past`, `repair_stop`, `late`, `early_late`, `combined`, `full_english`, `room_placed`, `room_random`, `room_spread`, plus `maps`, `shrink`, `serving` | 13 |
| Confirmation (seeds 2, 3) | `hindi_only`, `random`, the overall winner | 6 |
| Conditional | `room_placed_slow`, `random_compute_matched`, `hindi_only --prefix` (plus its English checkpoint), and the best revision's seeds 2 and 3 | 0–5 |
| Phase 7 | `drift_past`, `room_placed_frozen` | 1–2 |

That is 19 required Hindi runs, against 51 before. The English checkpoints aren't counted as
Hindi-stage runs. If the winner is `random`, nothing extra is confirmed.
