# Summarization experiments

Owns `src/summarization_experiment.py`: phases, arms, fixed settings, winner rules, gates, measures, statistics, serving timing, manual-check sheets, and reports for [Idea_Refined_2.md](/Idea_Refined_2.md). It also owns the placement and shrink rules below. `AdapterRoomPipeline` (`src/pipelines/adapter_room.py`) carries out C1, C2, and E. `GenerationShrinkPipeline` (`src/pipelines/generation_shrink.py`) carries out F. Neither pipeline trains, and neither chooses replay; that is [revision.md](/design/revision.md). They call the model operations in [models.md](/design/models.md): adapter names, `AdapterPlan`, probe gradients, `shrink_adapters`, and `merged_copy`. The [plan guardrails](/design/summarization-plan.md) apply.

```sh
python -m src.summarization_experiment [--pilot] [--output <dir>] [--phase N]
```

`src/summarization_experiment.py` runs Phase 0 through Phase 7 in order. The default protocol is `scientific`. `--pilot` selects the pilot. The default output is the current directory. Each phase lives in `src/phases/phaseN.py` and calls the existing revision, adapter-room, and shrink pipelines. With no `--phase`, a directory that already has `progress.json` continues from that last phase through Phase 7. A directory with no progress runs Phase 0 through Phase 7. `--phase N` runs that phase only, reading earlier reports from the same output. Each attempt directory is kept. A finished `report.json` is reused and is not deleted. Only `scientific` results are reported. `pilot` and `smoke` are development evidence. The runner does not read or require `preregistration.json`.

**Resume.** A finished phase report is the boundary between phases: a later run reads it and does not repeat that phase. Inside a phase, each finished arm writes `arm.json` next to its `run.json`. A later run of that phase reads `arm.json` and does not load or retrain that arm. An arm with no `arm.json` resumes through the attempt checkpoint. Phase 0 writes `progress.json` and `final/batch` after its one batch; a later run rebuilds the report from those files. Restarting a finished attempt does not add the reload to its wall-clock time.

**Phase report.** `details` records `time` (phase seconds, and per arm the wall-clock seconds, GPU hours, and peak memory), `runs` (passes as epochs, optimizer steps, and the chrF, ROUGE-L, and right-language scores), and `final_model`. `final/<arm>/` is a copy of that arm's adapter snapshot so a later comparison can load it without replaying the attempt. It is not a second recovery checkpoint.

The shared phase wrapper owns cached-report validation, status/progress logging,
and JSON/CSV/Markdown report publication. It binds settings and prepared-data
identity before reusing a report or finished arm. Reports publish last, after
the referenced snapshots and tabular logs. Failed/interrupted work keeps its
prior checkpoints and records its error. Analysis jobs report zero training
epochs; Phase 0 reports its actual fraction of an English epoch.

Scientific and pilot runs use the fixed settings below. Hindi arms use seeds 1, 2, and 3, with
per-seed attempts and averaged validation metrics for decisions. Source checkpoints
and final exports remain available per seed. A pilot/smoke result is never reused
as a scientific result.

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
| Model | `google/mt5-small`, bf16 or fp32 |
| LoRA | Rank 16, alpha 32, all projections of all blocks |
| Limits | 256 input tokens, 64 output tokens |
| Replay | 2 pool rows per 16 Hindi rows |
| Slots | Half from the ranking, half random |
| Warm-up | The first 10% of Hindi steps, with random replay |
| G2 filter | 0.2 |
| G3 | `beta = 1`, `stop_similarity = 0.9`, `refresh_steps = 200` |
| E | `gamma = 1`, `floor = 0.25`, `new_room_budget = 638,976` |
| C1 | `stability_threshold = 0.5` |
| F | Tolerance 1 point, energy 0.9, grid 12, 8, 6, 4 |
| Decoding | Greedy, up to 64 new tokens, the same for every arm |
| Meaningful difference | 1 ROUGE-L or 1 chrF point |
| Seeds | English stage: seed 0. Hindi stage: seeds 1, 2, 3 |
| Passes | `english_passes = 1`, `hindi_passes = 1` |

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
| `random_compute_matched` | G1's control | `selector=random`, `extra_random_steps` set from measured times |
| `late`, `early_late` | C2 | `selector=random`, block plan |
| `combined` | D | Best blocks plus best revision |
| `full_english` | Reference | `selector=full` |
| `room_placed`, `room_random`, `room_spread`, `room_placed_slow` | E1–E4 | D's setup plus room plan |
| `room_placed_frozen` | E5, optional | As above |
| `hindi_only --prefix` | G4 | `selector=none` from the prefixed English checkpoint |
| `maps` | C1 | Analysis job on the English checkpoint and X |
| `shrink` | F | Analysis job on `random` and the overall winner |
| `serving` | Serving timing | Analysis job |
| `report` | All comparisons | No model work |

G5 isn't a job. Every evaluation reports both unguarded and guarded scores.

C1 and C2 are `ablation`. E and F are `project`.

## Adapter placement (C1, C2, E) and shrinking (F)

These rules decide which adapter parts can change, where new room goes, and how much room is kept. Part, cell, and block names, and the `AdapterPlan` the model installs, are in [models.md](/design/models.md).

### C1: the location maps

The `maps` job uses the **English checkpoint**, an immutable saved artifact, not the evolving learner. The change map reads saved adapters and needs no forward pass. The pull and clash maps use a separate **probe** role: the English checkpoint plus a fresh LoRA adapter on every projection with `B = 0` and a deterministic `A`. Gradients are taken with respect to the probe's `B` factors only, with no optimizer step. The learner is untouched.

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

**Run on:** `random` (A) and the overall winner, chosen by the winner rule below.

**Outputs:** `k_E`, `k_G`, the full table of ranks tested with scores and storage, and the shrunk adapter saved at `k_G`. Cost is recorded as `shrink_generation`.

| Result | Reported as |
|---|---|
| `k_G < k_E` | Reading the output finds a smaller adapter than the weights do |
| `k_G = k_E` | The energy rule is enough |
| `k_E` fails | Only the generation test caught the damage |
| `k_G` differs between A and the winner | The training recipe changes how much room is needed to remember |

## Winner rules

Chosen on **validation**, never test:

| Choice | Rule |
|---|---|
| G3's parent | Of `loss` and `drift`, the one with lower English validation forgetting (chrF), averaged over seeds |
| Best revision | Of `loss`, `drift`, `past`, and `repair_stop`, the one with the lowest English validation forgetting whose Hindi validation gain is within 1 point of `random`'s. Ties go to the cheaper arm by total update time |
| Best blocks | Of `all`, `late`, and `early_late`, by the same rule |
| Overall winner | Of `combined` and the E arms that ran, by the same rule |

If no arm meets the Hindi condition, the winner is `random`, and that's reported.

## Gates

| After | Condition | Then |
|---|---|---|
| Pilot | Most of 20 summaries readable; truncation keeps nearly all summary words | Continue. Otherwise change limits or steps, or switch to Qwen3-0.6B |
| X | English validation forgetting at least 1 point **and** Hindi validation gain at least 1 point | Continue |
| X | English forgetting under 1 point | Stop the replay story. Report it. Run the reverse order |
| X | Hindi gain under 1 point | Fix the setup |
| C1 | Two-sample agreement below `stability_threshold` | Don't run E |
| E | The clash map gives no block below zero | Don't run E. Report it |
| E | E1 ties E3 | Stop E there. Report "placement doesn't matter" |

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

**The `serving` job** times greedy generation on the same 200 test articles per language, at batch size 1, with no warm-up and one pass (user override, 2026-11-08). It reports the median and 95th percentile across article requests, plus mean summary length. These timing passes are separate from the three Hindi training seeds. It covers:

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

1. Average each article's score over the 3 seeds for each arm.
2. `Δ = mean over articles of (score_1 - score_2)`.
3. `paired_bootstrap` with 1,000 resamples of articles and a fixed seed gives the 95% interval.
4. A difference **counts** if `|Δ| >= 1` point and the interval excludes 0. Otherwise it's reported as a tie.
5. Also report the per-seed `Δ` and its standard deviation across seeds.

Never pick seeds, drop failed seeds silently, or report a confidence interval from one run.

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

| Tier | Jobs | Training runs |
|---|---|---:|
| 1 | `english`, `hindi_only`, `random`, `loss`, `drift`, `past`, `late`, `early_late`, `combined`, `full_english` (9 Hindi arms x 3 seeds), plus `maps`, `shrink`, `serving` | 27 |
| 2 | `room_placed`, `room_random`, `room_spread`, `room_placed_slow`, `repair_stop` | 15 |
| 3 | `english --prefix`, `hindi_only --prefix`, `random_compute_matched`, `room_placed_frozen` | 9 |

The English checkpoints aren't counted as Hindi-stage runs. Run the tiers in order, and the plan's running order within them.
