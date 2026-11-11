# Phase 2 exploratory results

All five replay arms completed: four Hindi epochs, 500 updates per arm, seed 1. Each arm starts from the same saved English adapter. The prepared data has 2,000 training, 200 validation and 500 test examples per language; the English replay pool has 500 rows. Each update uses 16 Hindi examples and two English replay examples.

**Scientific gate remains `fix_english`.** The starting English validation chrF is 20.7666, below the lead-1 baseline of 22.0174. These are explicitly authorized exploratory results from one seed; completion does not pass the original scientific gate.

## Validation comparison

Higher scores are better. Forgetting is starting English chrF minus final English chrF; a negative value means improvement. Hindi gain is final minus starting Hindi chrF.

| Method | English chrF | Hindi chrF | English forgetting | Hindi gain | Arm wall minutes |
|---|---:|---:|---:|---:|---:|
| Before Hindi | 20.7666 | 13.5387 | +0.0000 | +0.0000 | - |
| Hindi only, source seed 1 | 14.3909 | 22.1558 | +6.3757 | +8.6170 | - |
| Random replay | 21.2132 | 22.4333 | -0.4466 | +8.8946 | 31.87 |
| Loss-based replay | 20.3709 | 22.0211 | +0.3957 | +8.4824 | 25.76 |
| Generation-drift replay | 19.4991 | 21.6819 | +1.2675 | +8.1431 | 36.62 |
| Past-summary replay | 19.1196 | 21.7659 | +1.6470 | +8.2272 | 24.42 |
| Repair-stop replay | 20.8830 | 21.7392 | -0.1164 | +8.2005 | 30.41 |

## Test comparison

Test scores are report-only. All rows below use plain decoding consistently. The full report also contains guarded decoding, all metrics and paired bootstrap comparisons.

| Method | English chrF | English ROUGE-L | Hindi chrF | Hindi ROUGE-L |
|---|---:|---:|---:|---:|
| Before Hindi | 20.5259 | 14.8116 | 14.0076 | 12.6323 |
| Hindi only, source seed 1 | 15.3401 | 4.1678 | 22.6001 | 19.0772 |
| Random replay | 20.7537 | 14.5883 | 22.5609 | 18.9330 |
| Loss-based replay | 20.0509 | 14.2779 | 23.3617 | 19.5052 |
| Generation-drift replay | 19.3158 | 14.2963 | 23.2445 | 19.3967 |
| Past-summary replay | 19.1035 | 13.6842 | 22.7006 | 19.6073 |
| Repair-stop replay | 20.5217 | 14.4308 | 22.8383 | 19.3815 |

## Interpretation

Random replay has the highest English and Hindi validation chrF among these five arms. All replay methods retain much more English than the same-seed Hindi-only source arm; none of the targeted methods improves both validation chrF scores over random replay. Loss-based replay has the highest Hindi test chrF, but test scores do not select the method. This single-seed pilot does not establish a general method ranking.

The repair-stop arm used the loss selector as its parent recipe and retired 0 pool examples. Stopping therefore produced no observed retirement in this run.

Summed recorded replay-arm wall time is 149.08 minutes (2.48 hours). This includes training, analysis, checkpoint work and evaluation. It is not accelerator busy time or an estimate for a different host. Source Hindi-only timings come from a different XPU host.

ROUGE-L is the project's unofficial whitespace metric. Right-language share is a Unicode-script proxy; every Phase 2 evaluation view recorded 100%. Scores are not accuracy percentages. The source base revision was recorded as `main`, rather than an immutable model revision.

## Saved artifacts

- [Full Markdown report](/results/code/phase2-exploratory/phase2/report.md), [JSON](/results/code/phase2-exploratory/phase2/report.json), [CSV](/results/code/phase2-exploratory/phase2/report.csv).
- Each method directory contains its completed `arm.json`, metric CSVs and per-example predictions.
- [Final adapters](/results/code/phase2-exploratory/phase2/final) contain the five trained adapter exports, metadata and checksum manifests. They require the matching mT5 base; they are inference exports and do not include optimizer/RNG state for exact training resume.
- Recovery checkpoints, optimizer state and verbose runtime logs remain on the training PC and are excluded from Git. The original local run was not deleted or reset.
- [Phase 1 training report](/results/code/summarization-passes5/phase1/training-report.md) records the imported English/Hindi baseline and its incomplete serving benchmark.

## Verification

All five final adapter checksum manifests passed. Recomputed chrF matched the saved metrics for all 14,000 prediction rows; 13 CPU handoff and phase-rule tests passed.
