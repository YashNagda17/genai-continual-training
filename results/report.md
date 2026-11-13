# Experiment results
Model: mBART multilingual summarization, 619,502,592 parameters, LoRA rank 16 / alpha 32, learning rate 0.001.
XL-Sum English and Hindi each use 2,000 train, 200 validation, and 500 test rows; English replay pool: 500.
Training uses four epochs per language and seed, Hindi batch size 16 and two replay rows. English seed: 0.
Decoding is greedy, at most 256 new tokens, with no repeated trigram. chrF and ROUGE-L follow the project metric implementations (unofficial).


| Phase | Experiment | Outcome | Reports |
|---|---|---|---|
| 1 | English checkpoint and Hindi-only | continue | [MD](/results/phases/phase1/report.md), [CSV](/results/phases/phase1/report.csv), [JSON](/results/phases/phase1/report.json) |
| 2 | Replay revisions | completed | [MD](/results/phases/phase2/report.md), [CSV](/results/phases/phase2/report.csv), [JSON](/results/phases/phase2/report.json) |
| 3 | Maps, block placement, and combination | completed | [MD](/results/phases/phase3/report.md), [CSV](/results/phases/phase3/report.csv), [JSON](/results/phases/phase3/report.json) |
| 4 | New adapter room | skipped | [MD](/results/phases/phase4/report.md), [CSV](/results/phases/phase4/report.csv), [JSON](/results/phases/phase4/report.json) |
| 5 | Adapter shrinking and serving | completed | [MD](/results/phases/phase5/report.md), [CSV](/results/phases/phase5/report.csv), [JSON](/results/phases/phase5/report.json) |
| 6 | Language prefix | skipped | [MD](/results/phases/phase6/report.md), [CSV](/results/phases/phase6/report.csv), [JSON](/results/phases/phase6/report.json) |
| 7 | Cached knowledge distillation | screening_failed | [MD](/results/phases/phase7/report.md), [CSV](/results/phases/phase7/report.csv), [JSON](/results/phases/phase7/report.json) |

## Scores

Plain decoding. Seeds 1,2,3 means a three-seed mean; other rows have one seed.
Phase 3 combined uses seed 1; Phase 4 combined averages seeds 1, 2, and 3.

| Phase | Arm | Seeds | EN val chrF | HI val chrF | EN test chrF | HI test chrF | EN test ROUGE-L | HI test ROUGE-L |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | english | 0 | 29.7365 | 1.6203 | 29.3292 | 1.9424 | 20.8517 | 0.7443 |
| 1 | hindi_only | 1,2,3 | 27.0709 | 26.1857 | 26.7274 | 27.2457 | 18.0620 | 23.6156 |
| 2 | random | 1,2,3 | 28.8275 | 26.0232 | 28.3526 | 27.4790 | 19.8043 | 23.7042 |
| 2 | loss | 1 | 28.4117 | 26.9046 | 28.1203 | 27.9175 | 19.9421 | 23.3918 |
| 2 | drift | 1 | 28.5116 | 25.9697 | 28.5831 | 26.3937 | 19.6201 | 23.3863 |
| 2 | past | 1 | 29.2285 | 27.6702 | 28.6074 | 27.9818 | 20.4792 | 23.7117 |
| 2 | repair_stop | 1 | 29.9572 | 26.5153 | 29.1561 | 27.5327 | 20.3176 | 23.2182 |
| 3 | late | 1 | 29.6905 | 21.0650 | 29.0723 | 21.5728 | 19.7001 | 19.2766 |
| 3 | early_late | 1 | 29.0275 | 26.4041 | 29.7812 | 27.2884 | 20.6179 | 24.7092 |
| 3 | combined | 1 | 30.2737 | 26.3246 | 29.7898 | 27.2244 | 21.2415 | 24.4140 |
| 4 | combined | 1,2,3 | 29.6883 | 26.0905 | 29.6569 | 26.8817 | 21.3816 | 24.2872 |
| 7 | kd_0_25 | 1 | 29.4962 | 26.7247 | 28.6749 | 27.1892 | 20.9095 | 23.8358 |
| 7 | kd_1 | 1 | 29.4819 | 27.5184 | 28.8503 | 28.3967 | 21.3200 | 23.8332 |

Full precision, guarded metrics, seed rows, and resource measurements are in [CSV](/results/report.csv) and [JSON](/results/report.json).
Filter `level=aggregate` or `level=seed` before averaging; do not pool both levels. Aggregate steps/time sum the reported seed runs.

## Phase outcomes

- Phase 1: English validation chrF exceeds lead-1; forgetting and Hindi gain pass the gate.
- Phase 2: Compared random, loss, drift, past-summary, and repair-stop replay. Repair-stop uses drift as its parent and retired no rows.
- Phase 3: Selected repair_stop and early_late. Gradient maps were unstable.
- Phase 4: New adapter room was skipped because the gradient maps were unstable.
- Phase 5: Random: energy rank 11, generation rank 8. Combined: energy rank 10, generation rank 8.
- Phase 6: Validation script-recovery share 0.014674644376430226 is below 0.25; no prefix training ran.
- Phase 7: KD strengths 0.25 and 1.0 did not meet the selection criterion.

## Experiment details

Settings and source records are in [manifest.json](/results/manifest.json) and [metadata.json](/results/sources/drive-2026-11-13/metadata.json).
KD compares strengths 0.25 and 1.0 against random replay. No completed full-English reference scores are available.
