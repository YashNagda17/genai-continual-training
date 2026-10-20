# Focused search and verification: five linked reproductions

**Date:** 2026-10-02. **Request:** select five actual full-method reproductions, linked as a course-project story, with models at most 1B. **Inputs:** `Idea.md`, `SKILL.md`, `skills/literature-review.md`, `skills/scripts.md`, repository execution guidelines, `results/summary.md`, and the previous 15-paper report. This is a focused refinement of that map rather than a new broad initial sweep.

## Targeted queries

Queries included the following (search results were discovery aids; factual claims use primary sources):

1. `"SmallToLarge" "Pythia-410M" NeurIPS 2024`
2. `"RDS+" "LEAD" instruction data selection`
3. `"transfer-selective-replay" "0.5B"`
4. `"On-Policy Replay" continual fine tuning "self-confidence"`
5. `continual instruction tuning data selection format forgetting 2025 2026`
6. `instruction tuning coreset static selection replay memory small language models learning trajectories`

The combined RDS+/LEAD query returned irrelevant acronym matches; it was not used as evidence. Directly inspected primary pages resolved the relevant methods. Additional targeted checks on small proxies, task-format forgetting, and replay storage were used to test the feasibility and novelty framing. Search is not claimed to establish exhaustive coverage, citation rankings, or priority for the proposed empirical extension.

## Primary evidence inspected

| Work | Paper evidence checked | Repository evidence checked | Decision |
|---|---|---|---|
| RDS+ | [Method and appendix](https://arxiv.org/html/2503.01807): weighted final-layer pooling, query selection, multi-task aggregation; Tulu results and scaling analysis | [Official repository](https://github.com/hamishivi/automated-instruction-selection) | Include as complete scaled static-selection reproduction |
| S2L | [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/97fe251c25b6f99a2a23b330a75b11d4-Paper-Conference.pdf) and [HTML](https://arxiv.org/html/2403.07384): 70M proxy, 410M learner, same-step 30K setting, clustering details, source-wise implementation, checklist | [Official repository](https://github.com/BigML-CS-UCLA/S2L): trajectory/training code and configuration folders | Include; new selection anchor with a native sub-1B learner |
| LEAD | [Published PVLDB PDF](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) and [method](https://arxiv.org/html/2505.07437): IDU, gradient approximation, EXP3, results, publication metadata | [Official README](https://github.com/HKUSTDial/LEAD) and source-directory structure: four-stage pipeline; explicit under-construction notice | Include with highest integration-risk label; do not reduce to IDU-only |
| TSR | [Paper](https://arxiv.org/html/2607.15587): Algorithm 2, shared initialization, replay/distillation split, configuration, overall/plasticity and per-task tables | [Official repository](https://github.com/ymeng3/transfer-selective-replay): native scripts, automatic dataset fetch, author 16GB claim | Include as native 0.5B continual anchor; keep growing-memory and bounded-memory runs separate |
| OPR | [Paper](https://arxiv.org/html/2605.29495): task-boundary rollout, RU/SC, average-token log-probability, budget ablation, confidence failure, historical candidate access | [Official repository](https://github.com/Yancey2024/OnPolicyReplay): both scorer variants, core trainer, preprocessing and launch flow | Include as complete scaled generation-filter-SFT reproduction |

All five were read beyond abstracts with targeted method/results/limitations checks. Repository status means a live README and visible source structure were inspected; it does not mean installation or an end-to-end source audit was completed. No training, code execution from these repositories, or measured memory/time result was obtained.

## Refinement decisions and novelty boundaries

The previous CSReL and SAR choices have attractive native small-vision reproductions, but adding a text adaptation weakens the shared text-instruction story. The previous OASIS choice had an empty paper-linked code repository at the earlier 2026-09-30 check; its current availability was not re-established here, so it was not selected for immediate full reproduction. Existing constraint-preservation and attention-distillation papers remain in the broader report, but they were not preferred to the five complementary data-selection/replay mechanisms. This narrowing does not claim that the five are the best-performing methods in the field.

The format-focused supplementary search found adjacent multimodal work, including [SEFE](https://proceedings.mlr.press/v267/chen25n.html), which was screened at abstract level and excluded from this text-only reproduction shortlist. Gradient coreset/targeted-selection results such as [GIST](https://proceedings.mlr.press/v306/min26a.html) were discovery candidates, not fully assessed alternatives; no claim is made about their small-model reproducibility. The already-reviewed instruction-constraint literature also prevents treating format failure itself as newly discovered. These observations support a restrained empirical framing, not a claim of new selection or replay machinery.

**Verdict:** the five method components are existing algorithms and must be reproduced in full. The proposed linked experiment concerns whether their definitions of utility agree across static training and fixed-access replay, with separate semantic/format measurements and full cost accounting. That exact comparison is a candidate course contribution whose novelty remains unestablished. The final focused report contains five detailed paper analyses and treats native runs, scaled reproductions, and later continual extensions as separate evidence levels.
