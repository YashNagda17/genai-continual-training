# Challenge Report: Excess-token admission
Date: 2026-11-01 | Venue target(s): ICLR, NeurIPS, ACL | Lit review used: results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md

## Idea Statement (verbatim)

On one instruction stream and a model at or below 700M, build the OASIS last-layer gradient only from response tokens whose cross-entropy is above the arrival batch's median, then keep Equation 2, SIREN, and the admission rate. Claim a change of at least 1 point in retained semantic accuracy against the current mean-token energy, at the same admission rate, with token NLL reported separately and held inside a pre-registered margin. Reject the claim if the admitted sets do not change, if a random token mask at the same keep rate matches the median mask, or if token NLL falls past that margin. Leave the replay probability alone in this run.

## Evidence Base

- Lit review round 3: [literature-review-pipeline-upgrades-2026-11-01.md](/results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md)
- Domain correction that forced this rewrite: [Domain-excess-token-2026-11-01.md](/results/domain/Domain-excess-token-2026-11-01.md)
- Prior course claim, withdrawn as the primary arm: [Challenge-sum-gradient-energy-2026-11-01.md](/results/challenger/Challenge-sum-gradient-energy-2026-11-01.md)
- Rho-1, arXiv:2404.07965. Over 16 points on GSM8K and MATH for 1B/7B selective pretraining against full-token training. Token census 26% high-to-low, 51% low-to-low, 11% high-to-high, 12% low-to-high. Self-reference up to 3.3 points.
- PDCL, arXiv:2310.00154. Replay weight is a multiplier updated from a constraint residual. Figure 4: a bad tolerance zeros or explodes it.
- MSSR, arXiv:2603.09892, Table 1. Qwen2.5-7B averages 52.1 no replay, 52.8 fixed, 53.7 loss-triggered, 53.7 accuracy-triggered, 54.5 full method.
- Code: the head gradient averages every supervised response token. The example configuration uses per-example mean, replay probability 0, distillation weight 0.

## Gap-Coverage Map

| Lit-review gap | Addressed? | Strength | Notes |
|---|---|---|---|
| Mean token gradient dilutes already-fit tokens | directly | weak | The mask is the Rho-1 idea without the reference model. Their own self-reference gap shows how much that costs. |
| Length bias of \|\|g_mean\|\|² | partially | weak | A mask changes length only as a side effect. n² remains a control, not the claim. |
| Fixed replay probability | no | none | Moved to `dual-replay-weight` and kept out of this run. PDCL and MSSR already own the scheduler. |
| Fisher prefix and one task signature | no | none | The layer idea has no TSR actuator. The random-layer arm stays a control. |
| Four scores in four units | partially | strong as a critique, weak as a method | This run repairs the scalar inside one score. It does not unify the pipeline. |

## Novelty Audit

- Above-median token energy. Closest work: Rho-1 selective language modeling, then CTS. Classification: incremental. Both select tokens with a trained reference. This arm uses a batch median because a second pretraining run does not fit the project. That is a weaker statistic, and it has to be labeled as one.
- Summed-gradient energy. Closest work: the OASIS summary, which already says sum and mean reductions change rankings. Classification: already noted. Demoted to a control.
- Demand-routed share. Closest work: PDCL's dual step and MSSR's loss trigger. Classification: incorrect as written, because the units do not match, and incremental once rewritten as a multiplier. Not this run.
- Layer-local signature. Closest work: TreeLoRA. Classification: the proposed action does not exist on TSR's output. Drop as a course mechanism.
- Random-layer control. Classification: a baseline. Keep it for any later Fisher claim.

## Impact Audit

- Who benefits: a course measurement of one mask, plus a written account of why the hybrid's four scores are not one objective.
- At what scale: one stream, one model at or below 700M, the admission rate already configured. Replay can stay off, which matches the example configuration.
- Counterfactual: the project spends the course multiplying energy by n², which ranks a step the trainer does not take, or comparing a squared gradient with a loss.
- Improvement threshold crossed: no. Rho-1's 16 points required a curated reference, 1B–7B models, and billions of tokens. The self-reference number, about 3 points, is the honest ceiling to remember, and it is not a prediction.

## Assumption Audit

| Assumption | Category | Consequence if false | Evidence required |
|---|---|---|---|
| Half the response tokens are safe to drop | empirical | The mask deletes the skill | Random mask at the same keep rate |
| Semantic accuracy and token NLL can be traded with a declared margin | evaluation | A semantic gain with a collapsed NLL is a different system | Both metrics in the same table |
| The median is not a function of the test labels | leakage | The arm is invalid | The median uses the training batch only |
| One seed can support a 1-point claim | statistical | The result is noise | Three seeds, or the report says one seed |

## Evaluation Weakness Audit

- Missing baselines: mean-token energy, n² energy, random token mask. A curated reference model is the Rho-1 system and is out of scope. Putting it in would change the budget.
- Benchmark issues: short-answer tasks hide the mask. The stream has to contain long responses, and the report has to show the length and the keep rate.
- Ablation gaps: changing SIREN, the replay probability, and the mask together makes the table unreadable.
- Statistical issues: a 1-point bar on one stream is small next to OASIS's own TRACE gap, which is about 1.2 points over random at 6.25% selection on large models in the paper summary.
- OOD / stress-test absence: the 90th-percentile mask is the overshoot CTS already demonstrated. It belongs in the report if the median arm is positive.

## Venue Acceptance Check

- ICLR / NeurIPS / ACL: a median mask on a published energy, without Rho-1's reference, is below a methods bar. The durable part for a course writeup is the negative controls: n², the random mask, and the unit error in the replay rule.
- The original pipeline as a contribution falls short for a different reason. It composes four published stages. This round does not repair that. It refuses to add a fifth unpublished stack on top.

## Brutal Verdict

1. Rho-1 already trains on the tokens with excess loss, and the version without a curated reference is the small gain. A batch median will be read as that ablation with the reference removed.
2. Eleven percent of Rho-1's tokens stay hard. A median rule can keep those and drop the tokens that were going to be learned.
3. CTS gained 9.1 GPQA points with a mild cut and lost 5 with a hard cut, on a 14B model. The mask has a cliff, and this project has not picked the keep rate by anything but a median.
4. n² energy, probe inner products, and a loss-versus-energy contest do not become a paper by being run in one table.
5. PDCL's multiplier is the right replay object and its guarantee does not survive 700M parameters and a small buffer. Shipping it beside the token mask repeats the two-variable failure of the round-1 claim.

## Contribution Ladder

Current rung: Weak as a method. Viable as one pre-registered mask with three controls.

What moves it up one rung: admitted sets that actually change, a random mask that loses, a semantic gap of at least 1 point, and token NLL inside the margin, on a model at or below 700M. A methods venue still needs a reason this is not Rho-1 with the reference deleted.

## Cross-Domain Pollination

- TRIZ negations: every supervised token deserves equal weight in the admission gradient; a larger gradient norm is a more useful example; a replay coin-flip is a constraint.
- Conventional path: mask the admission gradient to the above-median response tokens. Same backward family. Same gate.
- Blend A (selective pretraining — excess loss over a reference — exploratory): this is Rho-1. It needs a second corpus and a second training run. Out of the course budget. Kept as the named upper bound.
- Blend B (constrained optimization — a multiplier on a past-task loss — combinatorial): this is PDCL's dual step. It repairs the replay probability and does not repair admission. Selected as a later arm, not as part of this claim.
- Blend C (forgetting curve — schedule replay by a human retention function — transformational): MSSR fits the curve and declines the biological claim. The blend fails the formal map. Dropped.
- Selected path: conventional mask. Boden classification: incremental restriction of an existing gradient. The blends are the published neighbors, and running them would reimplement those papers.

## Refined High-Value Pivot

- Anchor gap: the admission gradient is the mean of every response token, so tokens the model has already fit vote in the ranking.
- Selected path from blending: the conventional mask. Blend A is the ceiling. Blend B is the separate replay correction.
- Falsifiable claim: "On one instruction stream and a model at or below 700M, above-median token energy changes retained semantic accuracy by at least 1 point against mean-token OASIS energy, at the same admission rate, with token NLL inside a pre-registered margin. Unchanged admitted sets, a tie with a random mask of the same keep rate, or an NLL drop past the margin rejects the claim."
- What to keep: SIREN, CSReL, the current admission rate, and the logged ideas as named controls or later arms.
- What to drop: n² as the primary claim, a probe in this run, a teacher KL term in this score, a per-layer TSR route, and any biological schedule.
- Experimental plan: mean energy, median-token energy, n² energy, random mask. Semantic accuracy and token NLL. One stream. Replay probability held at the configuration's value.
- Expected venue fit: course report. The pipeline critique is part of the writeup. It is not a replacement algorithm for all four stages.

## Multi-Lens Validation

- Theorist: the gradient that enters Equation 2 has to be the gradient of a declared scalar. The median mask declares "cross-entropy of the upper half of this batch." That is a real scalar. It is not Rho-1's excess over a reference, and it is not exact match.
- Practitioner: the token losses exist inside the head loss before they are averaged. The course edit is a mask on that reduction. A reference pretraining run is a different project.
- Skeptic: the damaging question is why the random mask would lose. If it does not, the median is a keep rate. The negative result that kills the contribution is a tie with that mask, or a semantic gain bought by a large NLL regression.
- Janusian check: yes. The useful outcome is the tradeoff between semantic accuracy and token NLL as the keep rate moves. A claim that dropping easy tokens is uniformly better contradicts CTS's hard-compression drop and Rho-1's aleatoric tokens.
- Revisions made based on lenses: `sum-gradient-energy` is no longer the course claim. `demand-routed-share` is corrected into `dual-replay-weight` and excluded from this experiment. `layer-local-signature` is dropped as a mechanism because TSR cannot act on one layer.
