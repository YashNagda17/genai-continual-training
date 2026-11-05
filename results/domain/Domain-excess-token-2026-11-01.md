# Domain Grounding Report: Excess-token admission, and the replay multiplier
Date: 2026-11-01 | Domain: constrained stochastic optimization, with token-level loss as the summand | Inputs: results/challenger/Challenge-sum-gradient-energy-2026-11-01.md, results/domain/Domain-sum-gradient-energy-2026-11-01.md, results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md, Papers/OASIS-summary.md, Papers/aL-SAR-summary.md, Papers/TSR-summary.md, src/models/model_causal_lm.py, src/pipelines/oasis.py, src/combined_pipeline.py

## Current Idea

The logged course substitution multiplies OASIS energy by the square of the response length. The four earlier logged substitutions replace that energy by a probe inner product, replace the replay probability by a contest between energy and a cached loss, split the task signature by layer, or replace Fisher freezing by a random layer draw. This report checks those causal claims against the loss the trainer steps on and against the constraint the replay branch is supposed to enforce.

## Domain Interpretation

- Primary domain: stochastic optimization of a token-average cross-entropy, under explicit constraints on past tasks.
- Secondary domain: the geometry of one gradient step. The step moves along the gradient of whatever scalar was differentiated. A score that is not that gradient ranks a different update.
- Why this is the right framing: every stage in the hybrid emits a number, and those numbers are currently in different units. Length rescaling stays inside the wrong average. A token mask changes the scalar that is differentiated. A dual variable changes the weight on a constraint that is actually a loss.

## Claim-to-Theory Map

| Claim | Domain | Theory / Principle | Assumptions | Source | Status |
|---|---|---|---|---|---|
| OASIS energy is the squared norm of the mean response-token head gradient | implementation | I = \|\|g_mean\|\|² with g_mean = (1/n) Σ_t g_t on labels other than −100 | The head path divides by the per-example count for both reduction names | `model_causal_lm.py`; OASIS Eq. 2 in the paper summary | Supported |
| Multiplying that energy by n² ranks the step the trainer takes | optimization | The default step is on the per-example mean, whose parameter gradient is g_mean | `src/example.py` uses `per_example_mean` | `_reduce`; the sum-gradient idea | Incorrect |
| Sum and mean token reductions change the ranking when lengths differ | selection | The two gradients differ by the factor n | Lengths vary inside the batch | OASIS paper summary, informativeness section | Supported as a sensitivity. It was already recorded there |
| A mean over tokens spends update mass on tokens that are already fit | empirical regularity of next-token loss | In Rho-1's math census, 51% of tokens stay low-loss and 26% move from high to low | Their corpus, their reference model, their 1B/7B pretraining | Rho-1, arXiv:2404.07965 | Partially supported. The census is not this stream |
| The highest-loss token is the token to learn | token selection | Rho-1's excess is model loss minus a reference loss, and 11% of tokens stay high for both | A reference trained on curated data. Without it, high loss mixes noise and signal | Rho-1 token census; CTS, arXiv:2505.17827, where a 42% token cut drops GPQA | Unsupported as stated. The supported object is excess over a reference, which this course arm does not have |
| A batch-median mask is a usable stand-in for that excess | approximation | Dropping below-median tokens removes the lower half of the loss inside the batch the scorer already sees | The median of one batch is stable enough to rank examples, and the kept tokens are not the aleatoric 11% | Rho-1 self-reference, reported up to 3.3 points, against over 16 with a curated reference | Partially supported. The gap between the two Rho-1 settings is the warning |
| Replay weight and current energy can compete for the same slots | constrained optimization | A Lagrangian adds λ times a constraint residual to the current loss. Both terms have to be losses, or both have to be gradients | Same units, a declared tolerance, dual ascent with projection onto λ ≥ 0 | PDCL, arXiv:2310.00154, Algorithm 1 | Incorrect for `demand-routed-share` as written. Energy is a squared gradient. A cached loss rise is a loss |
| The optimal replay weight is the sensitivity of current-task value to the forgetting tolerance | duality | ∂P/∂ε_k = −λ_k at the saddle, under convexity in the predictor and a complexity term | Convexity in the function, not in θ. The deviation of λ scales like sqrt(d / n_buffer) | PDCL Theorem statement and the authors' own limit on memory and parametrization | Partially supported as an algorithm. Unsupported as a numerical guarantee at 700M |
| A fixed replay probability is the same object as that multiplier | optimization | A constant α does not move when the constraint residual changes sign | The hybrid draws the branch from `replay_probability` once per update | `tsr.py` `draw_branch`; PDCL §2 | Supported as a mismatch |
| Loss-triggered replay is an adequate dual step | scheduling | MSSR Table 1 places loss-triggered replay within about a point of its full scheduler on 7B–8B models, and its caption says a raw loss threshold chatters | The threshold is on a noisy minibatch loss, with no tolerance and no projection | MSSR, arXiv:2603.09892, Figure 1 and Table 1 | Partially supported. The chatter is why a tolerance and a slow step are the actual dual update |
| An Ebbinghaus decay is the retention law of this model | memory | MSSR uses a parametric strength curve and writes that it does not assume the model follows the human curve | A fitted decay, not a measured synaptic time constant | MSSR §3 | Unsupported. Analogy is suggestive and not formally valid |
| aL's batch rule chooses layers this batch needs | natural gradient / empirical Fisher | Eq. 9 multiplies every layer trace by one shared feature-gradient scale | The chain rule does not make layer gradients proportional across inputs. The paper summary says so | aL-SAR summary, Appendix Eq. 9; `al_sar.py` `batch_freezing` | Partially supported. The batch contributes a difficulty scale. The layer shape is the stored trace |
| TSR can drop one layer whose probe gradients change sign | routing | The actuator is a single teacher weight and one frozen teacher snapshot | Layer-wise routing would be a different system. TreeLoRA is that system, on a different action | TSR summary, Algorithm 2 | Incorrect as an action TSR can take |
| Init LoRA-B gradients compare tasks in the tangent kernel at θ_0 | lazy training | ∇_A L = 0 and ∇_B L ≠ 0 at B = 0, so the signature lives in that subspace | Later tasks are compared in the same coordinates only if every signature uses that initialization. Fine-tuning that leaves the initialization is outside the lazy regime the kernel describes | TSR summary, Eq. 4 | Partially supported. The coordinate system is real. Transfer after many steps is not implied |
| SIREN's negative cosine is a one-step cancellation bug | pool selection | Negative cosine raises the score. The hybrid then trains the admitted pool over later passes, it does not sum the SIREN set as one gradient | Pool coverage wants complementary examples. Cancellation would matter if H were one optimizer batch | OASIS summary Eq. 8; combined-pipeline lifecycle | Unsupported as a bug. It is the paper's coverage term |
| OASIS's sigmoid rate equals the printed threshold's normal integral | calibration | The threshold is defined as the normal expectation of a sigmoid | Scores are approximately standard normal after the z-score. The summary's quadrature does not match the printed thresholds | OASIS summary, Appendix A.7 check | Incorrect for the printed table. The integral and the printed numbers disagree |

## Assumption Audit

| Assumption | Why it matters | If false | Evidence needed |
|---|---|---|---|
| The course metric is per-example semantic accuracy, and token NLL is reported beside it | A token mask can raise exact match and raise NLL on the tokens it ignored | The arm looks like a win on one column and a loss on the other | Both columns, pre-registered |
| Below-median tokens are the already-fit ones | That is the only reason to drop them without a reference model | The mask deletes the skill and keeps noise | The random-mask arm at the same keep rate. If it ties, the median is not the mechanism |
| The head pass already has per-token losses | The mask has to be free of a second model | A second forward would change the cost claim | `_losses` builds those losses and currently reduces them before OASIS sees a vector. Returning a mask is a contract change, not a second network |
| One median per arrival batch is stable | A tiny batch makes the median one token | The mask becomes an arbitrary cut | Log the keep rate. Reject the run if the batch has too few supervised tokens to have a median |
| The replay probe is not the test set | Otherwise λ is fit to the number that will be reported | The dual arm is a leak | The split is fixed before the run and written down |
| ε is set from the probe loss at task commit, not searched on the test metric | PDCL shows a loose ε zeros λ and a tight ε diverges | The sweep recreates a tuned replay probability | One declared rule, then the fixed-probability frontier as the baseline |
| Fisher traces used for freezing were estimated on the current loss only | The freeze can drop layers the current loss is flat in and an old task is not | Retention falls while FLOPs fall | The random-layer control already logged. It stays the test. It is not a new method |
| Class labels exist for SAR | The retrieval formula is a class similarity | On unlabeled instructions the similarity term is undefined | The design contract already rejects full SAR without a grouping. That limit stands |

## Corrections to the Current Idea

- Claim: n² times mean-gradient energy removes a bias and ranks the right examples.
  - Why it is wrong: under `per_example_mean` the parameter step is g_mean. The identity \|\|g_sum\|\|² = n² \|\|g_mean\|\|² ranks the gradient of the summed token loss, which is a step this configuration does not take. The OASIS summary already says the sum and the mean rank different examples when lengths differ. That is a sensitivity of Equation 2, not a new mechanism. LESS Table 13 still found its best MMLU subset short of the pool, so pushing length up can move the metric the wrong way.
  - Corrected domain view: leave the length factor alone until the scalar inside the gradient is the loss whose tokens matter. `sum-gradient-energy` stays logged as a negative control for the token-mask arm.
  - Source: `_batched_head_gradients`; OASIS summary, informativeness; LESS Table 13.
- Claim: a probe inner product is the first-order repair for energy.
  - Why it is weak: the inner product of two mean gradients is the derivative of the probe's mean loss. It inherits the dilution Rho-1 measured, and LESS Table 13 is the published failure of the unnormalized product. GREATS needs a validation set the project must keep off the test split.
  - Corrected domain view: `goal-weighted-admission` remains a later arm, and only on the same token mask as the trainer, with cosine if a product is used at all.
  - Source: LESS Table 13; Rho-1 equation (4).
- Claim: cached loss rises can compete with OASIS energy for training slots.
  - Why it is wrong: a Lagrangian cannot add a squared gradient to a loss. The comparison has no common unit, so the winner is an accident of scale.
  - Corrected domain view: the replay object is a multiplier on a replay loss. Update it by the sign of (probe loss − tolerance), and project it to stay nonnegative. Map it to a probability only after that. This is `dual-replay-weight`. It does not replace CSReL and it does not rank current examples.
  - Source: PDCL Algorithm 1; MSSR Table 1, where loss-triggered replay is already close to a heavier scheduler.
- Claim: PDCL's theorem gives the replay weight for this model.
  - Why it is too strong: the bound they derive needs convexity in the predictor and a sample term on the order of sqrt(d / n). A 700M network is not that convex program, and a small buffer makes the deviation large. Their speech result already loses the sample-selection gain in the small-buffer regime.
  - Corrected domain view: use the dual step as the algorithm. Do not quote the vision curves or the deviation bound as an expected accuracy.
  - Source: PDCL abstract limitation on memory and parametrization; Figure 4 on ε.
- Claim: a per-layer cosine with sign cancellation is a TSR upgrade.
  - Why it is wrong: TSR emits one weight and one teacher. It has no per-layer actuator. aL-SAR does, and its batch term is one scalar on a stored Fisher trace, so handing it mixed-sign layers still does not make the prefix batch-specific in the way the formula suggests.
  - Corrected domain view: `layer-local-signature` does not match the system it would edit. `random-layer-control` remains the test of whether the Fisher prefix beats a matched FLOP draw. Neither is the course change.
  - Source: TSR Algorithm 2; aL-SAR Eq. 9; `al_sar.py`.
- Claim: the human forgetting curve tells the branch probability how to move.
  - Why it is wrong: MSSR fits a curve of that shape and states that the model is not assumed to follow it. A slice result about synapses is a different system.
  - Corrected domain view: the analogy is not formally valid. The empirical baselines are fixed replay and loss-triggered replay.
  - Source: MSSR §3.

## Operating Regime and Failure Regime

- Where a token mask has a reason to work: long responses in which many supervised tokens are already predicted and a minority still carry the cross-entropy. Rho-1's 51% / 26% split is the published picture of that regime, on math pretraining rather than on this stream.
- Where it should fail: short answers, where the median cut keeps one token and the two energies nearly agree. Noisy targets, where the worst tokens are the 11% Rho-1 called aleatoric. Aggressive masks: CTS lost 5 GPQA points at a 42% token cut after gaining 9.1 at a 13% cut.
- Where a dual replay weight has a reason to work: a stream with a real retention constraint, a probe that is stable, and a tolerance taken from the probe loss at commit. PDCL's classification runs are the setting where the saddle-point step was measured.
- Where it should fail: ε far from the unconstrained loss, which zeros or explodes λ. A probe too small for the loss to be a constraint residual rather than noise. MSSR's loss-triggered arm already captures the crude version and sits within about a point of their full method on 7B models. A tie with that arm is a failure of the extra machinery.
- Boundary on the original pipeline: the four scores are a squared last-layer gradient, a cosine in the initialization's LoRA-B subspace, a Fisher-per-FLOP prefix, and a difference of auxiliary losses. Only the last of these is in loss units. The branch probability multiplies none of them. That is the structural limit. Stacking the five logged edits does not produce one Lagrangian.

## Theory-Backed Reformulation

- Refined idea statement: keep the admission gate, SIREN, the writer, and the trainer's per-example mean. Change only the tokens inside the admission gradient. Differentiating the cross-entropy of response tokens above the batch median, and squaring that gradient, is the course arm. In a separate experiment, set the replay probability from one projected multiplier on a test-disjoint probe loss. Do not run the two experiments as one method.
- Mechanism: the mean gradient is the average of easy and hard token gradients. Zeroing the easy half changes the vector SIREN sees and the energy Equation 2 squares. The dual arm never touches that vector. It changes α in the draw that chooses current versus replay, using the residual of a declared loss constraint.
- Why the token arm is the one theory picks first: it is a change of the scalar whose gradient the step actually follows, it uses losses the head computation already forms, and it applies when the example configuration has replay probability 0 and distillation weight 0. The dual arm applies only once history exists.
- What must be true: the median cut must remove tokens that do not move semantic accuracy, the random cut of the same size must not copy the gain, and token NLL must not fall past the margin. For the dual arm, λ must move with the probe residual, and a fixed probability plus a raw loss trigger must not dominate it.

## Falsifiable Claims

1. On this stream, examples admitted by the mean-token energy and by the above-median energy differ as sets whenever response lengths and token losses both vary. If the admitted sets match, the mask is idle and the idea stops.
2. Retained semantic accuracy moves by at least 1 point against mean-token energy, the random-token mask does not match that move, and token NLL stays inside the pre-registered margin. Failure of any one of the three rejects the arm. `sum-gradient-energy` is reported on the same plot and is not required to lose by a theorem.
3. For the separate replay arm, the multiplier's path is monotone in the probe residual, and neither a fixed probability in {0, 0.25, 0.5, 1} nor a loss-threshold trigger beats it on both new-task and retained semantic accuracy. A tie with the loss trigger rejects the dual step as an improvement.

## Experiment Implications

- Token arm: current OASIS energy, above-median energy, n² energy, random mask at the same keep rate. Same admission rate, same model at or below 700M, semantic accuracy and token NLL side by side, length histogram still logged so the length story is visible.
- Replay arm, only after the token arm is scored: fixed probabilities, one loss-threshold trigger, and the projected multiplier. Same token budget, same byte cap, same writer. The probe IDs are fixed before training and excluded from the test report.
- Stress tests: a short-answer slice, where the mask should do nothing. A run with the median replaced by the 90th percentile, which is the CTS-style overshoot. A one-task run, where the dual variable has no constraint and must stay at zero.
- What not to treat as a result: Rho-1's 16-point math gap, PDCL's figure curves, and MSSR's 7B table. They justify the arms. They are not predicted deltas.

## Remaining Unknowns

- Whether this stream's token losses split into an already-fit majority and a useful minority. Rho-1's percentages are from OpenWebMath pretraining.
- How much of SIREN's inclusion-exclusion changes when the gradients are masked. The cosine is on a different vector. That interaction is untested and is still one code path.
- The numerical value of ε that keeps λ finite on this probe. PDCL had to pick it per dataset.
- No model was trained for this report.

## Analogical Domain Transfer

The structural property of the token claim is: an average of heterogeneous summands is a poor gradient if a named subset of the summands is the functional you report.

Rho-1 and CTS are the same domain, so they are evidence rather than analogies. The CFD dual-weighted residual has the formal piece "differentiate the functional you named." It does not derive a median cut, and its Poisson error ratio does not transfer. Status: the warning is the same one already used in the round-1 domain report. It does not add a new factor of improvement.

The structural property of the replay claim is: a constraint residual updates a multiplier, and the multiplier weights the constraint term. That is Lagrangian duality in the same optimization problem, not a cross-domain analogy. PDCL is the source. The assumptions that fail here are convexity in a function class and a useful complexity term at this width. Those failures are stated above. They demote the theorem to an algorithm sketch.

The Ebbinghaus curve fails the formal map. MSSR says the model is not assumed to share it, and a synaptic time constant has no counterpart in a branch probability. Status: suggestive and not formally valid. No biological number is evidence for either arm.
