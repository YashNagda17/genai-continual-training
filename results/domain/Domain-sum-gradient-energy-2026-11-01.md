# Domain Grounding Report: Length-stable OASIS energy
Date: 2026-11-01 | Domain: statistical estimation of a sequence gradient, with a CFD residual analogy | Inputs: results/challenger/Challenge-sum-gradient-energy-2026-11-01.md, results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md, src/models/model_causal_lm.py, src/pipelines/oasis.py

## Current Idea

OASIS admits current examples by the squared norm of a last-layer gradient. In this repository that gradient is the gradient of the per-example mean of response-token cross-entropy. The course substitution multiplies that energy by n², the square of the supervised-token count, and leaves SIREN's cosine, the admission rate, and the scoring backward as they are.

## Domain Interpretation

- Primary domain: estimation. The score is a statistic of a sequence of token gradients, and the sequence length is a covariate of that statistic.
- Secondary domain: goal-oriented approximation. A residual norm is a valid refinement indicator only for the functional whose residual it is.
- Why this is the right framing: the code's identity ||g_mean||² = ||g_sum||² / n² is exact. Whether the short sequence then wins is an empirical property of how g_sum grows with n. LESS measured that growth for instruction gradients. Biology does not determine it.

## Claim-to-Theory Map

| Claim | Domain | Theory / Principle | Assumptions | Source | Status |
|---|---|---|---|---|---|
| The head gradient used for admission is the mean of response-token gradients | implementation | g_mean = (1/n) Σ_t g_t, with prompt positions excluded by label −100 | The tied and untied head paths both divide by the supervised count, which they do | `model_causal_lm.py` `_batched_head_gradients` | Supported |
| Squared energy of that mean equals summed-gradient energy over n² | algebra | ||c v||² = c² ||v||² for the positive scalar c = 1/n | n ≥ 1, which the code requires | same function; OASIS `informativeness` | Supported |
| That norm falls with completion length | instruction tuning | An average of token gradients concentrates, and its norm is anti-correlated with length | Token gradients behave like a shared direction plus noise that averages down. LESS says the tokens are not i.i.d., so the noise model is incomplete | LESS, sequence-gradient section and Table 13 | Partial |
| Inner-product selection on the unnormalized gradient hurts accuracy | measurement | Table 13, Llama-2-7B | Static 5% subset, 7B, Adam features, not this admission gate | LESS Table 13 | Supported in that paper, Unknown here |
| Under `per_example_mean`, current energy matches the trainer | optimization | The batch loss is the mean of per-example mean losses, so each example's pull is g_mean | The run uses `per_example_mean`. `src/example.py` does | `model_causal_lm.py` `_reduce`; `example.py` | Supported for that reduction |
| Under `per_token_mean`, the example's pull scales with n ||g_mean|| | optimization | The batch gradient weights each example by its token count | Candidates are scored alone, and a lone example has the same mean loss under both reduction names. The code comment says this | `_batched_head_gradients` docstring | Supported as a batch-level fact, Partial at admission time |
| n² energy is the adjoint of semantic accuracy | goal-oriented estimation | The dual-weighted residual weights the residual by the adjoint of a named functional | The functional is differentiable and the adjoint is that functional's derivative | Govoeyi and Richter, arXiv:2604.01835; LESS Appendix J | Incorrect |
| A length-stratified random arm isolates the gradient from the length covariate | survey sampling | Holding a confounder fixed leaves the remaining ranker as the thing under test | Strata are wide enough to rank inside, and the admission rate is matched | standard stratified sampling | Partial |
| Synaptic tagging says long traces should be preferred | biology | A weak pathway decays unless a strong one tags it | Slice physiology, not token averages | Chong, Ang, Sajikumar, Communications Biology, 2025 | Unsupported. Analogy is suggestive and not formally valid |

## Assumption Audit

| Assumption | Why it matters | If false | Evidence needed |
|---|---|---|---|
| Response tokens, not prompt tokens, enter g | A prompt mask would be a different project | The length effect could be template length | Labels of prompt positions are −100, and `_losses` drops those positions. Already true in this code |
| n is available without a second backward | The performance claim is "same FLOPs" | The arm is no longer free | `counts` is computed inside the head backward and not returned. Returning it is a contract change, not a new pass |
| SIREN cosine ignores a positive scale | The substitution stays one variable | Redundancy and energy would move together | Identity cos(n u, m v) = cos(u, v). Check it on one batch before trusting the arm |
| The project metric cares about long answers | Otherwise the short bias can be the right bias | Summed energy lowers the metric that matters | Split semantic accuracy by answer-length bucket |
| Distillation weight is zero, or is held fixed | KL is another functional | A KL-on run attributes a KL change to the scale | `src/example.py` uses distillation weight 0. A nonzero weight is out of this arm |
| Three seeds | A 1-point gap is small | The result is a single draw | Say the seed count in the report. Do not invent a significance test |

## Corrections to the Current Idea

- Claim: probe-aligned admission is the first-order correction to energy.
  - Why it is weak: the first-order expansion ranks the update of the functional that was differentiated. A held-out probe is a third functional. The inner product of mean gradients is the statistic LESS Table 13 shows selecting MMLU length 11.3 and scoring 47.3, against 50.2 for the cosine version.
  - Corrected view: `goal-weighted-admission` remains a later arm, and only with a cosine and a length log. It is not the course route.
  - Source: LESS Table 13; GREATS, which also needs a validation set and states that its ghost product is an SGD derivation.
- Claim: matching the training loss removes the length bias.
  - Why it is weak: with `per_example_mean`, the trainer's per-example gradient is g_mean. The length bias is inside that loss. A faithful score reproduces it.
  - Corrected view: summed energy is a change of functional, from the mean token loss to the summed token loss. It is justified as a test against the LESS failure mode, not as "the true gradient."
  - Source: `_reduce` and the head-gradient docstring; LESS Appendix J on loss versus accuracy.
- Claim: the reduction flag already chooses mean versus sum for OASIS.
  - Why it is wrong: `_batched_head_gradients` accepts `per_example_mean` and `per_token_mean` and divides by the per-example count in both cases.
  - Corrected view: the flag is a no-op for the admission vector. The training loss does honor it.
  - Source: `model_causal_lm.py` around the head-gradient scale.
- Claim: adding the era-teacher KL to the score is required for objective match.
  - Why it is a different claim: the trainer adds response-masked KL only when `kd_weight` is positive and a teacher exists. The example configuration sets that weight to 0. Adding KL scores every arrival, including rejects, and costs a teacher forward the admission step does not pay today.
  - Corrected view: leave KL out of this arm. Log it as a deferred second variable.
  - Source: `combined_pipeline.py` `next_update`, `tsr.py` `objective`, `example.py`.
- Claim: awake replay or synaptic tagging picks the length weight.
  - Why it is wrong: the structural map from a token count to a synaptic tag is metaphorical. The slice result does not imply a factor of n².
  - Corrected view: the analogy is not formally valid. It is not evidence for this arm.
  - Source: round-1 domain report; Chong, Ang, and Sajikumar, 2025.

## Operating Regime and Failure Regime

- Where the identity holds: any batch with at least one supervised response token. The n² relation does not depend on the data distribution.
- Where the accuracy claim can hold: instruction data where completions vary in length and the reported metric penalizes a model that only learned the short ones. LESS's TydiQA and BBH rows are the published hint. TydiQA cosine selection reached length 44.4 and 56.2, against vanilla 19.7 and 52.3. BBH cosine reached length 101.2 and 41.5, against vanilla 19.0 and 37.4.
- Where it should fail: short-answer tasks, of which MMLU is the published example. Cosine there won at length 15.1, far below the random length 105.4. A stream with almost equal lengths. A run whose distillation weight is large enough that the CE gradient is the wrong direction for the step actually taken.
- Boundary conditions: n = 1 makes the two energies equal. A tied embedding adds the input-embedding rows into the same vector; the 1/n factor still comes from the logit-gradient scale. Adam's second moment, which LESS models and this energy does not, can reorder the same examples.

## Theory-Backed Reformulation

- Refined idea statement: keep the admission gate. Replace ||g_mean||² with ||g_sum||² = n² ||g_mean||². Measure the admitted length distribution. Adopt the change only if retained semantic accuracy moves by at least 1 point and format validity does not fall, against current energy and against length-stratified random admission, at matched rate and matched scoring FLOPs.
- Mechanism: the summed vector is the gradient of the sum of response-token losses. Its squared norm does not shrink by n² when a completion is long. SIREN's cosine is unchanged by that positive scale.
- Why the measurement is worth running: LESS showed a several-point accuracy gap, on a different setup, between a length-collapsed inner product and a magnitude-normalized score. This code's energy is the squared norm of that same kind of average.
- What must be true: token gradients must accumulate slower than n on this data, the metric must care, and the stratified control must not copy the gain.

## Falsifiable Claims

1. On the project stream, the median supervised length of examples admitted by ||g_mean||² is shorter than the median length of arrivals. If it is not, stop. The n² factor has nothing to correct.
2. ||g_sum||² moves that admitted median toward the arrival median at the same admission rate. If it does not, the implementation does not match the identity above.
3. Retained semantic accuracy moves by at least 1 point against current OASIS, format validity does not fall, and length-stratified random admission does not match the summed-energy arm. Failure of any one of those three rejects adoption. The result can still be reported as a length curve.

## Experiment Implications

- Datasets: one instruction stream whose responses are not all the same length. Report the length histogram of arrivals, admissions, and rejects.
- Baselines the theory requires: current ||g_mean||², ||g_sum||², and length-stratified random at the same rate. Cosine probe alignment is the LESS baseline for a later arm, not a third factor in this one.
- Stress tests: a short-answer slice and a long-answer slice of the same stream, scored separately. A one-token-response subset, where the two energies coincide. A repeat with the configured distillation weight left at 0, which is the example default.
- Models: at or below 700M. The LESS numbers are Llama-2-7B and do not transfer as a predicted margin.
- Seeds: three if the course can pay for them. One seed is a pilot and has to be labeled as one seed.

## Remaining Unknowns

- Whether this stream's mean-gradient norm is anti-correlated with length. LESS's figures are for their warmup checkpoints and Adam features.
- Whether SIREN's energy corrections, which mix raw energies across a selected subset, undo part of the n² rescaling inside a batch. The cosine term is invariant. The energy term is not, and the net rank change is untested.
- How large a distillation weight has to be before the CE-only score ranks a different set from the CE-plus-KL score. Unknown at weight 0, which is the example configuration.
- No run was trained for this report.

## Analogical Domain Transfer

The structural property is: a nonnegative residual magnitude that is a decreasing function of a nuisance covariate will over-sample the low end of that covariate.

CFD dual-weighted residuals have the formal piece "do not refine by residual magnitude unless the residual is the functional you report." Govoeyi and Richter use that identity on a Poisson problem and report about a 10× functional-error drop on one oscillatory case. The assumptions that do not carry over are a linear(ized) PDE residual, an adjoint derived from a named differentiable functional, and a mesh. Instruction cross-entropy has no Poisson energy, and semantic accuracy has no adjoint in this code. The translated warning is the one that survives: ||g_mean||² is the residual of the mean token loss, and it is confounded with n. The 10× figure is not an expected accuracy gain. Status of the analogy: partial for the warning, not formally valid as a derivation of n².

Survey sampling has the formal piece for the control arm. Stratifying on length and drawing at the same rate inside strata holds the covariate fixed. The assumption that carries is that length is observed before the draw, which it is. The assumption that may fail is thin strata, where "top of the stratum" is one example. Status: partial, and only for the control.

Biology was checked again and does not meet the formal map. A completion length is not a synaptic tag, a replay rate, or a multi-timescale plasticity state. No biological number is evidence for this substitution.
