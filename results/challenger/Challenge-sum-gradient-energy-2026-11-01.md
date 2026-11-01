# Challenge Report: Length-stable OASIS energy
Date: 2026-11-01 | Venue target(s): ICLR, NeurIPS, ACL | Lit review used: results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md

## Idea Statement (verbatim)

On one instruction stream and a model at or below 700M parameters, replace the OASIS admission energy with the squared norm of the summed response-token last-layer gradient. That quantity is the current mean-gradient energy multiplied by the square of the supervised-token count. Keep SIREN, the admission rate, the writer, and the scoring backward. Claim a change in retained semantic accuracy of at least 1 point against current OASIS energy, at the same scoring FLOPs, with format validity reported separately. Reject the claim if the admitted response-length distribution does not move, if both metrics stay within 1 point, or if length-stratified random admission at the same rate matches the summed-energy arm.

## Evidence Base

- Lit review: [literature-review-pipeline-upgrades-2026-11-01.md](/results/literature-review/literature-review-pipeline-upgrades-2026-11-01.md), including the round-2 section
- Prior challenge, left in place: [Challenge-pipeline-upgrades-2026-11-01.md](/results/challenger/Challenge-pipeline-upgrades-2026-11-01.md)
- Code: `src/models/model_causal_lm.py` `_batched_head_gradients` divides each response-token logit gradient by the supervised count; `src/pipelines/oasis.py` `informativeness` squares the resulting vector. The reduction argument is checked and then unused on this path. Prompt labels are already −100.
- LESS, ICML 2024, Table 13 and the sequence-gradient appendix. Llama-2-7B: vanilla inner product MMLU 47.3 at length 11.3, cosine 50.2 at length 15.1, random 46.5 at length 105.4. TydiQA 52.3 / 19.7, 56.2 / 44.4, 52.7 / 105.4. BBH 37.4 / 19.0, 41.5 / 101.2, 38.9 / 105.4.
- Logged ideas this round refines, and does not replace: `goal-weighted-admission`, `demand-routed-share`, `layer-local-signature`, `random-layer-control` in [optimizations.md](/optimizations.md).

## Gap-Coverage Map

| Lit-review gap | Addressed? | Strength | Notes |
|---|---|---|---|
| Energy is a weaker admission score than a probe inner product | partially | weak as a fix, strong as a prior | The probe product is the LESS inner product that Table 13 shows selecting length 11.3. Cosine is their fix, and it still selects a short MMLU subset. |
| Fixed replay probability | no | none | Left as `demand-routed-share`. A second variable in the same run would make the length result unreadable. |
| Fisher prefix and one task signature | no | none | Left as the random-layer control and the layer-local idea. TreeLoRA's vision gaps are tenths of a point. |
| Rehearsal-free LoRA on instruction targets | no | none | Still untested. SD-LoRA's table is vision class-incremental learning. |
| Score versus the reported metric | directly | weak | Summed energy changes the length scale of the score. It is not semantic accuracy, and LESS Appendix J says cross-entropy and accuracy can diverge on long outputs. |

## Novelty Audit

- Summed-gradient energy. Closest work: LESS cosine normalization of sequence gradients. Classification: incremental. LESS removes magnitude, including the length-driven magnitude, by a cosine. This arm keeps the energy and multiplies it by n². Those are different statistics. Cosine of an energy with itself is 1, so the LESS fix cannot be pasted onto Equation 2.
- Goal-weighted admission. Closest work: GREATS and LESS. Classification: already done in static fine-tuning. The logged wording specifies an inner product. Table 13 is the result for that product. If the arm is ever run, the product has to be a cosine, and the selected lengths have to be logged. It is not the course route.
- Demand-routed share. Closest work: RoD. Classification: incremental. A cached commit-time loss on the same row cancels that row's fixed length, so the length bias is weaker here than in cross-example energy ranking. Rises compared across rows of different lengths can still prefer short rows. It stays a later arm.
- Layer-local signature. Closest work: TreeLoRA. Classification: already done as layer-wise gradient similarity. The extra rule, dropping a layer whose within-task probes change sign, is untested and the published accuracy gaps are tenths of a point.
- Random-layer control. Closest work: LISA. Classification: already done as a single-task schedule. Here it is a control, not a method.

## Impact Audit

- Who benefits: a course project that can change one score and measure it. A reviewer of a methods paper gets a negative or a small substitution, not a new pipeline.
- At what scale: one stream, one model at or below 700M, the admission rate the run already uses. The example configuration admits with `per_example_mean`, replay probability 0, and distillation weight 0, so the length effect is present with no teacher and no replay.
- Counterfactual: without the measurement, the project can spend the course on a probe inner product that LESS already showed prefers short answers, and can report a gain that is a length artifact.
- Improvement threshold crossed: no. The 1-point bar is a pre-registered reject rule. LESS's 2.9-point MMLU gap between vanilla and cosine (47.3 versus 50.2) is a 7B static subset result and is not a prediction for this stream.

## Assumption Audit

| Assumption | Category | Consequence if false | Evidence required |
|---|---|---|---|
| Mean-gradient energy in this code falls with response length on the project data | empirical | The n² rescaling changes nothing visible, and the arm is a no-op | Admitted and rejected length histograms before any accuracy claim |
| Short admissions hurt retained semantic accuracy on this stream | empirical | Length moves and accuracy does not, or accuracy falls | The 1-point rule, format validity separate |
| SIREN's cosine is invariant to a positive per-example scale | mathematical | Rescaling the stored gradient vector would also change redundancy | Cosine(n u, m v) = cosine(u, v) for n, m > 0. The energy term is the part that changes |
| Per-example mean loss is the wrong functional for the reported metric | theoretical | Matching the trainer is then the right score, and summed energy is a biased one | LESS Appendix J is the support, and it is a warning, not a proof for this metric |
| One seed is enough | statistical | A 1-point move can be noise | Three seeds if the course budget allows; otherwise the report says the result is one seed |

## Evaluation Weakness Audit

- Missing baselines: current OASIS energy, length-stratified random at the same rate, and a cosine-only ranker only if someone reintroduces `goal-weighted-admission`. Uniform replay and the rehearsal-free arm stay in the later suite.
- Benchmark issues: a short-answer benchmark can reward the bias LESS measured. Semantic accuracy and format validity have to be split, and the length histogram is a required output.
- Ablation gaps: adding the teacher KL to the score in the same arm confounds the scale change. The example run has distillation weight 0. A positive distillation weight is a different experiment.
- Statistical issues: the round-1 claim of 3 points bundled two substitutions. That claim is withdrawn. One point on one stream is the whole bar, and it is still easy to overfit.
- OOD / stress-test absence: a stream whose answers are all short, or all the same length, cannot show the effect. The run has to include a length range, and the report has to say when it did not.

## Venue Acceptance Check

- ICLR / NeurIPS: a one-line rescaling of a published energy, with a 1-point target on one stream, is below the methods bar. It is a possible negative-result or analysis paper only if the length curve is the contribution and the controls are pre-registered.
- ACL: the instruction-length story is in scope, and the same bar applies. A metric split without a comparison to cosine selection will be read as incomplete, because LESS is the obvious citation.
- Course project: the bar is one identified substitution, a matched FLOP counter, and a reject rule. This idea meets that bar. The four logged ideas, run together, do not.

## Brutal Verdict

1. As a new algorithm this is a scale factor. A reviewer who has read LESS will call cosine the prior art and will be right that cosine was their fix, even though cosine is not this statistic.
2. Table 13's best MMLU subset is still short (length 15.1 versus a random length of 105.4). Undoing the short bias can throw away the examples that actually help.
3. The round-1 3-point claim is not identified. Nothing in GREATS, RoD, or LESS measures this pipeline.
4. Default training in `src/example.py` is `per_example_mean`. Under that reduction the current energy is the squared gradient of the loss the trainer applies. Summed energy is a deliberate mismatch with that loss. If the mismatch is the wrong way, the arm loses by construction.
5. One stream and a 1-point margin will not survive a second dataset or a second seed unless the length histogram moves for a reason the stratified control does not copy.

## Contribution Ladder

Current rung: Weak as a method. Viable as a pre-registered measurement.

What moves it up one rung: a length histogram that moves, a stratified-random arm that loses, and a retained-semantic gap of at least 1 point with format validity not falling, on a model at or below 700M. A second seed keeps it there. A methods venue needs a reason the scale factor is not LESS's cosine, written before the numbers.

## Cross-Domain Pollination

- TRIZ negations: the field assumption that a larger gradient is a more useful example; the assumption that the admission score should be the training loss; the assumption that a probe from another task is a better functional than the batch in hand.
- Conventional path: multiply the existing energy by n². Same backward. Same cosine inside SIREN.
- Blend A (survey sampling — stratify the batch by response length and admit the top energy inside each stratum — combinatorial): holds length fixed by force. It answers a different question, whether energy has any rank left once length is matched. That is the control, not the method.
- Blend B (CFD residual refinement — trust a large residual and cluster there — exploratory): this is today's energy. Short completions are the boundary layer the sampler over-refines. Dual-weighted residuals say to weight by the adjoint of a named functional. They do not derive an n² factor. The analogy stops at "residual magnitude is the wrong weight."
- Blend C (biology, synaptic tagging — a weak trace should not be stored — transformational): the round-1 report already marked this analogy as not formally valid. A long completion is not a weak synapse. This blend is dropped again.
- Selected path: conventional. The blends either repeat the current score, repeat a control, or import a mechanism whose assumptions do not hold. Boden classification: incremental estimator correction, not a new combination.

## Refined High-Value Pivot

- Anchor gap: OASIS energy in this code is ||g_mean||², and g_mean is the average of response-token head gradients. LESS measured that average's norm as anti-correlated with completion length.
- Selected path from blending: the conventional scale change. Blends A and B supply the control and the warning.
- Falsifiable claim: "On one instruction stream and a model at or below 700M, summed-gradient energy changes retained semantic accuracy by at least 1 point against current OASIS energy, at the same admission rate and the same scoring FLOPs, with format validity reported separately. No movement in admitted median length, a gap under 1 point on both metrics, or a tie with length-stratified random admission rejects the claim."
- What to keep: SIREN, CSReL as the writer, the current admission rate, the last-layer backward, and the four logged ideas as later arms.
- What to drop: the bundled 3-point claim, a probe in this run, a teacher KL term in this score, a biological gate, and any stack of GREATS, RoD, and TreeLoRA.
- Experimental plan: current energy, summed energy, length-stratified random. Same admitted count, same scoring FLOPs, semantic accuracy and format validity separate, length histogram required. Three seeds if the budget allows. Distillation weight stays at the run's configured value and is not swept here.
- Expected venue fit: course report. A methods venue only if the result is framed as a length characterization with the reject rule honored.

## Multi-Lens Validation

- Theorist: under `per_example_mean`, ||g_mean||² is the squared per-example gradient of the training loss. Multiplying by n² ranks the gradient of the summed token loss instead. The first-order story justifies the change only if the functional of interest weights tokens rather than examples. That is a hypothesis, and LESS Table 13 is the empirical reason to test it.
- Practitioner: the count is already in the head backward and is discarded. The course cost is a multiply and a histogram. A probe backward is a larger project and inherits the inner-product failure.
- Skeptic: the damaging question is why this is not LESS's cosine. The answer is that Equation 2 is an energy, and cosine deletes the energy. The negative result that kills the contribution is a length histogram that moves while accuracy stays inside 1 point, or a stratified-random tie.
- Janusian check: yes. The useful paper is the curve of retained accuracy against admitted length. A claim that longer admissions are simply better contradicts Table 13, where the best MMLU subset was still short of the pool.
- Revisions made based on lenses: the 3-point bundled claim is withdrawn. The course route is one scale. Probe alignment, demand routing, layer signatures, and the random-layer control stay logged and stay out of this run. KL-matched scoring is named and deferred, because `src/example.py` sets the distillation weight to 0 and adding it would be a second factor.
