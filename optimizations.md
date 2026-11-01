# Algorithmic optimization ideas

Append distinct algorithmic candidates in six lines; reference existing entries instead of duplicating them. Exclude routine engineering work. Logging an idea neither establishes novelty nor authorizes changing a baseline. Examples/tests must obey project budgets; cite the relevant paper section or code.

```text
Idea (YYYY-MM-DD, short ID): <title>; status: untested.
Observation: <specific bottleneck or failure and its paper/code context>.
Mechanism: <algorithmic change and why it might help>.
Example: <concrete small input/scenario and the behavior to try later>.
Test: <baseline, matched budget, metric, and result that would reject the idea>.
Risk / novelty: <assumption or failure mode>; novelty unverified; <known related work, if any>.
```

Idea (2026-11-01, goal-weighted-admission): score an OASIS candidate by its last-layer gradient's inner product with a declared probe gradient, instead of by gradient energy; status: untested.
Observation: OASIS ranks current examples by relative last-layer energy and redundancy ([OASIS summary](/Papers/OASIS-summary.md)); GREATS (NeurIPS 2024) beats GradNorm by 3.8 points on its nine-subject MMLU slice, and dual-weighted residuals weight a residual by a functional adjoint rather than by residual size (Govoeyi and Richter, arXiv:2604.01835).
Mechanism: keep the admission gate, replace the energy term with a ghost inner product against a fixed, test-disjoint retention probe plus a current-task probe, and leave SIREN redundancy unchanged so the substitution is one variable.
Example: a batch of four instructions in which two have large gradients on template tokens and one has a small gradient aligned with a held-out old-task probe; the aligned example should outrank the template-energy examples.
Test: matched admitted count and matched scoring FLOPs against OASIS energy on one stream at or below 700M; reject if retained semantic accuracy moves by under 1 point, or if a noise probe matches the real probe.
Risk / novelty: the probe can be a disguised validation leak, and Adam's normalizer can make the SGD ghost product the wrong direction; novelty unverified; GREATS and LESS already select by gradient alignment in static fine-tuning.

Idea (2026-11-01, demand-routed-share): set the replay-versus-current slot fraction by comparing current reducible loss with the rise from a cached commit-time loss, instead of TSR's fixed branch probability; status: untested.
Observation: the hybrid draws a replay branch at a configured probability ([combined pipeline](/design/combined-pipeline.md)); RoD (arXiv:2609.40089) lets adaptation and replay compete and, at matched legal adaptation, reports validation-loss forgetting of 0.071 versus 0.125 for a 10% mixture.
Mechanism: cache the scalar training loss of each retained row at memory commit, rescore it with the current learner, and let rows with a positive rise compete with current-batch scores for the same slots. Do not train RoD's adaptation specialist.
Example: after a task that leaves old losses flat, the cached rises are ~0 and the next update should be almost all current data; after a task that doubles two retained losses, those two rows should enter the batch.
Test: frontier against fixed probabilities {0, 0.25, 0.5, 1} at the same token budget, byte cap, and writer; reject if any fixed probability dominates the demand rule on both new-task and retained semantic accuracy across three seeds.
Risk / novelty: a cached scalar is not RoD's reference-model loss and can track template drift rather than forgetting; novelty unverified; RoD and RHO-LOSS already allocate by reducible loss when a reference model exists.

Idea (2026-11-01, layer-local-signature): replace TSR's single probe gradient with one cosine per layer, and do not route a layer whose within-task probe gradients change sign; status: untested.
Observation: TSR's own limitation is that an average signature can cancel opposing subskills ([TSR summary](/Papers/TSR-summary.md)); TreeLoRA (ICML 2025) routes from layer-wise gradient similarity, with only tenths-of-a-point vision gains over HiDe-LoRA.
Mechanism: compute the probe gradient separately on each LoRA group, drop groups with mixed signs from the teacher routing weight, and keep aL-SAR's freeze rule untouched.
Example: a task whose layer-0 probe agrees across items and whose layer-8 probe has two opposite clusters; layer 8 should receive no transfer weight from that task.
Test: matched probe FLOPs against the current single-vector TSR on the same stream; reject if retained semantic accuracy and routing entropy do not both move by a pre-registered margin, or if the mixed-sign layers were already unused by aL-SAR.
Risk / novelty: short task lists may not need a tree or a bandit, and sign cancellation in probe gradients may be template noise; novelty unverified; TreeLoRA, ASO-LoRA, and TSR already use gradient geometry to route.

Idea (2026-11-01, random-layer-control): before crediting aL-SAR's Fisher-per-cost prefix, unfreeze a random subset of layers with the same expected FLOPs and always leave the end layers trainable; status: untested.
Observation: aL-SAR freezes a prefix from Fisher information per cost ([aL-SAR summary](/Papers/aL-SAR-summary.md)); LISA (NeurIPS 2024) randomly samples middle layers and raises LLaMA-2-7B MT-Bench from LoRA 4.45 to 4.94 on a single fine-tune, which is not a forgetting result.
Mechanism: one control arm. No change to retrieval. The sampler draws the same expected number of trainable layers as the Fisher rule and holds the draw for the same number of steps.
Example: if the Fisher rule trains 4 of 16 layers on a batch, the control also trains 4 of 16, chosen without Fisher, for the same step block.
Test: same token budget and same unfrozen-FLOP counter; reject the Fisher rule if retained and new-task semantic accuracy both stay within 1 point of the random arm across three seeds.
Risk / novelty: a matched FLOP counter can lie if frozen layers still receive a full backward; this arm is a baseline, not a candidate method; novelty unverified; LISA is the published single-task version of the random schedule.

Idea (2026-11-01, sum-gradient-energy): rank an OASIS candidate by the squared norm of its summed response-token last-layer gradient, equal to the current mean-gradient energy times the square of the supervised-token count; status: untested.
Observation: admission squares the last-layer gradient of the per-example mean of response-token cross-entropy (`src/models/model_causal_lm.py` `_batched_head_gradients`, `src/pipelines/oasis.py` `informativeness`). That average's norm shrinks as the completion grows. LESS Table 13 (ICML 2024, arXiv:2402.04333) reports Llama-2-7B MMLU 47.3 at selected length 11.3 for the vanilla inner product, 50.2 at length 15.1 for cosine-normalized features, and 46.5 at length 105.4 for random.
Mechanism: keep SIREN's cosine and the admission rate. Score Equation 2 on the gradient of the summed token loss, which multiplies today's energy by n². The supervised count is already computed and discarded. Add no probe and no teacher pass. `goal-weighted-admission` still dots the length-biased mean gradient; this idea changes only the energy's length scale.
Example: two responses with the same per-token gradient direction, lengths 8 and 64. Today's energy ranks the short one higher whenever the token gradients fail to add perfectly. Summed energy ranks the long one higher when those gradients accumulate.
Test: one instruction stream, model at or below 700M, matched admission rate and matched last-layer scoring FLOPs, against current OASIS energy and against length-stratified random admission at the same rate. Reject if the admitted median response length does not move, if retained semantic accuracy and format validity both stay within 1 point of current OASIS, or if the stratified-random arm matches summed energy.
Risk / novelty: cosine-normalized LESS still won MMLU with a short subset, so removing the short bias can lower accuracy; prompt tokens are already masked here, and cosine-versus-dot-product is already LESS; novelty unverified.
