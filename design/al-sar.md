# ALSARPipeline

Owns `src/pipelines/al_sar.py` and `ALSARPipeline`. Keep **similarity-aware retrieval (SAR)** and **adaptive layer freezing (aL)** separately selectable for ablation; the full method contains both.

Source and derivation: [aL-SAR reference](/Papers/aL-SAR-summary.md), §§3.1–3.2, Eqs. 4–8, Algorithm 1, Appendices A.1 and A.18.

## Retrieval contract

Track discounted per-example use `c_i`, class totals `C_y`, and the running class-gradient similarity matrix `S`. Effective frequency and retrieval probabilities follow Eqs. 7–8:

`c_hat_i = c_i + sum_y S[y, y_i] * C_y`

`p_i = softmax_i(-c_hat_i / T)`, with `T > 0`.

Low effective frequency gets greater sampling probability. Similarity is signed; replacing it with absolute cosine, embedding similarity, or raw frequency changes the method. Compute probabilities with stable arithmetic. Eq. 8 defines one softmax over negative effective frequencies; do not implement a second softmax over exponentials from Algorithm 1's inconsistent notation.

Standalone Algorithm 1 inserts arriving records using its Greedy Balancing Sampler, retrieves from that memory, then decays and increments counts. Its per-step retention factor is `1 - B/(k*|M|)`; preserve this ordering, validate its range, and distinguish the factor from the paper's prose decay notation. Sampling replacement and count multiplicity must be explicit. Algorithm 1 independently updates class exposure without explicit subtraction on eviction; preserve that history in the native profile pending pinned-code verification. The hybrid's active-population totals below are a declared adaptation.

The hybrid replaces standalone memory insertion with CSReL and conditions retrieval on TSR's chosen task. This is a named adaptation. It maintains transient usage for observed current records plus retained history; `C_y` sums those active counts, so current training can affect old examples through class similarity. When the current chunk closes, remove counts for records not retained and reconcile totals. Persist the declared update clock from [combined-pipeline.md](/design/combined-pipeline.md).

For each successful hybrid update, decay all active counts by `rho = 1 - B_eff/(k*N_active)`, then increment consumed IDs by their multiplicity and recompute class totals. `B_eff` counts examples across accumulation, not tokens; `N_active` counts unique observed-current plus retained-history records before the update. Require `N_active > 0` and `0 < rho < 1`; never clamp silently. New records start at zero, retained counts survive reselection, and CSReL-only candidates start at zero if retained.

## Adaptive-freezing contract

- Model adapters expose ordered layer groups, forward cost, backward cost per group, layer Fisher-trace estimates, and the final-feature gradient statistic.
- Apply the paper's information-per-computation and batch-freezing criterion (Eqs. 4–5), selecting a prefix boundary. Include the no-freeze option. Fixed layer counts or gradient-magnitude thresholds alone are not aL.
- Before valid Fisher/feature estimates exist, use a full trainable-layer update; a zero denominator also selects no freezing. Reject nonfinite statistics. Legal prefixes leave a trainable path; break BFC ties toward less freezing.
- Periodic full-layer updates refresh Fisher estimates (Algorithm 1 uses every fourth iteration). Preserve that schedule in a native implementation; expose changed schedules as adaptations. Frozen layers cannot supply fresh Fisher estimates for free.
- The `paper_equations` profile uses an EMA of squared final-feature gradient norms (Appendix Eq. 9). Algorithm 1 instead squares an EMA norm; reproducing that estimator requires a distinct profile and pinned-code comparison.
- Update class similarities using the specified per-example gradients from the sampled parameter subset intersected with unfrozen layers. Store the subset identity, EMA state, update cadence, and behavior when no valid gradient pair exists.
- Let the model executor apply the freeze plan and verify actual unchanged parameters. Account for parameter/activation backward work separately; `requires_grad=False` alone does not establish the claimed FLOP saving.

## Language-model boundary

Native SAR uses class labels. A language-model task ID is not automatically a valid replacement, and target vocabulary tokens are not automatically sample classes. Require genuine class annotations or an explicitly documented training-only grouping adaptation before enabling SAR. If none is supplied, reject full aL-SAR; an aL-only ablation can remain available under its own name.

LoRA freezing requires a declared mapping from ordered model blocks to trainable adapter groups. Refreshing aL statistics must not temporarily unfreeze a backbone that the training protocol keeps fixed. Changing from full-model to adapter-only Fisher/gradient statistics is recorded as an adaptation.

Initial aL support requires one microbatch per optimizer update. Accumulation remains available without aL; enabling both requires a named batch-statistic aggregation/freeze-plan extension. Do not choose a new prefix independently for each microbatch.

## State

Checkpoint example/class counts, similarity matrix and parameter subset, Fisher/feature-norm EMAs, layer-cost model, current freeze mask, cadence counters, and RNG.
