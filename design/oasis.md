# OASISPipeline

> **Current plan:** not used by the [summarization plan](/design/summarization-plan.md). This contract is unchanged and applies to earlier work only.

Owns `src/pipelines/oasis.py` and `OASISPipeline`: **current-arrival admission** in the hybrid. This is a bounded-access adaptation; native Appendix A.2 experiments sample training batches from an unbounded historical memory, whose access/storage must be declared.

Source and derivation: [OASIS reference](/Papers/OASIS-summary.md), §§3.3–3.4, Eqs. 2–8, Appendices A.7 and A.19.

## Algorithm contract

For each incoming batch, before training on that batch:

1. Obtain the specified **per-example last-layer parameter gradients** `g_i` and informativeness `I_i = ||g_i||²` (Eq. 2). Do not use a batch-averaged gradient, hidden-state similarity, or loss alone.
2. Apply SIREN's iterative selection order and redundancy adjustment. Recompute adjusted informativeness from the original scores and the current set `H`, using signed gradient cosine and the higher-order inclusion-exclusion terms in Eq. 8. `H` is the temporary ordering set, not the final admitted examples.
3. Normalize adjusted scores with the **previous** state: `z_i = (I_tilde_i - mu_prev) / sqrt(var_prev)` (Eq. 4). Initial statistics and a numerical variance floor are explicit configuration and recorded implementation conventions.
4. Compute `p_i = sigmoid(z_i - I_T)` and perform the specified probabilistic admission. Return IDs, aligned scores/probabilities, and cost; never replace it with fixed top-k selection.
5. Update running mean/variance once using the **raw incoming batch's** mean informativeness, including rejected examples: `mu_next = beta * mean(I) + (1-beta) * mu_prev`; `var_next = beta * (mean(I)-mu_prev)^2 + (1-beta) * var_prev` (Eq. 3). Do not substitute ordinary within-batch sample variance.

Retain statistics across task boundaries. A distribution change does not implicitly reset the selector. An empty admitted batch is valid: update its observed-data statistics, skip its current-data learner update, and report the event.

## Explicit source ambiguities

The `paper_equations` profile resolves paper ambiguities as follows:

| Issue | Contract |
|---|---|
| Algorithm 1 uses `mu_t`/`sigma_t` while Eq. 4 specifies prior statistics | Use the state before incorporating the current batch, then update it after admission. |
| Algorithm 1's pairwise loop can be read as overwriting rather than summing | Use the full Eq. 8 expression from original scores; do not repeatedly overwrite pairwise contributions. |
| Text says Bernoulli sampling; Eq. 6/Algorithm 1 draw one threshold | Use one uniform threshold per batch shared across its samples, as Algorithm 1 states. Independent draws are a distinct sampling variant. |
| Appendix A.7 gives a normal-model target-rate equation and tabulated thresholds | Numerically solve the stated expectation for the requested rate and test the solution; do not treat the listed numbers as a universal calibration. |
| Appendix A.1 rescales batch-mean variance by batch size and uses a different EMA coefficient convention | Follow main Eqs. 3–4 in this profile. Do not silently import the appendix's rescaled variance or coefficient meaning; reconcile against pinned code before reproduction claims. |

A differing released-code profile needs a separate method identifier and tests.

## Cost and numerical boundaries

- SIREN's literal higher-order subset sum can grow exponentially with batch size. Implement a tiny exact reference and enforce a feasible configured selection-batch limit. Any truncation, sketch, or recursive approximation requires a documented derivation or an adaptation label; silently dropping Eq. 8 terms is prohibited.
- Last-layer gradient computation for one arrival is one decoder forward and one decoder backward for the whole batch, still returning one gradient vector per example. Declare the exact layer/parameter view. A frozen/tied language-model head requires capability checks and cost accounting; do not unfreeze the learner merely to score examples.
- Define zero-norm cosine as zero contribution and preserve negative adjusted scores; do not clamp them without a named variant. Reject nonfinite input statistics. Require positive normalization scale after the declared numerical floor.
- The normal approximation is a modeling assumption, not a guarantee under shifting data. A target selection rate is an expected rate; record the realized rate and do not enforce an exact per-batch quota. Explicit rates zero/one may use deterministic none/all controls.
- Keep `mu`, `var`, arrival count, threshold/calibration, parameter-view fingerprint, and RNG in checkpoints. Temporary gradients and `H` are released after the decision.

