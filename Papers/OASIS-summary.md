# OASIS — Online Sample Selection for Continual Instruction Tuning

**Authors:** Minjae Lee, Minhyuk Seo, Tingyu Qu, Tinne Tuytelaars, Jonghyun Choi. **Venue:** ACL 2026. **Reference date:** 2026-10-20.
**Sources:** [Published paper](https://aclanthology.org/2026.acl-long.158/), [local PDF](/Papers/OASIS.pdf), [author repository](https://github.com/snumprlab/oasis), [implementation contract](/design/oasis.md).
**Local SHA-256:** `063a46994924a62090997ce4b49d0e883bd1c0007590c496f6e8fa05d6267f30`. `2026.acl-long.158.pdf` is an identical copy.

## Problem and reported results

OASIS selects training candidates while the learner and data distribution change. ORIS normalizes gradient informativeness against stream statistics; SIREN discounts intra-batch overlap without repeated learner updates. Admission is probabilistic.

At 6.25% selection, Table 3 reports TRACE final accuracy of **53.48 ± 0.17** for OASIS, **52.27 ± 0.03** for random selection, and **57.73 ± 0.54** for full data. At 25% selection, Table 2 reports MICVIT **77.95 ± 0.93**, versus **79.66 ± 0.43** with full data. Results use three seeds and mainly large models; Appendix A.8 also reports a model named Qwen-VL-2.5-0.5B, whose full parameter count still needs verification for our ceiling. [Paper, Tables 2–3 and §4](https://aclanthology.org/2026.acl-long.158.pdf).

## Problem, access model, and notation

For a candidate batch \(B_t=\{(x_i,y_i)\}_{i=1}^{N_B}\), let \(\theta_L\) denote the specified last-layer parameters and

$$
g_i=\nabla_{\theta_L}\ell(f_\theta(x_i),y_i).
$$

The candidate's supervised target is available when scoring. This is not label-free selection. For language modeling, the definition of an example, response-token mask, token reduction, and last-layer parameter view must be fixed before comparing gradient magnitudes.

Two settings must remain distinct:

| Setting | Candidate access | Consequence |
|---|---|---|
| Published experimental protocol, Appendix A.2 | All encountered samples retained; training batches drawn from this memory | Historical sampling and storage contribute to the observed result |
| Project hybrid | Current incoming records scored once per declared arrival; historical replay supplied separately | A bounded-access adaptation; it cannot inherit the paper's numerical claims automatically |

Even in the native setting, SIREN's temporary set \(H\) is an ordering device inside a candidate batch. It is neither the persistent replay memory nor the final admitted subset.

## Formulas and mechanism

### 1. Informativeness — Eq. 2

$$
I_i=\operatorname{tr}(g_i g_i^\top)=\|g_i\|_2^2.
$$

This is a last-layer gradient-energy surrogate. It is not a direct measurement of information gain, and an empirical per-example gradient outer product is not automatically the full expected Fisher matrix. The exact loss reduction matters: sum-over-token and mean-over-token gradients give different rankings when sequence lengths differ.

### 2. Running reference statistics — Eq. 3

Let \(\bar I_t=N_B^{-1}\sum_{i\in B_t}I_i\), and let \(v_t\) denote the paper's running squared scale:

$$
\mu_t=\beta\bar I_t+(1-\beta)\mu_{t-1},
\qquad
v_t=\beta(\bar I_t-\mu_{t-1})^2+(1-\beta)v_{t-1}.
$$

The second expression tracks fluctuations of **batch means**, not ordinary within-batch sample variance. The main equations use \(\beta\) as the coefficient of the newest observation. Update these statistics from raw informativeness of all candidates in the scored batch, including rejected candidates; selected-only updates would bias the reference population.

### 3. SIREN redundancy adjustment — Eqs. 7–8

Iteratively choose the most informative remaining candidate and add it to \(H\). Before choosing the next candidate, adjust the remaining candidates using the original scores and current \(H\). The pairwise expression is

$$
\widetilde I_i=I_i-\sum_{h\in H}\cos(g_i,g_h)I_h.
$$

The paper extends this with inclusion–exclusion terms:

$$
\boxed{
\widetilde I_i=I_i-\sum_{h\in H}\cos(g_i,g_h)I_h
+\sum_{\substack{U\subseteq H\\|U|\geq2}}
(-1)^{|U|}\cos(g_i,\bar g_U)\bar I_U
}
$$

where

$$
\bar g_U=\frac{1}{|U|}\sum_{u\in U}g_u,
\qquad
\bar I_U=\frac{1}{|U|}\sum_{u\in U}I_u.
$$

Positive cosine reduces the pairwise score; negative cosine can increase it. Example: \(I_i=4\), \(I_h=3\), and cosine \(0.8\) give \(1.6\); cosine \(-0.5\) gives \(5.5\). Clamping signed similarity or negative adjusted scores changes the method.

The full formula is a modeling approximation to information overlap, not a consequence of cosine similarity being a literal set-intersection measure. Do not assume that inclusion–exclusion guarantees nonnegative or calibrated information estimates.

### 4. Relative informativeness and admission — Eqs. 4–6

After SIREN, use statistics from **before** the current batch:

$$
z_i=\frac{\widetilde I_i-\mu_{t-1}}{\sqrt{v_{t-1}}},
\qquad p_i=\operatorname{sigmoid}(z_i-I_T).
$$

The project's `paper_equations` profile follows Eq. 6 and Algorithm 1 by drawing one \(r\sim U(0,1)\) for the batch and admitting \(i\) if \(p_i>r\). Each sample has marginal admission probability \(p_i\), but admissions are correlated because the random threshold is shared. Independent per-example Bernoulli draws preserve these marginals while changing batch-size variance and joint selection; they are a separate variant.

Initialize statistics and define a numerical variance floor explicitly. An empty admitted batch is valid. In our current-arrival profile, update the raw-arrival statistics and skip the current learner update; historical replay follows the separately declared orchestration policy.

### 5. Target-rate calibration — Appendix A.7

Under the assumed \(Z\sim\mathcal N(0,1)\), the threshold is related to a target rate \(q\) by

$$
q=f(I_T)=
\int_{-\infty}^{\infty}\operatorname{sigmoid}(z-I_T)
\frac{e^{-z^2/2}}{\sqrt{2\pi}}\,dz.
$$

This sets an expected rate under that distribution. It does not enforce an exact per-batch quota or guarantee the same rate after real distribution shifts and SIREN adjustment.

An equation sanity check performed for this reference used deterministic midpoint quadrature on \([-10,10]\) with spacing \(0.002\) and bisection:

| Target rate \(q\) | Threshold solving the stated integral, approximately | Appendix A.7's printed threshold | Integral's rate at printed threshold |
|---|---:|---:|---:|
| 0.0625 | 3.1193 | 2.06 | 0.1486 |
| 0.125 | 2.2854 | 1.53 | 0.2171 |
| 0.25 | 1.3149 | 0.89 | 0.3231 |

The printed values do not solve the displayed standard-normal expectation. Appendix A.7 also allows further empirical tuning of thresholds, which must be distinguished from solving that equation. This calculation checks the formula only; it does not establish what released code actually executes.

## Unresolved source details

The [design contract](/design/oasis.md) selects an equation-based profile; reconcile these discrepancies against pinned author code:

| Source discrepancy | Current interpretation / required follow-up |
|---|---|
| Eq. 4 specifies prior statistics, while Algorithm 1 uses current-index notation | Use prior statistics; avoid incorporating the candidate into its own normalizer |
| Algorithm 1's pairwise assignment can appear to overwrite contributions | Sum from original scores according to Eq. 8 |
| Prose says Bernoulli selection, but Eq. 6 and pseudocode use one random threshold | Use the shared draw; label independent draws separately |
| Main Eq. 3 uses a batch-mean variance, while Appendix A.1 Lemma A.3 introduces batch-size rescaling to estimate per-sample variance | Do not silently substitute the proof's rescaled quantity into the main implementation |
| Appendix A.1 uses an EMA retention-coefficient convention different from main Eq. 3 | Record which coefficient multiplies the newest observation; avoid copying the symbol without its definition |
| Appendix A.7's integral and printed thresholds disagree | Store the calibration procedure, threshold, and measured realized rate |

These distinctions are substantive: they change normalization, selected examples, or their joint distribution. The theorem's local stationarity, weak dependence, and bounded-moment assumptions also require scrutiny under abrupt continual-learning shifts. A normal approximation is not an unconditional guarantee of normally distributed scores.

## Optimizations and limitations

**Native optimizations.** Last-layer scoring avoids full-model per-example gradients. Running statistics replace rescoring the entire observed history for a global reference. SIREN approximates the effect of training on related candidates without repeatedly updating and rerunning the model between individual selections.

**Engineering optimizations that preserve the chosen profile.**

- Release candidate gradients after each admission decision; retain stream statistics and replayable RNG state.
- Bound candidate batch size. The literal higher-order sum contains exponentially many subsets; a tiny exact implementation is the reference for any derivation of an equivalent faster expression.
- Use exact tensor contractions for gradient norms and inner products when their equivalence to explicit gradients has been checked. A vocabulary projection can have \(V\times d\) parameters even in a small model; materializing every dense example gradient may exhaust memory.
- Batch or chunk scoring without changing the logical selection batch, SIREN order, statistics updates, or sampling draw. Splitting one logical batch into independent decisions changes the method.

Truncating higher-order terms, projecting gradients, using hidden-state cosine, or replacing probabilities with top-k selection are algorithmic adaptations. Approximation may become necessary for feasibility, but it must have a separate method identifier and accuracy/cost evaluation.

**Cost boundaries.** Account for every candidate forward, required scoring backward, similarity computation, higher-order adjustment, and admitted-sample training backward. “Near inference cost” does not mean zero-cost scoring. The native infinite-memory protocol also has a growing storage cost that must not disappear from comparisons.

## Fit in this project's pipeline

The hybrid applies OASIS only to current arrivals; filtering replay would change TSR/SAR allocation. [OASISPipeline](/design/oasis.md) owns scoring/state conventions; [CombinedPipeline](/design/combined-pipeline.md) owns updates and recovery. A frozen/tied output head needs a scoring capability that does not make it trainable.

## Close work already covered in the literature reviews

| Related method and existing review | Shared family / variant | Difference relevant to implementation |
|---|---|---|
| **Data Diet / GraNd**, [foundational review §6](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md); [primary paper](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf) | Both follow the gradient-magnitude data-selection family | GraNd uses early-training gradient norms; EL2N uses prediction error. OASIS uses squared last-layer norms with continually updated normalization and intra-batch redundancy adjustment |
| **Adapt-∞**, [incremental review §7](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md); [primary paper](https://arxiv.org/abs/2410.10636) | Both use gradient information to reduce redundant continual instruction training | Adapt-∞ forms pseudo-skill clusters and chooses selection strategies within them; LITE additionally prunes the retained pool. OASIS makes batch-level probability decisions and does not by itself provide a bounded memory writer |
| **LEAD**, [linked review §3](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md); [primary paper](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) | Both follow adaptive allocation of training compute using learner feedback | LEAD combines loss-reduction uncertainty with hierarchical bandit selection over a candidate pool; OASIS scores current candidates using gradient energy, similarity, and stream statistics |
| **RDS+**, [linked review §1](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md); [primary paper](https://arxiv.org/abs/2503.01807) | Both use similarity in data selection | RDS+ compares target and candidate hidden representations; OASIS compares supervised gradients between batch members. Substituting one representation for the other changes the objective and cost |
| **S2L**, [linked review §2](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md); [primary paper](https://arxiv.org/abs/2403.07384) | Both seek a useful smaller training set | S2L clusters loss trajectories from a separately trained small proxy; OASIS performs online decisions with the changing learner. Proxy preparation, access assumptions, and repeated subset training must be counted separately |

