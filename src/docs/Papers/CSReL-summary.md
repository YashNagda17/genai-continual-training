# CSReL — Coreset Selection via Reducible Loss in Continual Learning

**Authors:** Ruilin Tong, Yuhang Liu, Javen Qinfeng Shi, Dong Gong. **Venue:** ICLR 2025. **Reference date:** 2026-10-20.
**Sources:** [Published paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf), [local PDF](/src/docs/Papers/ICLR-2025-coreset-selection-via-reducible-loss-in-continual-learning-Paper-Conference.pdf), [author repository](https://github.com/RuilinTong/CSReL-Coreset-CL), [implementation contract](/design/csrel.md).
**Local SHA-256:** `ed38d1fcb290e8e7089268343211e2a77e948b9e7b710e6e4eba8d2b22a6ad6a`. `CSReL` is a duplicate of this PDF.

## Reported results

Table 1 reports Split CIFAR-100 final average accuracy of **17.48 ± 0.21** for CSReL-CL, **18.47 ± 0.17** for CSReL-CL-Prv, and **13.82 ± 1.31** for uniform selection. On Split CIFAR-10, CSReL-CL and uniform reach **38.97 ± 2.61** and **37.05 ± 3.06**. These are classification results; the generative SLM port is unvalidated. [Paper, Table 1 and §5.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

## Problem and notation

Let \(D=\{(x_i,y_i)\}_{i=1}^{N}\) be the permitted training candidates and \(m\) the coreset capacity. Let \(S\subseteq D\) be the selected subset, \(\ell_i(\theta)\) the per-example supervised loss, \(\theta_S\) a model trained on \(S\), and \(\theta_D\) a model trained on \(D\). The paper calls \(\theta_D\) the **holdout model**. This name does not authorize training on a validation or test set.

CSReL seeks a replay subset whose training effect approximates the candidate set. Unlike highest-loss selection, it discounts examples that remain difficult even for a fitted reference model.

## Formulas and their meaning

### 1. Bilevel coreset objective — Eq. 1

Using nonnegative sample weights \(w\), the idealized objective is

$$
\min_{w\geq 0,\ \|w\|_0\leq m} L_D(\theta^*(w)),
\qquad
\theta^*(w)\in\arg\min_\theta\sum_{i=1}^{N}w_i\ell_i(\theta),
$$

where \(L_D(\theta)=\sum_i\ell_i(\theta)\). The inner optimization trains on the chosen weighted subset; the outer optimization measures its usefulness on the full permitted training set. This is a combinatorial selection problem, not a tractable instruction to enumerate all subsets.

Earlier greedy approaches use an implicit gradient of the outer loss. With \(H\) the inner-loss Hessian at the fitted subset model, the relevant expression is

$$
\frac{\partial L_D(\theta^*(w))}{\partial w_i}
=-\nabla_\theta L_D(\theta^*)^\top H^{-1}\nabla_\theta\ell_i(\theta^*).
$$

This motivates selecting examples that improve the outer objective, but computing the influence approximation can be costly and sensitive to difficult/noisy examples. CSReL substitutes a loss-based estimate; it does not calculate this inverse Hessian in ordinary selection. [Paper, §§3.1–3.3, Eqs. 1–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

### 2. Reducible loss — Eq. 6

The paper begins with the hypothetical gain from adding example \(i\) to \(S\),

$$
G_i=\log p(y\mid x;S\cup\{i\})-\log p(y\mid x;S),
$$

where \((x,y)\) denotes the full candidate set \(D\), not only example \(i\). It uses Bayesian/conditional-independence reasoning and model approximations to obtain

$$
\boxed{G_i\approx \ell_i(\theta_S)-\ell_i(\theta_D).}
$$

The subtraction direction is essential: prefer **high subset-model loss and low reference-model loss**. This is an estimated usefulness score, not a certified causal improvement for the continual learner.

Example: losses \(3.0,0.4\) give a score of \(2.6\), while \(8.0,7.8\) give \(0.2\). CSReL prefers the former; highest-loss selection prefers the latter.

### 3. Continual replay objective — Appendix E, Eq. 17

For current and replay minibatches \(B_t\) and \(B_M\), the standalone learner uses

$$
L_{\mathrm{cnt}}=
\frac{1}{|B_t|}\sum_{i\in B_t}\ell_i(\theta_{\mathrm{cnt}})
+\alpha\frac{1}{|B_M|}\sum_{i\in B_M}\ell_i(\theta_{\mathrm{cnt}}).
$$

The two terms are normalized separately. This is not generally the same as an unweighted mean over their concatenation. Omit the replay term when history is empty; do not divide by zero. The hybrid uses its separately specified learner objective.

## How selection actually runs

Algorithm 1 is a repeated fitting-and-selection procedure:

1. Train the reference model on \(D\); compute its losses for candidate IDs.
2. Initialize \(S=\varnothing\) and divide capacity \(m\) across the configured outer rounds.
3. Initialize and fit a subset model on the current \(S\).
4. Score every remaining candidate using the current subset model and reference losses.
5. Add that round's top-scoring records without replacement; repeat until capacity is reached.

The changing subset model is why later selections can differ from a one-shot top-\(m\) ranking. The empty-set initialization, integer allocation, fitting schedule, and tie handling must be explicit. Our design uses the declared untrained initialization for an empty subset; agreement with a pinned author implementation remains a reproduction check.

At task end, CSReL-CL allocates the global example budget approximately equally across seen tasks. It selects the current coreset and shrinks older ones using only their retained records. Appendix E stores reference losses for old exemplars so their reference model need not be retrained during later reselection.

## Variants in this paper

| Variant | Formula or algorithmic change | Meaning for our pipeline |
|---|---|---|
| CSReL-CL | Reference is trained on the current task; task-end construction follows Algorithms 1–2 | Baseline task-end writer |
| CSReL-CL-Prv | \(G_i^{\mathrm{Prv}}=\ell_i(\theta_{S_t})-\ell_i(\theta_{D_t\cup M})\), Eq. 8 | Reference sees current training data plus retained history; intended to account for interference |
| CSReL-RS | \(G_i^{\mathrm{RS}}=\ell_i(\theta_{S_t})-\ell_i(\theta_{\mathrm{cnt}})\), Eqs. 9/18 | Streaming memory admission; not interchangeable with task-end greedy construction |
| KD-aware extension | \(G_i^{\mathrm{KD}}=\ell_{\mathrm{MSE}}(x_i,o_i;\theta_S)-\ell_{\mathrm{MSE}}(x_i,o_i;\theta_{D_t\cup M})\), Eq. 11; combine as \(\alpha_sG_i^{\mathrm{RS}}+\beta_sG_i^{\mathrm{KD}}\), Eq. 12 | Adds stored-logit supervision and associated storage; not automatically enabled by TSR KL |

For CSReL-RS, Algorithm 5 computes the batch-relative factor

$$
s_i=|B|\frac{\exp(G_i^{\mathrm{RS}})}{\sum_{j\in B}\exp(G_j^{\mathrm{RS}})}.
$$

It scales the reservoir admission test by \(s_i\), replaces a uniformly chosen stored position on admission, and retrains the memory model when enough updates accumulate. Preserve the algorithm's integer random-draw convention; simply implementing continuous probability \(K s_i/n\) can differ at small counts. Algorithm 6 reuses the current batch's continual-model losses from before the learner update. The memory model is retrained on current-task memory and reset at task changes. [Paper, §§4.2–4.4 and Appendices E–G](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

## Optimizations, cost, and limitations

**Optimizations in the paper:** replacing inverse-Hessian/gradient influence scoring with forward loss differences; caching older exemplar reference losses; using the continual model as the reference approximation in RS; amortizing RS memory-model retraining over multiple admissions. These optimize different variants and should not be mixed without a named change.

**Semantics-preserving implementation choices:** cache reference losses only while their model, target, preprocessing, and reduction remain unchanged; stream candidate scores; retain only the top candidates needed for a round; execute reference and subset model roles sequentially. Release reference weights once their required loss table is complete and no recovery checkpoint needs them.

**Costs that remain:** reference training, subset retraining at every round, candidate forwards, current-task retention during selection, and checkpointing. Approximate scaling is reference training plus a sum of subset-training and candidate-scoring costs over rounds; the score alone is not the total selection cost.

**Failure modes and assumptions:**

- A poorly fitted reference can treat useful rare examples as irreducible. A biased reference can reproduce that bias in the memory.
- The convergence-based motivation does not guarantee equivalent behavior with severely truncated inner training.
- Equal task quotas can underrepresent large/diverse tasks. Example-count capacity does not imply a token/byte limit.
- Native evidence is primarily classification. Response length, template, and target masking affect generative losses; both scoring roles need a comparable loss definition.
- Choosing \(\theta_{\mathrm{cnt}}\) as a convenient substitute for \(\theta_D\) in task-end CL changes the method toward the RS approximation; it is not an invisible optimization.

## Fit in this project's pipeline

Use [CSReLPipeline](/design/csrel.md) to propose memory after learner training. Current-task candidates remain accessible through this phase, including examples rejected by OASIS. Older candidates come only from committed replay. The lifecycle owner validates budgets and atomically commits the proposal; CSReL never mutates the learner while training its auxiliary models.

## Close work already covered in our reviews

| Related work and references | Shared variant and essential difference | How to use the relationship |
|---|---|---|
| **iCaRL**, [primary paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) | Also follows exemplar-based memory construction. Its herding approximates feature means; CSReL approximates reducible learning utility. Full iCaRL additionally includes classifier/distillation choices. | Compare writers with a fixed reader and learner; do not label herding-only as full iCaRL. |
| **GSS**, [primary paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) | Also follows model-dependent memory selection, using gradient diversity rather than loss reducibility. | A writer alternative; shared use of model signals does not make the objectives equivalent. |
| **S2L**, [primary paper](https://arxiv.org/abs/2403.07384) | Also follows proxy-learning-based subset construction. It clusters complete loss trajectories and samples across clusters; it does not subtract two model losses or rebuild a greedy subset model. | A possible current-task coreset comparator; adapting it to bounded replay is a separate experiment. |
| **LEAD**, [primary paper](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) | Also follows adaptive utility-based selection. Its utility comes from training feedback plus hierarchical scheduling, not a full-pool reference and repeated subset fits. | Compare cost and score staleness; this is primarily a training scheduler, not a drop-in task-end writer. |
| **LITE-Adapt-∞**, [primary paper](https://arxiv.org/abs/2410.10636) | Also addresses permanent pool pruning, through a different dynamic multimodal selection/pruning system. | Prior art against claiming that permanent historical-data reduction alone is novel. |

**Direct lineage identified by CSReL itself:** Appendix C says it adopts [Greedy Coreset's matching-pursuit framework](https://proceedings.neurips.cc/paper/2020/hash/aa2a77371374094fe9e0bc1de3f94ed9-Abstract.html) and adapts [Mindermann et al.'s RHO-LOSS scheduling idea](https://proceedings.mlr.press/v162/mindermann22a.html) to permanent subset construction. This is stronger evidence than a resemblance inferred from our reviews. After final coreset commitment and loss of raw historical access, discarded records cannot simply be revisited in later training epochs as in an ordinary full-pool scheduler.

