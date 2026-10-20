# aL-SAR — Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling

**Authors:** Minhyuk Seo, Hyunseo Koh, Jonghyun Choi. **Venue:** ICLR 2025. **Reference date:** 2026-10-20.
**Sources:** [Published paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf), [proceedings record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html), [local PDF](/Papers/aL-SAR.pdf), [author repository](https://github.com/snumprlab/budgeted-cl), [implementation contract](/design/al-sar.md).
**Local SHA-256:** `d35ec11c8addcc5074f9a73736cd70b65871c5e16c956c03ba007f02620c07df`.

## Problem and reported results

aL-SAR combines adaptive layer freezing with retrieval based on recent effective training exposure. It compares total FLOPs and storage bytes, including auxiliary models/logits; equal epochs or replay counts can conceal these costs. The method does not optimize memory compression (Appendix A.27).

In Table 4's CIFAR-10 Gaussian-task ablation, vanilla replay reaches **70.08 ± 0.97** final accuracy at **163.74 TFLOPs**, SAR reaches **72.43 ± 0.38** at **171.94 TFLOPs**, and aL-SAR reaches **72.57 ± 0.79** at **146.80 TFLOPs**. Retrieval improves accuracy but adds compute; freezing offsets it in this experiment. [Paper, Table 4](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

## Problem, resources, and notation

Let \(M\) be episodic memory, \(B\) the training batch size, \(y_i\) an example's class, and \(L\) the number of ordered layer groups.

The native algorithm updates memory with a Greedy Balancing Sampler before retrieving a training batch from memory. Thus SAR is not itself an optimized memory writer. That separation matters when replacing the writer with CSReL.

## Similarity-aware retrieval formulas

### 1. Discounted use frequency

An old use should have less influence than a recent use. In Algorithm 1, retrieval probabilities are computed from the current state, after which the existing counts are decayed and sampled examples are incremented. With the algorithm's frequency scale \(k\),

$$
\rho=1-\frac{B}{k|M|},\qquad
c_i\leftarrow \rho c_i+\text{number of times }i\text{ is retrieved}.
$$

Here \(\rho\) names the multiplicative retention factor; the paper also uses a decay notation in the prose. Validate its range for the chosen batch/memory sizes. Sampling with replacement, batch multiplicity, and count-update timing must be specified.

### 2. Class-gradient similarity — Eq. 6

The conceptual similarity is

$$
S_{a,b}
=\mathbb{E}_{z_i\sim D_a,\ z_j\sim D_b}
\left[\cos\!\left(\nabla_\theta\ell(z_i),\nabla_\theta\ell(z_j)\right)\right].
$$

Rather than maintain every sample-pair similarity, the implementation estimates class similarities using exponential moving updates and per-example gradients from a small sampled parameter subset. Algorithm 1 samples **0.05%** of parameters for that purpose, intersecting the subset with currently unfrozen parameters. The exact subset identity belongs in resume state. Similarity remains signed: conflicting gradients may reduce estimated effective exposure.

### 3. Effective frequency and retrieval — Eqs. 7–8

Let \(C_y=\sum_{j:y_j=y}c_j\) be discounted exposure for class \(y\). Algorithm 1 independently decays and increments class totals, with no explicit subtraction when an example leaves memory. Thus native class exposure must not be assumed to equal a freshly recomputed sum over only the retained examples; verify eviction behavior against pinned code. Then

$$
\hat c_i=c_i+\sum_{y\in\mathcal Y}S_{y,y_i}C_y,
\qquad
\boxed{p_i=\frac{\exp(-\hat c_i/T)}{\sum_{j\in M}\exp(-\hat c_j/T)}} ,
\quad T>0.
$$

Training on related examples contributes exposure even if example \(i\) itself was rarely drawn. Low effective exposure receives higher probability. For illustration, effective counts \(1\) and \(3\) at \(T=1\) yield probabilities approximately \(0.881\) and \(0.119\); raw frequency alone cannot capture the similarity contribution.

Algorithm 1's printed probability notation can be read as a second softmax around exponentials. The project follows **Eq. 8's single normalized exponential**; a double softmax changes the distribution. [Paper, §3.2 and Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

## Adaptive-freezing formulas

### 1. Expected information per computation — Eq. 4

Let \(f_\ell=\operatorname{tr}(F(\theta_\ell))\) be the estimated Fisher trace for layer \(\ell\), \(C_F\) the full forward cost, and \(C_{B,\ell}\) that layer's backward cost. Freezing the prefix \(1,\ldots,n\) gives the surrogate

$$
Q_n=\frac{\sum_{\ell=n+1}^{L}f_\ell}
{C_F+\sum_{\ell=n+1}^{L}C_{B,\ell}}.
$$

This trades fewer updated parameters per batch against more batches affordable with the same computation. It is a model of information per FLOP, not a measured accuracy-per-FLOP guarantee.

### 2. Batch-freezing criterion — Eq. 5

The batch-aware score is

$$
\boxed{\operatorname{BFC}_n(z_t)
=\max_m Q_m\sum_{\ell=1}^{n}C_{B,\ell}
-\sum_{\ell=1}^{n}\operatorname{tr}(F_{z_t}(\theta_\ell)).}
$$

The first term estimates future information obtainable with saved backward compute. The second estimates information forgone on the current batch. Choose the maximizing valid prefix; our implementation includes \(n=0\), with no freezing, as an explicit safe candidate.

### 3. Cheap batch-specific estimate — Appendix A.1, Eq. 9

With \(x_L\) the final-layer **feature activation**, the approximation is

$$
F_{z_t}(\theta_\ell)\approx
\frac{\|\nabla_{x_L}\ell(z_t)\|^2}
{\mathbb E_z\|\nabla_{x_L}\ell(z)\|^2}
F(\theta_\ell).
$$

Track the denominator and layer traces with moving statistics. The shared scaling across layers is an approximation; the chain rule does not make arbitrary layer-gradient norms universally proportional across inputs. Algorithm 1 periodically trains all layers to refresh traces, using every fourth iteration in its presented schedule.

**Source ambiguity:** Eq. 9 uses an expectation of squared gradient norm, while Algorithm 1 squares an EMA norm. The [design's equation profile](/design/al-sar.md) uses the former; compare against pinned code before claiming reproduction.

## One native update

1. Admit the incoming record through the native balanced memory policy.
2. Compute effective frequencies and retrieve a batch using SAR.
3. Decay counts and increment the sampled examples/classes in the specified order.
4. Forward the batch and obtain the final-feature statistic.
5. Refresh all layer traces on the scheduled full-update iteration; otherwise choose a freeze prefix with BFC.
6. Obtain permitted per-example gradients, update class similarities, and update only unfrozen parameters.

The three clocks—sample arrivals, retrieved batches, and successful optimizer steps—are not automatically the same. Our hybrid intentionally uses the successful learner-update clock for exposure bookkeeping; that difference must remain visible.

## Optimizations and their limits

**Paper optimizations:** class-level rather than sample-pair similarity state; a small sampled parameter subset; reuse of training gradients; EMA statistics rather than repeated full-memory estimates; prefix freezing and periodic full refresh. These reduce overhead but do not make it zero.

**Engineering optimizations:** maintain class totals incrementally; cache invariant layer cost estimates; vectorize stable softmax; release per-example graphs after use; apply legal graph cuts for frozen prefixes. Reprofile or recompute cost estimates when sequence length, batch shape, adapter layout, or precision invalidates them.

**Boundaries:**

- Raw frequency is not equivalent to SAR. Table 7 shows raw-count retrieval can perform worse than random; discounting and similarity are substantive.
- Class-level approximation may obscure within-class diversity. In language data, a class may be absent altogether.
- A gradient parameter subset with no unfrozen coordinates cannot produce a fresh class similarity; do not fill it with fabricated zeros without a declared update policy.
- Adam-style momentum and weight decay can move “frozen” parameters if gradients/state are handled incorrectly.
- Freezing an intermediate block does not eliminate all backward work if upstream trainable parameters still require gradients.
- The paper's larger multimodal experiment demonstrates aL with LoRA; it does not establish full SAR on arbitrary instruction data or permit exceeding our 700M ceiling.

## Fit in this project's pipeline

The hybrid first draws a task with TSR, then samples from that task's retained examples using SAR. The resulting probability factorizes as \(p(j)\,p(i\mid j)\). Sampling globally after TSR would override its task choice. CSReL remains the sole memory writer. The hybrid's class totals sum active current-record and retained-history counts, removing unretained records at task closure; this active-population convention is explicit project behavior, not a proven equivalence to native cumulative class exposure.

Define class/group metadata from the eventual dataset or an explicitly tested adaptation. Do not import InsCL-style task distance or OASIS per-example informativeness into SAR and retain the same method label. For LoRA, specify how ordered model layers map to trainable adapter groups and what Fisher/gradient statistics represent.

State ownership and resume: [aL-SAR design](/design/al-sar.md).

## Close work already covered in our reviews

| Related work and references | Shared variant and essential difference | How to use the relationship |
|---|---|---|
| **MIR**, [review §4](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md), [primary paper](https://arxiv.org/abs/1908.04742) | Also follows adaptive replay retrieval. It ranks predicted loss increases after a virtual current update, whereas SAR estimates recent effective exposure. | Direct reader comparator; include its lookahead/scoring cost. |
| **ASER**, [review §5](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md), [primary paper](https://ojs.aaai.org/index.php/AAAI/article/view/17159) | Also values memories relative to old/new learning needs, using adversarial/cooperative KNN-Shapley estimates rather than frequency. | Another reader comparison with different feature and labeling assumptions. |
| **GSS**, [review §3](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md), [primary paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) | Also uses gradient similarity, primarily to retain diverse memory examples. SAR uses similarity to infer exposure during retrieval. | Shared signal family, different lifecycle stage; GSS is not “SAR with another temperature.” |
| **LEAD**, [review §3](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md), [primary paper](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) | Also follows adaptive scheduling from training feedback. It uses dynamic uncertainty and cluster-bandit allocation rather than discounted usage and aL's freezing criterion. | Compare adaptive sampling mechanisms without conflating their state or candidate-access requirements. |
| **OASIS**, [review §5](/results/literature-review/small-model-selection/six-recent-selection-papers-2026-09-30.md), [paper reference](/Papers/OASIS-summary.md) | Also uses cheap gradient/Fisher surrogates and running statistics. OASIS selects examples; aL chooses layer updates, while SAR tracks exposure. | Possible composition with distinct statistics; do not share incompatible gradient caches or counters. |

