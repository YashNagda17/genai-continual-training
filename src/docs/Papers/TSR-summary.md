# TSR — Rethinking Transfer in Continual Learning: A Replay-Based Realisation

**Authors:** Yang Meng, Zhenya Liu, Zhuokai Zhao, Yuxin Chen. **Status:** arXiv preprint, 2026-07-17, version 1. **Reference date:** 2026-10-20.
**Sources:** [Paper record](https://arxiv.org/abs/2607.15587v1), [local PDF](/src/docs/Papers/TSR.pdf), [author repository](https://github.com/ymeng3/transfer-selective-replay), [implementation contract](/design/tsr.md).
**Local SHA-256:** `c10d4e822179ebbd96b3538357bbfcdb2c87faba03f57d57ce3e5cf4d01baa81`. `Replay.pdf` is an identical copy.

## Reported results

On TRACE-8 with Qwen2.5-0.5B-Instruct, TSR versus experience replay reaches **0.475 ± 0.030 / 0.451 ± 0.031** overall, **0.439 ± 0.017 / 0.427 ± 0.024** plasticity, and **+0.035 ± 0.015 / +0.025 ± 0.020** backward transfer. These are means and standard deviations over five seeds, not a significance test. [Paper, Table 3 and Appendix C](https://arxiv.org/pdf/2607.15587).

## Problem and transfer interpretation

For tasks \(D_1,\ldots,D_K\), TSR aims to improve learning of \(D_t\) from compatible retained history when current-task data and updates are limited.

The paper's Eq. 2 expresses a conceptual condition:

$$
\text{positive transfer requires headroom, a persistent carrier, and a suitable source.}
$$

This is a research framing, not a universal quantitative theorem guaranteeing accuracy improvement. Headroom means the target has room to improve under its data/step budget. Replay provides repeated access to historical supervision; a one-time parameter initialization can be overwritten by subsequent target optimization in the paper's tested setting.

Pairwise transfer matrices and future-task analyses in the paper are diagnostic experiments. They do not authorize probing future tasks or using their test performance to select sources during an actual causal continual run.

## Formulas and method

### 1. Label-aware task signature — §4.3.2, Eq. 4 and Algorithm 2

For a fixed shared initialization \(\theta_0\), estimate

$$
g_t=\frac{1}{m}\sum_{b=1}^{m}
\nabla_B L_{\mathrm{CE}}(\theta_0;\mathcal B_{t,b}),
$$

where \(B\) denotes the **LoRA-B parameters** across the declared adapter layers, not the replay buffer. The task signature uses labels and a few permitted current-task minibatches. Probe gradients do not update \(\theta_0\); the paper's reference protocol uses ten probe batches.

For a LoRA update \(\Delta W\propto BA\), the initialization uses \(B=0\) and a shared initialized \(A\). Then

$$
\nabla_A L\propto B^\top\nabla_W L=0
$$

at initialization, while \(\nabla_B L\) can be nonzero. This is why replacing the specified B-gradient with A-gradients, hidden-state embeddings, the changing learner's gradients, or a different random adapter initialization is not equivalent.

Flatten layers in a stable order and use the same initialization, adapter coordinates, tokenizer/template, label masking, and loss reduction for all tasks. Without a common coordinate system, cosine similarity need not compare task relationships meaningfully.

### 2. Historical-task routing

For eligible historical tasks \(j<t\),

$$
s_{j,t}=\cos(g_j,g_t),
\qquad
w_{j,t}=\frac{\exp(s_{j,t}/\tau)}
{\sum_{k<t}\exp(s_{k,t}/\tau)},
\qquad \tau>0.
$$

Temperature \(\tau\) controls concentration. Small values emphasize the most aligned source; large values approach **uniform task sampling**, which equals uniform example sampling only when eligible task buffers have equal sizes.

Example: similarities \(0.8,-0.2\) with \(\tau=0.5\) give probabilities \(0.881,0.119\); even a negatively aligned source remains possible.

### 3. Batch source and teacher routing — Algorithm 2

Write each historical record as

$$
R_j=(g_j,M_j,\theta_j),
$$

where \(M_j\) stores replay examples and \(\theta_j\) is the learner snapshot after completing task \(j\). For each training update:

| Draw | Training examples | Teacher |
|---|---|---|
| Current branch, probability \(1-\alpha_{\mathrm{replay}}\) | A batch from current task \(D_t\) | \(\theta_{j^*}\), with \(j^*=\arg\max_j w_{j,t}\) |
| Replay branch, probability \(\alpha_{\mathrm{replay}}\) | Draw task \(j\sim w_{\cdot,t}\), then uniformly draw from \(M_j\) | The same source task's \(\theta_j\) |

Algorithm 2 chooses a branch for an update, while Figure 4's captioned schematic describes replay mixed into every batch. The project follows Algorithm 2's explicit branch logic. A mixed-batch implementation needs an explicit equivalence argument or adaptation label; compare this discrepancy against pinned author code before claiming reproduction.

For the example above, current batches use task A's teacher. A replay batch drawn from task B must use task B's teacher, even though A is globally more similar. Replacing every teacher with the most recent snapshot breaks the source-era contract.

On the first task there is no historical teacher or replay source. The project defines current-only supervised training with no distillation. An empty eligible source set later also needs an explicit fallback; an argmax or softmax over an empty set is invalid.

### 4. Supervised learning plus distillation

The paper's objective is

$$
L=L_{\mathrm{CE}}+\lambda L_{\mathrm{KD}},
\qquad
L_{\mathrm{KD}}=
D_{\mathrm{KL}}\!\left(p_{\mathrm{teacher}}\;\|\;p_{\mathrm{student}}\right).
$$

For language modeling, compare token distributions at the same teacher-forced contexts, vocabulary, token positions, and response mask. The teacher runs frozen and without a gradient graph; student gradients must remain active. Gold labels continue to define \(L_{\mathrm{CE}}\); teacher predictions do not silently replace the targets.

The reference configuration reports distillation temperature \(T_{\mathrm{KD}}=2\), distinct from routing temperature \(\tau\). A possible \(T_{\mathrm{KD}}^2\) rescaling and exact token/batch reduction must be checked against the pinned implementation; do not infer those conventions from the shorthand KL equation alone.

### 5. Evaluation quantities — §6.1, Eqs. 6–8

For a compatible higher-is-better task performance matrix \(A_{t,i}\),

$$
\mathrm{Overall}=\frac1K\sum_{i=1}^{K}A_{K,i},
\qquad
\mathrm{Plas}=\frac1K\sum_{i=1}^{K}A_{i,i},
$$

$$
\mathrm{BWT}=\frac1{K-1}\sum_{i=1}^{K-1}(A_{K,i}-A_{i,i}).
$$

Plasticity measures performance just after learning each task; it does not by itself isolate transfer from previous tasks. Compare against a matched target-only or source-agnostic baseline. The paper's forward-transfer discussion concerns improved target learning and must not be silently substituted for a zero-shot-before-learning FWT definition.

Appendix C describes exact-match evaluation for classification and evaluation loss for generation, while the result tables use accuracy/performance language. Confirm the benchmark-specific aggregation and orientation from code before reproducing the aggregate; averaging raw lower-is-better losses with higher-is-better accuracies is invalid. The general project metrics contract remains authoritative.

## Native protocol versus project choices

These are **published reference settings**, not selected models, datasets, or hyperparameters for our future runs:

| Item | Paper reference configuration |
|---|---|
| Learners | Qwen2.5-0.5B-Instruct and a 7B counterpart; only the smaller scale is potentially within current scope |
| Adapter | LoRA rank 16, scaling alpha 32, attention q/k/v/o projections |
| Optimization | AdamW, learning rate \(10^{-4}\), 500 steps per task |
| Small-model batch / length | Batch size 4; maximum sequence length 384 |
| Limited target data | \(N=50\) training examples per task in the described low-budget setting |
| Replay storage | 100 examples per task, as separately stated in Appendix C |
| Routing / distillation | Ten probe batches; \(\tau=0.1\); replay probability \(0.5\); \(\lambda=0.5\); \(T_{\mathrm{KD}}=2\) |

The stated 50-example training budget and 100-example per-task buffer require a data-access check: determine whether they use separate candidate pools, sampling with replacement, or another convention. Do not silently introduce unseen historical examples or manufacture unique records to fill a buffer.

Our finite global replay budget can empty task buffers; exclude those tasks and renormalize under [the hybrid contract](/design/combined-pipeline.md). A missing teacher for a nonempty task is an error. CSReL buffers and SAR retrieval are adaptations of native uniform-within-task TSR.

## Optimizations and limitations

**Native optimizations.** The fixed-initialization descriptor needs no auxiliary training updates. Task weights are computed once for the declared task descriptor, rather than repeatedly evaluating all source-task transfer experiments. The learner retains a single active adapter for inference, avoiding a growing inference-time adapter ensemble.

These savings do not eliminate probe backward passes, teacher forward passes, or storage of task-era teacher states. A comparison reporting raw replay-buffer bytes does not account for the entire tuple \((g_j,M_j,\theta_j)\).

**Engineering optimizations that preserve behavior.**

- Cache immutable signatures using a fingerprint of probe weights, adapter initialization, layers, tokenizer/template, loss reduction, and actual probe IDs.
- Keep one immutable frozen base shared on disk and exact per-era adapter states when every snapshot really shares that base. If base weights change, adapter-only checkpoints are insufficient.
- Load only the selected teacher on demand within the declared residency budget; avoid keeping all task teachers on the accelerator.
- Cache routing weights while the task signature and eligible source set are unchanged. Recompute after an eligibility change.
- Perform teacher inference without autograd. Release teacher logits and temporary buffers after the update; a batch's full-vocabulary logits can dominate activation memory.

Logit caching is exact only for the same immutable teacher and tokenized context/mask. Reusing logits for different examples or generating responses instead of matching teacher distributions changes the objective. A smaller or newer substitute teacher is also a distinct variant, even if an ablation suggests comparable quality.

**Scientific limitations.** Gradient alignment is a heuristic for compatibility; average signatures can cancel internally opposing subskills. Transfer headroom can disappear with larger target budgets. The paper finds that distillation can behave mainly as regularization in its tested regime, but that does not establish universal teacher irrelevance. Its findings about parameter transfer being transient are similarly regime-specific.

**Storage limitation.** Fixed examples per task and one era teacher per task grow with the task count. A fixed global example budget alone does not bound teacher/signature storage. The project must estimate the full task-horizon footprint, reject infeasible runs, and never silently delete an algorithm-required teacher to meet a disk target.

## Fit in this project's pipeline

TSR selects the source task and teacher; SAR replaces uniform within-task sampling, and CSReL replaces native buffer construction. Keep the teacher attached to its source and commit complete task records atomically. Contracts: [TSRPipeline](/design/tsr.md), [CombinedPipeline](/design/combined-pipeline.md), [checkpoint/teacher lifecycle](/design/model-weights.md).

## Close work already covered in the literature reviews

| Related method and existing review | Shared family / variant | Difference relevant to the pipeline |
|---|---|---|
| **InsCL**, [primary paper](https://aclanthology.org/2024.naacl-long.37/) | Both follow task-aware allocation of historical replay | InsCL allocates replay using instruction-distribution distance and then selects informative examples; larger distance protects difficult-to-retain history. TSR favors label-gradient compatibility to support current learning. Distance-for-retention and similarity-for-transfer can point in opposite directions |
| **SEEKR**, [primary paper](https://aclanthology.org/2024.emnlp-main.190/) | Both combine replay with distillation | SEEKR selects attention knowledge using sensitivity/forgettability and also distills outputs; TSR routes task-era output teachers with the source task. Attention/head/query budgets are not equivalent to TSR's task weights |
| **On-Policy Replay**, [primary paper](https://arxiv.org/abs/2605.29495) | Both revisit historical prompts with model-derived supervision | OPR regenerates and filters responses using the latest policy, then performs supervised replay. TSR retains gold targets and adds KL from an era-matched teacher. Response refresh and teacher-distribution matching must remain separate ablations |
| **GSS**, [primary paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) | Both use gradients to reason about historical data | GSS selects diverse individual memory examples to represent past constraints; TSR summarizes tasks and prefers aligned sources. Diversity and compatibility optimize different roles and can favor different examples |
| **RDS+**, [primary paper](https://arxiv.org/abs/2503.01807) | Both use target-conditioned cosine similarity for data allocation | RDS+ ranks candidates using hidden representations relative to target queries; TSR uses supervised task-mean LoRA-B gradients at a shared initialization. They have different label requirements, granularity, and compute costs |
| **SSR**, [primary paper](https://aclanthology.org/2024.acl-long.77/) | Both use model knowledge to support rehearsal | SSR synthesizes, refines, and selects replay examples; TSR retains observed examples and distills a routed teacher. Synthetic data generation is not an implementation detail of TSR |

