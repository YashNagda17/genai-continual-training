# Literature Review: Upgrades for the Incremental Fine-Tuning Pipeline

Date: 2026-11-01. Last updated: 2026-11-01, round 3. Papers deep-read: 16. Window: 2023-11-01 through 2026-11-01.

**Topic interpretation.** The current system is the task-chunk hybrid in [Idea.md](/Idea.md): OASIS admits current examples, TSR picks a historical task and a teacher, aL-SAR samples inside that task and freezes layers, and CSReL writes the replay memory. The question is which later papers can raise retained and new-task accuracy, or cut selection and training cost, without pretending that a stack of published selectors is a new algorithm. CFD and biology are used as mechanism sources, not as new application domains.

**Success criterion for this review.** Name ten works, say what each would replace, and separate a real substitution from a paper that only looks related.

**Venue assumption used downstream.** ICLR, NeurIPS, and ACL, because that is the venue tier of the current stack. No venue was named in the request.

**Evidence boundary.** Numbers below are copied from the papers. No method was reimplemented. OpenAlex citation counts retrieved on this date are incomplete version records and are not treated as influence ranks. The search audit is [search-log.md](/results/literature-review/pipeline-upgrades/search-log.md).

## What the current stages already are

OASIS scores a current batch by relative last-layer gradient energy and then down-weights redundant neighbors. TSR turns probe gradients at a fixed initialization into one weight over old tasks and attaches an era teacher. aL-SAR retrieves by discounted exposure and class similarity, and it freezes a layer prefix when Fisher information per cost is low. CSReL keeps examples whose loss still falls when a reference model has seen the pool. Those four roles are already published. An upgrade has to beat the role it replaces at matched access, bytes, and FLOPs.

---

## [2024] GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration (NeurIPS)

**Citation**: Jiachen T. Wang, Tong Wu, Dawn Song, Prateek Mittal, and Ruoxi Jia. "GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration." NeurIPS, 2024.
**Links**: [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed165f2ff227cf36c7e3ef88957dadd9-Paper-Conference.pdf) | [Code](https://github.com/Jiachen-T-Wang/GREATS) | AlphaXiv: no confirmed arXiv id in the sources read

### Idea

GREATS chooses, at every optimizer step, a subset of a candidate batch that most reduces loss on a validation set. A first-order Taylor step says that reduction is the inner product between the candidate gradient and the validation gradient, and a greedy second-order term penalizes candidates that point the same way as items already chosen. The expensive version materializes per-example gradients. The practical version uses a ghost inner product so the score comes from one backward pass rather than one backward per example. The experiments fine-tune Llama-2-7B and Mistral-7B with LoRA, and they also pretrain a small GPT-2, always with a validation set from the target domain available before training.

### Novelty

The online-batch problem is not new, and neither is gradient alignment. What is new is the set-level greedy objective, the Taylor approximation that makes the second-order redundancy term explicit, and the ghost estimator that brings the cost near ordinary training. The loss is adapted. The architecture is unchanged. The evaluation is a static fine-tune or pretrain, not a task stream, so the continual-learning protocol is unchanged and simply absent. The paper is explicit that the ghost trick is derived for SGD and that Adam is used only as a proxy.

### Results

On a nine-subject MMLU slice, GREATS reaches 54.2% after selecting half of each batch, against 50.4% for training without selection, 50.4% for GradNorm, and 50.8% for MaxLoss. On TydiQA the same comparison is 55.0% against 54.3%, 53.4%, and 54.7%. The MMLU setup fine-tunes Llama-2-7B with LoRA rank 128 on 5% of the LESS mixture. Throughput is 71.3 examples per second for the ghost implementation and 76.2 for ordinary training, against 4.2 for the direct per-example version. The gain over gradient-norm selection is the result that matters here. The gain over ordinary training on TydiQA is under one point. No significance test is reported for Table 2.

### Limitations

The method needs a validation set that represents the quantity being optimized, and the authors list that as a limitation. Perplexity is the selection objective, and they warn that it need not track downstream accuracy. The ghost inner product does not extend to Adam's moment normalization. The reported models are 7B, above this project's 700M cap, and the candidate pool is a static corpus rather than a bounded replay memory. Nine MMLU subjects are not the full 57-subject score. A corrupted or merely hard validation item can pull the whole batch toward it, because the score trusts the validation gradient.

### Relevance

This is the strongest replacement for OASIS's energy score. OASIS asks whether a current example has a large last-layer gradient. GREATS asks whether that gradient points at a declared probe. In this pipeline the probe can be a held-out slice of the current task, a held-out slice of retained tasks, or both, provided the split is fixed before scoring and test items never enter it. Threat to a claim that gradient energy is the right admission signal: high. Threat to the broader budgeted-replay question: low, because GREATS never allocates a replay share.

---

## [2024] LESS: Selecting Influential Data for Targeted Instruction Tuning (ICML)

**Citation**: Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. "LESS: Selecting Influential Data for Targeted Instruction Tuning." ICML, 2024. arXiv:2402.04333.
**Links**: [Paper](https://arxiv.org/abs/2402.04333) | [AlphaXiv](https://www.alphaxiv.org/abs/2402.04333) | [Code](https://github.com/princeton-nlp/LESS)

### Idea

LESS selects a static training subset for a named skill. It first runs a short LoRA warmup on a random 5% of the candidate pool for four epochs, stores low-dimensional Adam gradient features at those checkpoints, and then ranks the full pool by cosine alignment with the Adam features of a few target examples. The selected 5% is what the final model trains on. The datastore is built once and reused across target skills. The derivation is an Adam version of a TracIn-style influence sum: an example is useful if its optimizer step would have reduced loss on the target examples.

### Novelty

Influence functions and gradient-similarity search are older. The adaptation is the Adam feature, the LoRA projection that makes the datastore small enough to store, and the transfer setting in which a 7B selector picks data for a 13B or a different-family model. The problem, targeted instruction tuning, is a reframing of coreset selection rather than a new learning problem. The evaluation protocol is one-shot selection followed by one fine-tune. There is no task stream, no replay buffer, and no forgetting metric.

### Results

On Llama-2-7B, a LESS 5% subset scores 50.2±0.5 on MMLU, 56.2±0.7 on TydiQA, and 41.5±0.6 on BBH, against full-data scores of 51.6, 54.0, and 43.2. The 5% subset beats the full data on TydiQA and loses on MMLU and BBH. On Mistral-7B the same 5% beats full data on all three: 61.8±0.4 versus 60.4, 60.3±2.4 versus 57.7, and 56.0±1.0 versus 53.0. Against BM25, DSIR, and RDS on Llama-2-7B, LESS is ahead by 2.6, 3.5, and 1.7 points on those three tasks relative to the strongest of those baselines. Three seeds are reported. The paper also shows that a Llama-2-7B datastore can select data for Llama-2-13B and Mistral-7B, with the transferred subset sometimes beating the target model's own subset.

### Limitations

The abstract's "often outperforms the full dataset" is true for Mistral-7B and false for two of three Llama-2-7B tasks. The warmup must be trained before selection, so the cost is not a forward pass over the batch. The authors note that validation loss and instruction-tuning accuracy are not monotone. The method assumes the few-shot target examples are available and are the right skill, which a streaming task does not provide until that task arrives. Nothing in the paper measures retention of earlier tasks. Building the datastore is the dominant cost, and the paper's own accounting puts gradient computation ahead of the warmup.

### Relevance

LESS is the offline cousin of GREATS and a probe-aligned alternative to OASIS. It is usable inside this pipeline only at a task boundary: the current-task probe plays the role of the few-shot target, and the candidate pool is the current task plus the retained memory. It is a poor online admission rule, because the datastore goes stale as soon as the learner moves. Threat to OASIS-as-the-only-gradient-selector: moderate. Threat to the full hybrid: low. It should be a baseline for any admission claim, not a module to stack on top of OASIS.

The round-2 reading of the same paper changes what that baseline is allowed to ignore. LESS states that a sequence gradient is the average of completion-token gradients, that this average's norm is strongly anti-correlated with completion length, and that an inner product on the unnormalized gradient therefore selects short sequences. Table 13, on Llama-2-7B, prints random / vanilla-inner-product / cosine-normalized as MMLU 46.5 (length 105.4) / 47.3 (11.3) / 50.2 (15.1), TydiQA 52.7 (105.4) / 52.3 (19.7) / 56.2 (44.4), and BBH 38.9 (105.4) / 37.4 (19.0) / 41.5 (101.2). Every random-length cell is 105.4. Cosine removes the worst of the inner-product failure and still selects a short MMLU subset. The authors defer token-level selection as too expensive and say the length effect is a property of averaged token gradients, which is the quantity this repository squares for OASIS. Appendix J adds that cross-entropy and task accuracy need not move together on long generations. Those two pages are why a probe inner product is not the first course substitution.

---

## [2026] Replay on Demand (arXiv)

**Citation**: Lukas Thede, Shengzhuang Chen, Stefan Winzeck, Matthias Bethge, Zeynep Akata, and Jonathan Richard Schwarz. "Replay on Demand: An Emergent Curriculum for Balancing Adaptation and Forgetting in Continued Pretraining." arXiv:2609.40089, 2026.
**Links**: [Paper](https://arxiv.org/abs/2609.40089) | [AlphaXiv](https://www.alphaxiv.org/abs/2609.40089) | [Code](https://github.com/bethgelab/replay-on-demand)

### Idea

Replay on Demand (RoD) refuses to pick a replay percentage in advance. Adaptation examples and replay examples compete for the same training slots. An adaptation example is scored by how much of its loss a specialist reference model can still remove. A replay example is scored by how much its loss has risen since the pretrained model. The top of the combined list is the batch. Early in adaptation the replay scores are near zero, so new data wins. As old loss rises, replay takes a larger share, and it takes that share from the sources that actually degraded. Reference losses are computed once and cached. The experiments are continued pretraining of Nemotron and Qwen models from 4B to 35B on legal and German corpora.

### Novelty

Reducible holdout loss is from Mindermann et al. (2022), and CSReL already uses a reducible-loss difference to build a coreset. RoD's distinct move is to put new data and old data on that same scale and let the replay fraction fall out of the comparison. The paper says this directly: prior reducible-loss methods choose which points to train on, but they do not allocate compute between adaptation and retention. The evaluation is continued pretraining with a real or proxy pretraining corpus, not instruction tasks and not a bounded exemplar memory. The loss is adapted. The model is not.

### Results

On Nemotron legal adaptation, at matched adaptation loss, RoD cuts validation-loss forgetting from 0.125 to 0.071 relative to a tuned 10% replay mixture, which the paper calls a 43% relative reduction, and it cuts task-based forgetting from 6.7 to 4.9 points. On Qwen3.5-9B legal adaptation, RoD matches the 0.988 adaptation loss of 10% replay and cuts validation-loss forgetting from 0.075 to 0.031. On German Qwen3.5-9B it sits next to a 17% mixture: adaptation loss 1.790 versus 1.796, forgetting 0.028 versus 0.035. Against no replay, task-based forgetting falls by 60% on legal Qwen3.5-9B and by 43% and 46% on German 9B and 4B. A 4B reference is reported to recover a similar frontier for a 9B target. Every condition is one seed. The paper says small gaps should not be read as significant, and it treats consistency across families and scales as the check.

### Limitations

RoD assumes a large replay pool drawn from the pretraining distribution, or a proxy when that distribution is unavailable. The Qwen runs use Nemotron text as a proxy, and the paper says the proxy loss is a noisier measure of capability retention, especially for reasoning. The adaptation specialist is a model trained to convergence on the new corpus without replay. Inside the paper's suite that model is also the no-replay baseline, so the cost is hidden. A standalone run pays for an extra training. The scored candidate pool is a large packed corpus, not a few hundred stored instructions. Sequence length is 4096. None of the reported models fit under 700M. There is no instruction-format metric.

### Relevance

This is the paper that challenges TSR's fixed replay probability, not CSReL's choice of which stored rows to keep. The transferable piece is the competition. The non-transferable piece is the pretraining corpus and the specialist. A faithful adaptation would cache the loss of each retained example at the moment it was committed, and later score forgetting as the rise from that cached loss, instead of training a second model on data this project is not allowed to keep. Threat to "the hybrid needs a hand-set replay rate": high. Threat to "CSReL is the wrong memory writer": low. RoD cites CSReL and treats it as a different decision.

---

## [2025] TreeLoRA (ICML)

**Citation**: Yu-Yang Qian, Yuan-Ze Xu, Zhen-Yu Zhang, Peng Zhao, and Zhi-Hua Zhou. "TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree." ICML, 2025. arXiv:2506.10355.
**Links**: [Paper](https://arxiv.org/abs/2506.10355) | [AlphaXiv](https://www.alphaxiv.org/abs/2506.10355) | [Code](https://github.com/ZinYY/TreeLoRA)

### Idea

TreeLoRA builds a tree of tasks from the direction of their gradients, separately enough that similar tasks share a LoRA path and conflicting tasks split. Searching every previous task would cost a gradient per task. A lower-confidence-bound bandit walks the tree with one gradient query per round. Sparse gradient updates then train the chosen branch. The theory is a regret bound whose dependence on the number of tasks shrinks when the tree is smooth. Experiments cover ViTs on split CIFAR-100, ImageNet-R, and CUB-200, and four instruction models on the eight-task TRACE benchmark.

### Novelty

Task similarity from gradients is the same family as TSR. O-LoRA, HiDe-LoRA, and HiDe-Prompt already separate or share parameters across tasks. The new pieces are the hierarchy, the bandit that avoids a linear scan, the sparse update, and a regret statement. The problem formulation is standard class-incremental or TRACE continual learning. The loss is the task loss plus a penalty. The evaluation adds wall-clock time next to accuracy and backward transfer, which most adapter papers omit.

### Results

On split CIFAR-100, TreeLoRA accuracy is 88.54±0.05 against HiDe-LoRA at 88.46±0.04. On CUB-200 it is 73.66±0.22 against 73.48±1.35. On ImageNet-R it loses: 71.94±0.47 against HiDe-LoRA 72.28±0.15. Three runs. Two epochs already reach 87.73±0.12 on CIFAR-100, against 88.54 at 20 epochs. On TRACE with Mistral-7B, overall performance is 54.77±1.1 against O-LoRA 52.02±0.8, and backward transfer is 3.77±0.4 against 8.13±0.6. On Gemma-2B, O-LoRA's overall score is higher, 33.73±0.8 against 33.41±0.9, while TreeLoRA's backward transfer is better, 8.50±0.5 against 12.36±0.4. The paper's speed claim is up to 3.2× on ViTs and about 2.4× on LLMs against the slower contenders. Extracted time cells in the HTML tables are corrupted, so those seconds are not repeated here.

### Limitations

The accuracy margins over HiDe-LoRA on vision are smaller than a typical review would infer from "best on two of three datasets." One of the three is a loss. TRACE gains depend on the base model. The LLM runs use 1B to 7B models, and 1B already exceeds the 700M cap. The method stores a growing tree of adapters, which is a different memory from a replay buffer and is not charged in bytes of examples. The regret bound assumes a smooth tree and unbiased residual noise after sparsification. If tasks do not form that tree, the bound collapses to an ordinary bandit. There is no generative exact-match or format metric.

### Relevance

TreeLoRA is the direct answer to a known TSR weakness: one average probe gradient can cancel subskills that point in opposite directions. A layer-wise or branch-wise signature is the test, not a commitment to adopt the whole tree. The bandit matters only when the number of retained tasks makes a full probe expensive. At the short task horizons this project can store under a teacher budget, the bandit may be dead weight and a per-layer cosine may be the whole gain. Threat to TSR's single-vector router: moderate. Threat to the replay writer: low, because TreeLoRA's TRACE comparison still includes rehearsal methods and does not replace the buffer by itself.

---

## [2026] ASO-LoRA (ACL)

**Citation**: Yitong Wang, Xue Han, Wenchun Gao, Qian Hu, Jiahui Wang, Ziqing Wang, Lijun Mei, and Junlan Feng. "Soft Orthogonal Low-Rank Adaptation for Knowledge Sharing in Large Language Model Continual Learning." ACL, 2026.
**Links**: [Paper](https://aclanthology.org/2026.acl-long.842.pdf) | [Code](https://github.com/736619821/ASO-LORA) | AlphaXiv: no arXiv id confirmed

### Idea

ASO-LoRA trains a new LoRA block for each task, freezes the old blocks, and adds them together at inference so no task label is required. Hard orthogonality would push the new subspace off every old subspace. ASO-LoRA instead scales that penalty by the Spearman correlation of attribution scores between the new task and the old ones. Similar tasks are allowed to share directions. Dissimilar tasks stay closer to orthogonal. The models are T5-large, reported as 710M, and LLaMA-2-7B. The benchmarks are short topic-classification streams and a 15-task mixture of GLUE, SuperGLUE, and review datasets, in several orders.

### Novelty

O-LoRA already imposes orthogonality between task subspaces, and the paper is explicit that ASO-LoRA starts from that constraint. The addition is the attribution-score gate that softens it. Multi-LoRA inference without a task id is also already used by several mixture-of-adapters methods that appear as baselines. The problem and the datasets are the O-LoRA protocol. The loss gains one regularizer. This is an incremental change with a clear ablation target: hard orthogonality versus similarity-weighted orthogonality.

### Results

On T5-large, mean accuracy over three short orders is 77.0 for ASO-LoRA, 75.3 for O-LoRA, 72.5 for per-task fine-tuning, and 57.8 for replay. On the longer 15-task mixture the means are 72.8, 72.2, and 78.0 for ASO-LoRA, O-LoRA, and per-task fine-tuning. Order 5 of that mixture reverses the ranking: 67.2 against O-LoRA's 69.3. On LLaMA-2-7B the paper reports a 0.9 point average gain over O-LoRA on the short benchmarks. A single LoRA module is about 0.4% of the parameters. Many baseline numbers are taken from the O-LoRA paper rather than rerun.

### Limitations

A 1.7 point gain over O-LoRA, which disappears on one long order, is a weak reason to replace a router. The replay baseline at 57.8 is far below the adapter methods, so the table does not say that soft orthogonality beats a strong replay system. It says that a weak replay baseline loses to parameter isolation on these classification streams. T5-large at the paper's 710M count sits just over the 700M ceiling. LLaMA-2-7B is far over it. Attribution scores add a backward computation whose cost is not compared with TSR's ten probe batches. There is no instruction-following or format metric. Copied baselines make the historical columns hard to audit.

### Relevance

ASO-LoRA is a text-model alternative to TSR's "always transfer from the most aligned task." Its useful claim is the opposite policy: transfer only to the extent the tasks share an attribution profile, and otherwise isolate. That is a control the hybrid does not have. TSR can put mass on a mildly aligned task and then distill from it. ASO-LoRA would rather freeze that direction. Threat to TSR as a default transfer rule: moderate on classification, unknown on generative replay. The paper is not evidence that the replay buffer can be deleted.

---

## [2025] SD-LoRA (ICLR)

**Citation**: Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, and Ying Wei. "SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning." ICLR, 2025. arXiv:2501.13198.
**Links**: [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/92f43b1d33fae4aa1958f75317f0cec1-Paper-Conference.pdf) | [AlphaXiv](https://www.alphaxiv.org/abs/2501.13198) | [Code](https://github.com/WuYichen-97/SD-Lora-CL)

### Idea

SD-LoRA is a rehearsal-free class-incremental method for a pretrained vision transformer. Each task adds a LoRA direction, but the magnitude of existing directions can keep moving. The paper's analysis says the learned directions become less important as tasks accumulate, and that rescaling old directions moves the model along a path where old-task loss stays low while the new task improves. Two cheaper variants freeze or compress some of that growth. Inference uses the final model, with no task-id lookup and no stored images. The backbone in the main tables is ImageNet-pretrained ViT-B/16, with a DINO ViT as a second check.

### Novelty

InfLoRA already designs the new-task subspace to avoid old-task gradient interference. Prompt methods such as L2P, DualPrompt, CODA-Prompt, and HiDe-Prompt already avoid rehearsal. SD-LoRA's distinct claim is the magnitude-direction split, the observation that directions can be learned early and then rescaled, and the argument that this lands in an overlapping low-loss region. The problem is standard class-incremental learning. The evaluation metrics, final average accuracy and average anytime accuracy, are standard. The no-rehearsal constraint is the point, and it is a different problem from bounded replay.

### Results

On ImageNet-R with 20 tasks, Table 2 gives SD-LoRA 75.26±0.37 accuracy and 80.22±0.72 anytime accuracy, against InfLoRA at 69.89±0.56 and 76.68±0.57. Five runs. The same table shows 79.15±0.20 against InfLoRA 76.95±0.23 at 5 tasks, and 77.34±0.35 against 74.75±0.64 at 10 tasks. On ImageNet-A, SD-LoRA is 55.96±0.73 against HiDe-Prompt 42.70±0.60. The prose says SD-LoRA beats InfLoRA by 7.68 accuracy points and 4.62 anytime points on ImageNet-R with 20 tasks, and beats HiDe-Prompt by about 31.05 points on ImageNet-A. Those prose gaps do not match the tables read here, which show 5.37, 3.54, and 13.26 points. The table is the number used below. Full fine-tuning is worse than every prompt method on these splits, so it is not a serious upper bound.

### Limitations

This is class-incremental image classification with a frozen ViT and a shared classifier, not causal instruction tuning. There is no replay arm with a matched byte budget, so the paper does not show that a buffer would have lost. The magnitude vector still grows with tasks, even though images are not stored. The prose-table disagreement means the headline margins should not be quoted from the abstract. Rank is fixed at 10 on query and value projections. DomainNet gains are small. A method that works because class prototypes stay linearly separable does not automatically preserve long generated answers.

### Relevance

SD-LoRA is the boundary condition for the whole hybrid. If a rehearsal-free LoRA update matches the hybrid on the instruction stream, CSReL, SAR, and the teacher store are not earning their bytes. If it collapses once answers are long or tasks share the same output format, the buffer is doing work that subspace isolation does not. Threat to "replay is required": high on the vision benchmarks in the paper, and untested on this project's instruction setting. The correct use is a required baseline, not a component to merge into the hybrid.

---

## [2024] CoDyRA (arXiv)

**Citation**: Haodong Lu, Chongyang Zhao, Jason Xue, Lina Yao, Kristen Moore, and Dong Gong. "Adaptive Rank, Reduced Forgetting: Continual Learning with Dynamic Rank-Selective LoRA." arXiv:2412.01004, 2024. Venue: not confirmed in the text read on 2026-11-01.
**Links**: [Paper](https://arxiv.org/abs/2412.01004) | [AlphaXiv](https://www.alphaxiv.org/abs/2412.01004) | Code: the paper says code is available; the URL was not in the extracted text

### Idea

CoDyRA treats LoRA rank as the plasticity knob. The authors insert LoRA in many places in CLIP, vary the rank, and see a stable pattern: a higher rank learns the new task and forgets more, a lower rank forgets less and learns less, and the useful rank depends on the module and the task. They then learn a nonnegative importance on each rank, penalize that vector with an L1 term, prune the small ranks, and merge the survivors into the backbone. No images are stored and no task id is required at test time. The same recipe is run on TRACE with LLaMA and Gemma models.

### Novelty

AdaLoRA and other rank-allocation methods already prune LoRA ranks during ordinary fine-tuning. CoDyRA's claim is that, in a stream, the rank is specifically a forgetting regularizer, and that the right rank is not global. The empirical map from rank and placement to forgetting is the part that is more useful than the regularizer. The problem is multi-domain CLIP continual learning plus TRACE. The loss is the task loss plus an L1 penalty. Merging into the backbone distinguishes it from methods that keep a pool of adapters.

### Results

On 5-shot multi-domain task-incremental CLIP, CoDyRA's average accuracy is 73.3 against ZSCL at 64.4, and its last-task accuracy is 78.0 against RAIL at 77.2. Transfer accuracy is 69.9 against a zero-shot transfer reference of 69.4. With the paper's domain-wise autoencoder variant, average accuracy is 74.3 and last accuracy is 80.8. On unseen CIFAR-100, Places365, and ImageNet-1K, the merged model averages 58.00 against the frozen CLIP zero-shot average of 56.24. On the TRACE row labeled LLaMA-2-7B-Chat, CoDyRA reports overall performance 43.82±0.9 and backward transfer 3.25±0.5, against the TreeLoRA numbers it lists at 43.52±1.0 and 3.46±0.4. On the row labeled LLaMA-3-1B, it lists 37.46±0.8 against a TreeLoRA cell of 36.14±0.7. That 36.14 cell matches TreeLoRA's own LLaMA-3.2-1B row, so the model string is inconsistent across the two papers and should not be treated as one paired run until the checkpoints are identified.

### Limitations

The CLIP gains on last accuracy over RAIL are under a point unless the extra autoencoder is added, and that autoencoder is a task-id-like module the paper otherwise claims to avoid. TRACE edges over TreeLoRA are a few tenths of a point on the 7B model, inside the reported standard deviations. The method is rehearsal-free, so it does not answer how to spend a replay budget. Merging a pruned update into the base weights fights this project's rule that era teachers stay immutable and that a frozen base is shared. Rank shrinkage does not reduce the backward cost of a layer whose remaining rank is still nonzero, unless the implementation actually drops that path. No format metric is reported. The code URL was not verified.

### Relevance

CoDyRA is the rank-shaped alternative to aL-SAR's layer prefix. aL-SAR decides which layers receive a gradient. CoDyRA decides how many directions inside a layer are allowed to move, and it uses that limit as a forgetting brake. The cheap experiment is to sweep a fixed rank under the same token budget before adopting the L1 schedule. If a single small rank matches CoDyRA, the adaptive penalty is not the contribution. Threat to aL-SAR's Fisher rule: moderate as a hypothesis, low as evidence, because the paper never runs the Fisher baseline.

---

## [2024] LISA (NeurIPS)

**Citation**: Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, and Tong Zhang. "LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning." NeurIPS, 2024. arXiv:2403.17919.
**Links**: [Paper](https://arxiv.org/abs/2403.17919) | [AlphaXiv](https://www.alphaxiv.org/abs/2403.17919) | Code: not confirmed in the pages read

### Idea

LISA is not a continual-learning method. It is a memory-saving optimizer for ordinary fine-tuning. The authors observe that LoRA weight norms concentrate in the first and last layers while middle layers stay small. They turn that into a sampler: the embedding side and the head stay active, and a small random set of middle layers is unfrozen for a block of steps. Gamma is the expected number of active middle layers. The point of the randomness is to get a full-parameter-style update path while the optimizer state and activations of the frozen layers are not resident. The experiments fine-tune TinyLlama, Mistral-7B, LLaMA-2-7B, and, in the abstract, LLaMA-2-70B.

### Novelty

Layer freezing, progressive unfreezing, and block coordinate descent are older. The specific observation about LoRA-norm skew, and the claim that random middle-layer sampling with always-active ends beats both LoRA and full fine-tuning on instruction benchmarks, is the paper's contribution. There is no new loss. There is no replay, no task boundary, and no importance estimate beyond the decision to prefer the ends. That simplicity is why it belongs in this review.

### Results

On MT-Bench, TinyLlama moves from LoRA 1.90 to LISA 2.57, with full fine-tuning at 2.21 and GaLore at 2.61. Mistral-7B moves from LoRA 4.41 and full fine-tuning 4.64 to LISA 4.85. LLaMA-2-7B moves from LoRA 4.45 to LISA 4.94. The abstract's "10% to 35% over LoRA" matches the TinyLlama relative gap and is smaller on the 7B models. On MMLU, Mistral-7B LISA is 62.09 against LoRA 61.78 and full fine-tuning 61.70. The paper's memory table puts full fine-tuning of LLaMA-2-7B near 59 GB and the LoRA and LISA configurations near 21–28 GB at batch size 1. Those 7B runs are outside the 700M cap. TinyLlama is the in-cap model, and on that model GaLore wins MT-Bench.

### Limitations

Nothing in the paper is a forgetting result. A method that helps a single fine-tune can still erase earlier tasks, and the random schedule has no memory of which layers hold old skills. The importance distribution is taken from LoRA norms on the models they inspected. It may be wrong for a 0.5B instruct model, for a different adapter layout, or after several tasks. Gamma and the sampling period are extra hyperparameters. The paper does not show that the same layers would have been chosen by a Fisher-per-FLOP rule. Optimizer moments on frozen layers are only saved if the implementation actually drops them. A freeze that still backpropagates through the layer does not save the FLOPs aL-SAR is counting.

### Relevance

LISA is the simplicity test for aL-SAR. Before any Fisher trace, prefix search, or refresh cadence is credited, a matched run should unfreeze a random subset of layers with the same expected FLOPs, always leaving the ends trainable. If that matches aL-SAR on retention and adaptation, the Fisher criterion is not earning its bookkeeping. Threat to aL-SAR as a necessary mechanism: high as a required control, and currently unknown as an empirical result. LISA should not be described as a continual-learning upgrade.

---

## [2026] Goal-oriented error estimation for adaptive sampling of PINNs (arXiv)

**Citation**: Medard Govoeyi and Thomas Richter. "Goal oriented error estimation for adaptive sampling of PINNs." arXiv:2604.01835, 2026.
**Links**: [Paper](https://arxiv.org/abs/2604.01835) | [AlphaXiv](https://www.alphaxiv.org/abs/2604.01835) | Code: none stated

### Idea

The paper trains a neural network to solve a Poisson equation, either as a residual PINN or as a Deep Ritz energy. The quantity that matters is not the residual everywhere. It is a user-chosen functional, such as a local average of the solution. Dual-weighted residual estimation, the standard finite-element tool for that setting, writes the functional error as the residual tested against an adjoint solution. The adjoint is itself a network. Localizing that product produces a sampling density. New collocation points are drawn where the functional is sensitive, not merely where the residual is large. The experiments are low- and higher-dimensional Poisson problems with known solutions.

### Novelty

Residual-based adaptive refinement and residual-based resampling are established PINN heuristics. Becker and Rannacher's dual-weighted residual method is much older. The paper's move is to build the adjoint and the localized indicator entirely from networks, then resample collocation points from that indicator during training. The PDE, the loss, and the network are standard. The evaluation is the functional error and the effectivity of the estimator, which is the right metric for this claim and an unusual one for PINN papers that only plot the residual.

### Results

The authors report that the integrated estimator tracks the integrated true functional error with a ratio near one from early in training, including in a figure where the pointwise residual indicator oscillates in sign and would be a bad sampler. On one oscillatory Poisson case, resampling interior points from the indicator at epoch 400 leaves the functional error about one order of magnitude smaller than the uniform-point run, after averaging ten successive epochs because the curves are noisy. Other runs resample once, around epoch 200 or 300, or add points every 1,000 epochs. No LLM, no classification accuracy, and no comparison against GREATS-style gradient alignment appear. Citation count on the matched OpenAlex record is zero, which is expected for a two-month-old preprint and is not evidence against the finite-element identity.

### Limitations

The theory is for a linear functional of a Poisson solution whose energy is symmetric and positive. Instruction loss is not that energy. A network class is not a linear finite-element space, and the authors acknowledge that Galerkin orthogonality need not hold. The estimator is an approximation, not a guaranteed bound, once the true adjoint is replaced by a network. The numerical evidence is a handful of Poisson problems, one of which the authors describe as noisy. There is no demonstration that the same weight helps when the residual and the quantity of interest disagree on a discrete dataset. Using this paper as an empirical continual-learning result would be a category error.

### Relevance

This is the CFD source for a specific repair to OASIS, GREATS, and RoD. OASIS weights an example by residual size, which is the PINN residual sampler. Dual-weighted residuals weight that residual by how much it moves the quantity you named. If the quantity is old-task loss, an example with a huge current-task gradient and no effect on old tasks should lose to a smaller gradient that points at the retention probe. GREATS is the same first-order object when the validation gradient stands in for the adjoint. Threat to the pipeline's accuracy: none by itself. Threat to the justification of gradient-energy admission: high, because the finite-element identity says energy is the wrong weight unless every error component affects the declared functional.

---

## [2025] Awake replay: off the clock but on the job (Trends in Neurosciences)

**Citation**: Matthijs A. A. van der Meer and Daniel Bendor. "Awake replay: off the clock but on the job." Trends in Neurosciences, published online 21 March 2025. DOI: 10.1016/j.tins.2025.02.006.
**Links**: [Paper](https://doi.org/10.1016/j.tins.2025.02.006) | Code: not applicable | AlphaXiv: not applicable

### Idea

The review asks what awake hippocampal replay is for. The common story gives it two jobs: guiding the decision the animal is about to make, and consolidating memory during later sleep. The authors argue that the online-decision evidence is weak and mixed. They prefer two offline jobs. One is fictive learning: replayed trajectories update values after an outcome, with priority on the updates that change future choices. The other is tagging: a salient experience is reactivated while awake so that it is more likely to be replayed again during sleep and consolidated. They hypothesize that a trajectory is more likely to replay when prediction error is high, including novelty and unexpected changes in reward, stimulus, or context.

### Novelty

This is a review and a hypothesis, not a new algorithm. Complementary learning systems, prioritized replay, and synaptic tagging are all older. The sharp claim is the demotion of awake replay as an online controller, and the split between a fast tag and a later consolidation pass. That split is a statement about biological data. It is not a derived theorem about neural-network training. The evaluation is the authors' reading of rodent experiments. No new dataset is introduced.

### Results

There is no machine-learning number in the paper. The empirical content is a reinterpretation of existing replay studies, and this review did not re-audit those primary experiments. The claim that can be used without inventing a spike rate is qualitative and explicit: the authors judge online decision support to be marginally supported, and they judge tagging-for-later-consolidation and fictive value update to fit the cited evidence better. OpenAlex reports 13 citations on the journal record as of this search. That is a citation count, not a measure of how often the hypothesis has been tested in models.

### Limitations

A review can over-weight the experiments that fit the hypothesis. Prediction-error priority is a hypothesis in the paper, not a result it newly measures. Spatial trajectories in rodents are not instruction examples. Replay in the brain reactivates neural patterns, not raw pixels or raw tokens, which the authors use to distinguish biological replay from buffer replay. Nothing here estimates a byte budget, a FLOP count, or a forgetting rate for a language model. Treating the paper as evidence that a particular sampler will raise TRACE accuracy would be false.

### Relevance

The paper is a constraint on SAR and on the OASIS-CSReL handoff. SAR retrieves by discounted frequency and class similarity. The biological hypothesis says retrieval priority should rise with prediction error, and that the retrieval is a tag for a later write rather than the learning step itself. The hybrid already separates admission, retrieval, and task-end writing, but it does not let the admission decision gate the write: OASIS rejects remain eligible for CSReL. The review's tagging story predicts the opposite gate. That prediction is testable and is not, by itself, a reason to believe the gate will help. Threat to SAR's frequency score: low as evidence, moderate as an alternative hypothesis that MIR-style loss increase has to be included in the ablation.

---

## Comparison matrix

| Paper | Method class | Benchmark | Primary metric | Reported gain | Reproducible | Ablations |
|---|---|---|---|---|---|---|
| GREATS | Online goal-aligned batch selection | MMLU slice, TydiQA, Llama-2-7B / Mistral-7B | Accuracy, throughput | MMLU 54.2 vs GradNorm 50.4; ghost throughput 71.3 vs direct 4.2 | Code released | Partial: selection rules and validation size, not a task stream |
| LESS | Offline Adam-influence selection | MMLU, TydiQA, BBH | Accuracy at 5% data | Beats full data on 4 of 9 model-task cells in Table 2; loses on Llama-2-7B MMLU and BBH | Code released | Partial: baselines and transfer, no stream |
| RoD | Reducible-loss competition for replay share | Legal and German continued pretraining, 4B–35B | Adaptation loss vs forgetting | Legal Nemotron forgetting 0.071 vs 0.125 for 10% replay | Code released | Partial: fixed ratios and merging; one seed |
| TreeLoRA | Hierarchical gradient routing | Split CIFAR-100 / ImageNet-R / CUB; TRACE | Accuracy, BWT, time | CIFAR-100 88.54 vs HiDe-LoRA 88.46; ImageNet-R loss; TRACE Mistral OP 54.77 vs O-LoRA 52.02 | Code released | Partial: depth, bandit, penalty |
| ASO-LoRA | Similarity-gated orthogonality | T5-large and LLaMA-2-7B classification streams | Average accuracy | Short streams 77.0 vs O-LoRA 75.3; one long order loses | Code released | Partial: the soft gate; many baselines copied |
| SD-LoRA | Rehearsal-free decoupled LoRA | ImageNet-R/A, DomainNet, ViT-B/16 | Acc, anytime Acc | ImageNet-R 20-task 75.26 vs InfLoRA 69.89 in Table 2 | Code released | Thorough for a vision CIL paper |
| CoDyRA | Adaptive LoRA rank | CLIP MTIL; TRACE | Average / last accuracy; OP, BWT | CLIP average 73.3 vs ZSCL 64.4; TRACE edge over TreeLoRA is tenths of a point | Code claimed, URL not verified | Partial: rank maps; no Fisher arm |
| LISA | Random layer sampling | Alpaca fine-tune, MT-Bench | MT-Bench, memory | Llama-2-7B 4.94 vs LoRA 4.45; not a forgetting result | Code not confirmed | Partial: gamma and model scale |
| DWR-PINN | Adjoint-weighted sampling | Poisson PINN / Deep Ritz | Functional error | About 10× lower functional error on one oscillatory case | No code stated | Partial: uniform vs estimator |
| Awake replay | Neuroscience review | Rodent replay literature | None for ML | No ML number | Not applicable | Not applicable |

## Consensus and disagreements

Eight of the method papers agree that a large gradient, a large loss, or a large residual is not the same thing as usefulness. GREATS, LESS, RoD, and the DWR paper give four versions of the same correction: weight the update by the quantity it changes. That is established enough to stop treating OASIS energy as the default score.

They do not agree on rehearsal. SD-LoRA, ASO-LoRA, and CoDyRA get strong class-incremental or classification numbers without a buffer. RoD and the current hybrid assume a buffer and argue about how to spend it. Both can be true in different output spaces. Vision class prototypes and short classification labels are a different retention problem from long instruction targets.

TreeLoRA, ASO-LoRA, and TSR agree that gradient geometry carries task compatibility. They disagree on the action. TSR routes a teacher and a replay source. TreeLoRA routes an adapter branch. ASO-LoRA isolates unless attribution scores say otherwise. Those actions are not interchangeable.

## Solved, unsolved, and where the field sits

Rehearsal-free LoRA for class-incremental vision is maturing. Adding another orthogonal penalty there is an incremental paper. Budgeted generative replay for instruction streams is still active: the papers that measure forgetting on TRACE use 1B–7B models, and the papers that measure selection use a static fine-tune. Nobody in this set measures semantic accuracy, format validity, retained bytes, and selection-plus-training FLOPs on one stream under a 700M cap.

The persistent gap is the mismatch between the score and the quantity of interest. It has lasted because per-example gradients were expensive and because continual-learning papers could report accuracy without a byte or FLOP ledger. Two conditions have changed. Ghost inner products make an online alignment score cheap enough to compare with OASIS. Small instruct models make a TRACE-style stream fit, or nearly fit, under the project cap if the 1B runs are repeated at 0.5B. What that makes feasible is a matched substitution study, not a new selector family.

Adjacent possibilities that follow from those conditions:

- Score each OASIS candidate by its inner product with a current-task probe gradient and with an old-task retention probe, and keep energy only as a baseline.
- Let the replay probability be the fraction of slots won by retained examples whose loss rose since commit, with the commit-time loss cached so no second model is trained.
- Replace TSR's single probe vector with one vector per layer, and drop any layer whose within-task gradients cancel.

## Evaluation protocol that later experiments have to match

The instruction papers that actually measure a stream, TreeLoRA and CoDyRA, report TRACE overall performance and backward transfer over three seeds. The selection papers report the target accuracy at a declared data fraction. The replay-budget paper reports an adaptation-forgetting frontier against a sweep of fixed ratios, not against the single ratio the authors liked. This project already requires new-only, uniform replay, component ablations, and a cumulative-data reference, with access and total cost logged separately. Those controls stay. The added controls from this review are a fixed-ratio sweep, a random-layer or fixed-rank arm with the same FLOPs as aL-SAR, a rehearsal-free LoRA arm, and a probe-alignment arm whose probe is declared before the run.

Standard metrics elsewhere are average accuracy and backward transfer. Those are not enough here. Semantic correctness and format validity have to stay separate, because LESS itself shows that loss and accuracy can move apart during instruction tuning.

Benchmarks are not saturated in the instruction setting. They are close to saturated for the vision comparison between TreeLoRA and HiDe-LoRA, where the gaps are tenths of a point. A paper whose only win is 0.08 accuracy on CIFAR-100 will not carry a claim.

Known confounds: extra FLOPs spent on probes and reference models; a validation probe that overlaps the test set; a replay baseline with a tiny buffer that makes any adapter method look strong; a 7B result quoted as if it were a 0.5B result; a prose margin that the table does not contain, as in SD-LoRA; a single seed, as in RoD.

## Anchor papers

Influence cannot be ranked from the OpenAlex version counts collected here. The anchors are methodological. LESS and GREATS are the selection anchors: any admission rule has to beat random, gradient norm, and probe alignment. RoD is the replay-share anchor: any fixed probability has to beat a small sweep of probabilities and a demand rule. SD-LoRA is the rehearsal-free anchor: the buffer has to beat a method that stores no examples. TSR, OASIS, aL-SAR, and CSReL remain the system being modified. They are not re-reviewed here.

## Field orientation and lifecycle

Problem-first and active. The shared problem is sequential update under a memory and compute budget. Methods are still competing, and the instruction-stream protocol is not settled. Vision class-incremental LoRA is the maturing corner. Solution-first reading is the wrong one: the goal is not to find a new home for OASIS or for DWR.

## Brutal refinement

Composing OASIS, TSR, aL-SAR, and CSReL was a reasonable implementation boundary. It is not a research claim, and neither is replacing each box with the newest paper in that box. GREATS and LESS already do probe-aligned selection better than gradient energy in static fine-tuning. RoD already treats the replay fraction as an output. SD-LoRA already shows a setting where the buffer is unnecessary. The overlap is highest on the admission score and the replay fraction. The open measurement is whether those substitutions survive a bounded instruction memory, a 700M model, and a ledger that counts probes, teachers, and auxiliary fits.

The defensible core is three substitutions and two destructive controls. Replace energy admission with alignment to a declared retention probe. Replace the fixed replay probability with a competition against cached commit-time loss. Replace one task vector with a layer-wise vector only if the layer-wise version wins. Run random layer sampling against Fisher freezing. Run a rehearsal-free LoRA against the buffer. If the first substitution fails when the probe is replaced by noise, the goal-weighting story is wrong. If the rehearsal-free arm ties the hybrid, stop spending the review on coreset variants.

## Novelty re-check

Energy admission is already weaker than probe alignment in GREATS, and weaker than Adam-feature alignment in LESS, on the benchmarks those papers use. A fixed replay ratio is already weaker than a tuned ratio, and RoD is the current attempt to avoid tuning it. Layer routing and soft orthogonality are already published. Rehearsal-free LoRA is already published. What is not published, in the sources read, is the joint measurement of goal-weighted admission, demand-routed replay, and layer-local routing under one byte and FLOP budget on a generative instruction stream at or below 700M, with the rehearsal-free and random-freeze arms included. That measurement is an empirical comparison. It becomes a method paper only if one of those substitutions wins for a reason the ablations isolate.

## Near misses, not in the ten

LoDA (arXiv:2603.00191) splits a shared subspace from a task-private subspace and rescales the shared part. TORA (arXiv:2610.01702) routes a DistilBERT adapter by an SVD angle, with a stated threshold near 84.5 degrees, and either warm-starts or isolates. Both are additional routers. They were not deep-read to the table level after TreeLoRA and ASO-LoRA covered the same decision. BCSR is a 2023 bilevel writer that CSReL already argues against. SHARP's first preprint is May 2023, outside this window.

## Round 2 Additions (2026-11-01) — length scale of the admission gradient

Round 1 treated probe alignment as the upgrade for OASIS energy. The same LESS paper, read at Table 13 and the sequence-gradient appendix, says the unnormalized gradient already fails before any probe is chosen: averaging token gradients makes the norm fall with completion length, and the inner product then selects short answers. This round checks whether a course-sized fix of that scale is already a published selector. Grad-Mimic and LeaF are the closest new boundaries. Both use a gradient geometry that this pipeline does not compute. Neither one rescales an online energy by the supervised-token count.

### Round-2 synthesis

What is preempted: ranking examples by an inner product of mean token gradients, and calling that alignment a fix. LESS already shows that product selects short completions and loses to cosine. Grad-Mimic already projects a sample gradient onto a reference weight displacement, so "align the score with a reference model" is taken. LeaF already uses teacher-versus-student token gradients, so "look at token gradients during distillation" is taken.

What stays open: this repository's admission energy is the squared norm of a mean response-token head gradient, the reduction flag does not change that head path, and no paper in this set measures `n²` times that energy as the admission score of a bounded instruction stream under 700M. Cosine normalization is the LESS baseline, not the missing arm. A length-stratified random draw is the control that asks whether the gradient was doing anything once length is held fixed.

The defensible course core is that one scale. It is an empirical substitution with a reject rule. It is not a new selector family, and it does not revive the round-1 stack of probe admission plus demand-routed replay.

## [2025] Evaluating Sample Utility for Efficient Data Selection by Mimicking Model Weights (arXiv; ICML 2026 as stated in the paper)

**Citation**: Tzu-Heng Huang, Manjot Bilkhu, John Cooper, Frederic Sala, and Javier Movellan. "Evaluating Sample Utility for Efficient Data Selection by Mimicking Model Weights." arXiv:2501.06708, 2025. The PDF states that the work appears in the proceedings of ICML 2026.
**Links**: [Paper](https://arxiv.org/abs/2501.06708) | [AlphaXiv](https://www.alphaxiv.org/abs/2501.06708) | Code: the PDF says the code is released and the URL was not in the extracted text

### Idea

Grad-Mimic scores a training sample by how well its negative gradient points toward a fixed reference model's weights. At step t the target direction is v_t = θ_ref − θ_t, and the mimic score is the projection length of −g_i,t onto that direction, ⟨−g, v⟩ / ||v||. A high score means the sample's update walks toward the reference weights. A low score means it walks elsewhere and should be down-weighted. The online stage reweights the current batch with these scores. The offline stage averages the scores over training and uses that average as a filter. The reference is a stronger public checkpoint, not a validation set and not a loss on held-out answers. The reported systems are image classifiers and CLIP models trained from scratch on DataComp image-caption pairs.

### Novelty

Gradient-norm scores, validation-gradient alignment, and reference-model loss differences are the methods the paper positions itself against, including GREATS, LESS, and RHO-LOSS. The distinct object is the direction between two weight vectors. That direction needs no labeled probe and no second forward through a reference loss. The training problem is static curation and noise filtering. There is no task stream, no replay buffer, and no instruction-format metric. The architecture is unchanged. The contribution is the score and the two-stage use of it.

### Results

On a controlled label-noise study, the fraction of samples a mimic-score filter retains tracks the amount of noise with Pearson correlation 0.903. Across six image datasets, down-weighting low scores improves the reported training, and the paper states a 2.6× runtime reduction against the influence and AGRA competitors in its efficiency appendix. The same appendix says Grad-Mimic's runtime and memory sit near GraNd, the gradient-norm baseline, because both avoid a second validation backward. On DataComp, using a CLIP model trained on DataComp-1B as the reference, the paper states that mimic-score training reaches its curve in 20.7% fewer steps than ordinary training, and that a mimic filter stacked on existing CLIP-score filters trains with 4.7 million fewer samples. A separate quality plot reports Pearson 0.958 between average mimic score and performance gain across curated sets. The DataComp runs use batch size 4096 for 5 epochs, 12.8 million seen samples at the small scale and 128 million at the medium scale, on the final MLP of the image or text encoder. The paper says some DataComp URLs were broken, so the numbers may not match the public leaderboard.

### Limitations

The reference has to sit in a better region of weight space than the model being trained. An era teacher in this pipeline is a previous task checkpoint of the same learner, which is not that object, and the example configuration sets the distillation weight to zero. The projection uses the magnitude of the gradient along v, so a length-dependent gradient norm still enters the score. The experiments are vision and contrastive image-text training. Nothing in the pages read is an instruction stream, a 700M language model, or a forgetting metric. The 2.6× figure is against methods that compute an extra gradient, not against squaring a gradient the trainer already has.

### Relevance

Grad-Mimic closes the route "replace OASIS energy with alignment to a reference model." That alignment is published, it needs a reference the project does not have, and its efficiency win is against a second backward. The course substitution that remains is a rescaling of the energy this code already computes. Threat to summed-gradient energy: low, because the mimic score is a projection onto θ_ref − θ. Threat to goal-weighted admission: moderate, as a reminder that another alignment score exists and still depends on gradient magnitude.

## [2025] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning (arXiv)

**Citation**: Yiju Guo, Wenkai Yang, Zexu Sun, Ning Ding, Zhiyuan Liu, and Yankai Lin. "Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning." arXiv:2506.07851, 2025.
**Links**: [Paper](https://arxiv.org/abs/2506.07851) | [AlphaXiv](https://www.alphaxiv.org/abs/2506.07851) | [Code](https://github.com/RUCBM/LeaF) | Venue: not stated in the pages read

### Idea

LeaF treats some tokens in a reasoning trace as confounders that pull attention off the instruction-response relation. A large teacher and the student produce gradients on the same tokens. Tokens where those gradients disagree, or where the student's gradient is sensitive to a counterfactual edit, are marked as confounders and removed. Distillation then trains the student on the pruned sequence so its attention matches the teacher's focus on the remaining tokens. The decision is which tokens to drop inside one sequence. The decision is not which example to admit into a replay stream. Students are LLaMA-3.2-1B-Instruct, LLaMA-3.2-3B-Instruct, and Qwen2.5-Math-1.5B. Teachers in the reported table are LLaMA-3.3-70B-Instruct and Qwen2.5-72B-Instruct.

### Novelty

Token pruning and knowledge distillation are both older. The specific rule is a gradient comparison against a much larger teacher, followed by deletion of the offending tokens before the distillation loss. That changes the training sequence, not a selection weight on an intact sequence. The paper is a single-task distillation study on math, code, and multi-hop question answering. There is no continual task boundary and no memory budget. The smallest student in the table is about 1B parameters, above this project's 700M cap.

### Results

The abstract states average gains over standard distillation of 2.41 points on GSM8K, MATH-500, and OlympiadBench across the LLaMA-1B/3B and Qwen2.5-Math-1.5B students, 2.48 points on HumanEval+, LeetCode, and LiveCodeBench, and 3.24 points on HotpotQA, 2WikiMultiHopQA, and Musique. The LLaMA-3.2-1B table cells are narrower than those averages. Math average: 24.96 before distillation, 33.03 for distillation without a mask, 34.40 for the instruction mask, 34.71 for the instruction-and-response mask. Code average: 15.39, 17.39, 19.76, 19.77 on that same row order. Most of the 1B math gain is the distillation itself (24.96 to 33.03). The mask adds 1.68 points on that math average. A threshold sweep around 0.05, 0.10, and 0.15 keeps the 1B math average between 32.43 and 34.40. The paper's Figure 1 caption says that simply deleting confounders improved a small model by over 20% on a math corpus and more than 10% on a code corpus; that caption is a different comparison from the table averages above.

### Limitations

The teacher is a 70B-class model. This pipeline's teacher, when distillation is on, is an earlier checkpoint of the same learner, and the example configuration turns that term off. Pruning tokens changes the loss the student optimizes. Doing that inside OASIS would change which tokens are trained, which is a second method. Counterfactual responses are generated offline; the paper says 26k samples take about 50 minutes on 4 A100s for a 1B model. No cell in the pages read is a forgetting number, a byte cap, or a format-validity split. The 1B and 1.5B students sit above the project cap.

### Relevance

LeaF is the boundary on token gradients. It shows that a mean over tokens can hide a few tokens that dominate the update, which is the same averaging LESS deferred. It does not provide an admission score, and copying the prune would change the training objective. Threat to summed-gradient energy: low. The summed energy keeps every supervised token and only changes the example's rank. Threat to a future token-pruning arm: high, because that arm would be LeaF with a smaller teacher.

## Round-2 novelty verdict

Probe-aligned admission and demand-routed replay remain the round-1 ideas, and they stay logged. They are the wrong first course experiment. The inner product they rely on is the LESS failure mode in Table 13, and running both of them at once is two variables. Summed-gradient energy is the substitution that uses the backward OASIS already pays for. It is preempted as a research claim the moment it is described as "cosine selection" or "alignment to a teacher." Described as `n²` times the current OASIS energy, with length-stratified random as the control, it is still an untested measurement on this stream. No training was run for this round.

## Round 3 Additions (2026-11-01) — which tokens, and which constraint

Round 2 treated completion length as the defect in OASIS energy. A mean over response tokens is also a mixture of tokens the model has already fit and tokens it has not. Selective language modeling and constrained continual learning name those as separate objects: a token weight inside the loss, and a dual variable on a retention constraint. Four papers were read for that split. Their headline margins do not transfer to a 700M instruction stream.

### Round-3 synthesis

What is preempted: "train on the tokens with high excess loss" as a new objective. Rho-1 already does that with a reference model, and its self-reference variant is the no-extra-corpus version. "Set the replay rate from a loss signal" is also published: PDCL updates a dual variable from constraint violation, and MSSR compares fixed, loss-triggered, and accuracy-triggered replay on LLM fine-tunes. Multiplying gradient energy by n² is not in these papers, and it is not what they found. Rho-1 keeps the per-selected-token mean and drops tokens. PDCL never mixes a squared gradient with a loss.

What stays open: using the response tokens that still exceed a batch median as the only tokens inside this repository's OASIS energy, on one bounded instruction stream at or below 700M, with semantic accuracy and token NLL reported separately. A dual update of TSR's replay probability on one probe loss is a second, separate measurement. It has to beat a fixed probability and a loss-triggered rule, and PDCL's vision bounds do not certify it.

## [2024] Rho-1: Not All Tokens Are What You Need (arXiv)

**Citation**: Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. "Rho-1: Not All Tokens Are What You Need." arXiv:2404.07965, 2024.
**Links**: [Paper](https://arxiv.org/abs/2404.07965) | [AlphaXiv](https://www.alphaxiv.org/abs/2404.07965) | [Code](https://aka.ms/rho) | Venue: not stated in the pages read

### Idea

Rho-1 starts from the observation that a language-model loss is an average over tokens, and most of those tokens are not what later accuracy depends on. The method, selective language modeling, trains a reference model on a curated corpus, scores every token in a larger corpus by the excess loss of the training model over that reference, and then optimizes ordinary cross-entropy only on the top k% of that excess. The loss is still a mean, taken over the kept tokens rather than over the whole sequence. The setting is continual pretraining: 15B tokens of OpenWebMath, and a separate 80B general-token run. The models in the headline results are 1B and 7B.

### Novelty

Document-level filtering and high-quality data mixtures are older. The change is the unit of selection. A token can be kept inside a document that would have been discarded, and a token can be dropped inside a document that would have been kept. The reference model is the part that distinguishes a token that is hard for everyone from a token that this model has not learned. The architecture is unchanged. The protocol is one pretraining run plus a fine-tune, not a task stream with a bounded replay memory.

### Results

On the math continual-pretraining comparison, the paper says both the 1B and 7B selective runs beat the full-token baseline by over 16 points of few-shot accuracy on GSM8K and MATH, and that the selective run reaches the baseline's accuracy 5 to 10 times faster. The abstract states an absolute few-shot gain of up to 30 points across 9 math tasks on the 15B OpenWebMath corpus. After fine-tuning, Rho-1-1B and Rho-1-7B reach 40.6% and 51.8% on MATH. A token census during training puts 26% of tokens in a high-to-low loss path, 51% in low-to-low, 11% in high-to-high, and 12% in low-to-high. The 11% that stay hard are described as aleatoric. On 80B general tokens with TinyLlama-1B, the paper reports a 6.8 point average gain across 15 tasks. A self-reference setting, with no separate curated reference corpus, is reported at up to 3.3 points. Those are the authors' averages. The pages read do not include a seed count for the 16-point math gap.

### Limitations

The useful version trains a reference model on curated data before scoring. The self-reference gain is much smaller than the curated-reference gain. The 1B model is already above this project's 700M cap, and the token budget is tens of billions. Nothing in the pages read is a forgetting metric, a replay buffer, or an admission gate over examples. The 11% of persistently hard tokens are a direct warning that the maximum-loss token is not automatically the token worth learning. The paper does not claim the same gains for instruction streams whose answers are short.

### Relevance

Rho-1 is the reason a mean over every response token is a questionable admission gradient, and the reason n² times that mean is the wrong repair. The mean spends its mass on the 51% of tokens that are already fit. The repair they measured is a mask on excess loss, followed by a mean over the surviving tokens. Copying the full method would add a second pretraining run this project does not have. The piece that can be tested here is the mask, using token losses the head pass already computes, with a random mask of the same keep-rate as the control. Threat to summed-gradient energy: high. Threat to example-level probe alignment: moderate, because a probe of the diluted mean still dilutes the same tokens.

## [2023] Primal Dual Continual Learning (arXiv)

**Citation**: Juan Elenter, Navid NaderiAlizadeh, Tara Javidi, and Alejandro Ribeiro. "Primal Dual Continual Learning: Balancing Stability and Plasticity through Adaptive Memory Allocation." arXiv:2310.00154, 2023.
**Links**: [Paper](https://arxiv.org/abs/2310.00154) | [AlphaXiv](https://www.alphaxiv.org/abs/2310.00154) | Code: not in the pages read | Venue: not stated in the pages read

### Idea

PDCL writes continual learning as a constrained problem: minimize the loss on the current task subject to the loss on each previous task staying below a tolerance ε. The empirical dual puts a multiplier λ_k on each previous-task constraint. For a fixed λ the Lagrangian is an ordinary weighted sum of current loss and replay losses. The multiplier is then raised when the constraint is violated and projected back to zero when the constraint has slack. The paper uses that multiplier in two further ways. Its size ranks how hard a past task is to retain, so the replay buffer gives more slots to larger λ. At sample level, the per-example multiplier is treated as a sampling distribution over what to store. The experiments are image, speech, and medical classification with replay buffers, run inside the Mammoth setup, five seeds.

### Novelty

GEM, A-GEM, and GSS already treat forgetting as a constraint. GEM projects the current gradient so the old loss does not rise. PDCL solves the saddle point instead of projecting, and it reads the optimal multiplier as a sensitivity: how much current-task performance moves if the forgetting tolerance is loosened. The losses are assumed convex in the predictor, not in the network weights. The architecture is a standard classifier. The evaluation is class-incremental accuracy and forgetting. There is no language model and no token mask.

### Results

The paper's Figure 3 caption says that dual-variable weights improve on fixed regularization weights used by ring and reservoir replay, and that partitioning the buffer by λ improves on uniform partitions, including A-GEM's. Figure 6 compares average accuracy and forgetting after the last task. The text says the primal-dual versions are comparatively high on accuracy and low on forgetting across most buffer sizes, and that in the small-buffer speech setting the gain disappears and sample selection does not help. Figure 4 is the tolerance sweep: a loose ε drives the multipliers to zero, and a tight ε makes the problem infeasible and the multipliers diverge. The operating choice they report is near 1.1 times the unconstrained loss of that task, with selected values 0.08, 0.02, 0.05, and 0.005 on Tiny ImageNet, SpeechCommands, OrganA, and MNIST. The pages read do not contain a numeric accuracy table, so no accuracy delta is quoted here.

### Limitations

The deviation bound on the dual variables needs a function-class complexity term that grows with the square root of the parameter dimension over the number of stored samples. At 700M parameters and a small buffer that term is not a useful guarantee, and the paper says as much by tying the quality of λ to memory size and to how expressive the parametrization is. Cross-entropy is convex in the predicted distribution, not in θ. The duality gap for a deep net is therefore a modeling gap, not the bound in the convex function class. All reported tasks are classification. A fixed replay probability is exactly the hyperparameter the paper says the multiplier should replace, but only when both sides of the Lagrangian are the same loss and a tolerance is declared.

### Relevance

PDCL is the correction to any rule that compares a squared gradient with a loss rise. Those two numbers are not terms in one Lagrangian. The replay weight has to multiply a replay loss, and it has to move when that loss crosses a declared tolerance. The hybrid's branch probability is a constant in place of λ. Using PDCL's full buffer partition would also replace CSReL, which is a second change. The piece that stays testable is the scalar update of the branch probability from one probe loss. Threat to demand-routed share as written: high. Threat to OASIS energy: none. The dual variable does not rank current examples.

## [2026] MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning (arXiv)

**Citation**: Yiyang Lu, Yu He, Jianlong Chen, and Hongyuan Zha. "MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning." arXiv:2603.09892, 2026.
**Links**: [Paper](https://arxiv.org/abs/2603.09892) | [AlphaXiv](https://www.alphaxiv.org/abs/2603.09892) | Code: not in the pages read | The PDF keywords include ICML; a proceedings version was not confirmed in the pages read

### Idea

MSSR schedules replay during LoRA fine-tuning by giving each stored sample a memory strength and a stability, then triggering rehearsal from those states. Strength drops as a function of time since the sample was last trained, and a successful replay resets strength and raises stability so the next gap can be longer. The training loss is the usual mix of the new batch and the replay batch. The paper's own Figure 1 distinguishes this from three published alternatives: replay at a fixed interval, replay when loss crosses a threshold, and replay when accuracy has already dropped. The experiments fine-tune Mistral-7B, Llama-3.1-8B, Qwen2.5-7B, and Gemma2-9B. The short sequence is Alpaca-GPT4, then GSM8K, then competition math. A longer sequence has 11 tasks.

### Novelty

Experience replay, loss-triggered replay, and accuracy-triggered replay are the baselines the paper draws in Figure 1. The added state is the per-sample strength and stability, with a parametric decay the authors relate to the Ebbinghaus curve. They also write that they do not assume a language model forgets by the human curve. The schedule is the contribution. The model, the LoRA update, and the next-token loss are unchanged. There is no admission gate on the new task and no coreset writer of the CSReL type.

### Results

Table 1 reports final averages on MMLU, GSM8K, and MATH. For Qwen2.5-7B the averages are 52.1 with no replay, 52.8 with fixed replay, 53.7 with loss-triggered replay, 53.7 with accuracy-triggered replay, and 54.5 for the full MSSR method. Llama-3.1-8B is 52.5, 53.0, 53.2, 53.1, and 54.4 on that same order. Mistral-7B is 48.5, 49.2, 49.3, 49.9, and 50.8. The full method's margin over loss-triggered replay is 0.8, 1.2, and 1.5 points on those three models. The margin over training with no replay is about 2 points. Table 2 extends the comparison to 11 tasks on Gemma2-9B and the other backbones. The cells are a mixture of fractions and F1-style scores, and the paper marks winners per column. No seed standard deviation is printed in Table 1. Every backbone is 7B to 9B.

### Limitations

The interesting margin is the gap over loss-triggered and accuracy-triggered replay, and that gap is about a point on these tables. The paper's own caption says loss-triggered replay fires on noisy fluctuations, so a raw loss threshold is a jittery dual. Accuracy-triggered replay fires after the metric has already fallen. The decay curve has several fitted constants. None of the models fit under 700M. The replay set is drawn from the datasets in the stream, not from a CSReL coreset with a byte cap. A result at 7B with an unbounded training mixture does not certify a branch probability on a 0.5B model.

### Relevance

MSSR is the scheduler a replay-weight claim has to beat, and it is also the warning about borrowing a forgetting curve. The human curve is a parametric choice in the paper, and the paper says the model is not assumed to follow it. Fixed replay, loss-triggered replay, and accuracy-triggered replay already exist as arms. A new branch rule that only ties loss-triggered replay has not cleared the table the authors already published. Threat to a dual update of one probability: moderate, as a required baseline. Threat to token-level admission: low. MSSR does not choose tokens.

## [2025] Not All Tokens Are What You Need In Thinking (arXiv)

**Citation**: Hang Yuan, Bin Yu, Haotian Li, Shijun Yang, Christina Dan Wang, Zhou Yu, Xueyin Xu, Weizhen Qi, and Kai Chen. "Not All Tokens Are What You Need In Thinking." arXiv:2505.17827, 2025.
**Links**: [Paper](https://arxiv.org/abs/2505.17827) | [AlphaXiv](https://www.alphaxiv.org/abs/2505.17827) | Code: not in the pages read

### Idea

Conditional token selection compresses long chain-of-thought traces before a fine-tune. A reference model trained on high-quality reasoning data scores each thinking token by how much it contributes to reaching the answer, conditioned on the question and the answer. Tokens below a compression ratio are deleted, and the student trains on the shortened trace. The aim is a model that also emits fewer reasoning tokens at inference. The reported student in the abstract's main number is Qwen2.5-14B-Instruct. The benchmarks named in the abstract and the opening figures are GPQA and MATH500, with TokenSkip as the compression baseline.

### Novelty

Rho-1 selects tokens by excess loss during pretraining. This paper selects tokens by a conditional importance score during a reasoning fine-tune, and it deletes them from the trace rather than masking them inside an otherwise intact loss. The reference model is again a separate trained model. The problem is inference cost of long traces. There is no continual task sequence and no replay.

### Results

On GPQA with Qwen2.5-14B-Instruct, the abstract reports a 9.1 point accuracy gain together with 13.2% fewer reasoning tokens, from a 13% cut in training tokens. A 42% cut in training tokens is reported with a 5 point accuracy drop and a 75.8% cut in reasoning tokens. The opening figure caption says accuracy stays above TokenSkip across compression ratios on MATH500 for the models they plot. The pages read do not include the per-ratio table, so those curves are not quoted cell by cell. The reference-model training cost is discussed in an appendix and was not extracted as a FLOP count.

### Limitations

The student is 14B. The reference model is an extra training run. Deleting tokens changes the sequence the loss sees, which is a different edit from reweighting an OASIS score. The 42% compression result is a loss on accuracy, so "fewer tokens" is not a free improvement. No forgetting number is reported. Conditioning the score on the gold answer uses the answer at selection time, which a training-time mask may do and an evaluation-time mask must not.

### Relevance

CTS is the boundary on the slogan that some tokens do not matter. The result that supports a mask is the mild compression, which raised GPQA. The result that limits it is the aggressive compression, which lowered accuracy. A median mask inside OASIS can overshoot in the same way. It also shows that a second reference model is the tool these papers use to decide which token is real. A batch median has no such reference. Threat to an excess-token admission score: moderate as a failure mode, low as a method to copy.

## Round-3 novelty verdict

Token selection by excess loss and dual weights on replay losses are published. What is not published, in the sources read, is either one inside this hybrid's admission energy or branch probability, at or below 700M, with the other stages left untouched. The defensible core is the token mask on the energy this code already computes. The dual update of the replay probability is a later arm, and it is a correction of a dimension error in the logged demand rule rather than a new scheduler. No training was run for this round.
