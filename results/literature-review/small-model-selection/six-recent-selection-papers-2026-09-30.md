# Fifteen recent data-selection papers; six for the first reproduction cycle

**Latest reproduction recommendation (2026-10-02):** The user now wants five actual reproductions connected by a research story. See [Five linked complete reproductions](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md): **RDS+, S2L, LEAD, TSR, and OPR**. That report supersedes the first-six reproduction order below; this 15-paper review remains the broader literature map.

**Date and cutoff:** 2026-09-30; recent means a 2024-09-30–2026-09-30 first public version or peer-reviewed publication. **Decision:** review 15 works, prioritizing six distinct first reproductions: CSReL, similarity-aware retrieval (aL-SAR), RDS+, LEAD, OASIS, and transfer-selective replay (TSR). All 15 have a paper publication or first public version in the requested recent window; Mix-CPT first appeared as a 2024-07 preprint but was published at ICLR 2025. The [foundational shortlist](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md) remains useful for reservoir and other baselines.

**Reproduction honesty:** The user allows up to **1B parameters**. Use [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) as the common text learner, with identical checkpoint, tokenizer, LoRA, stream, and token budgets. TSR publishes a 0.5B language-model experiment, so its published setting is a direct reproduction target. CSReL and aL-SAR have native small-vision-model experiments and need a separate text port. Most other LLM results use 7–14B models, so their 0.5B results will test algorithm transfer, not replicate headline scores. [CSReL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html), [aL-SAR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html), [TSR](https://arxiv.org/html/2607.15587).

## Decision matrix

| Order | Paper and status | Selection axis | Small-model path | Estimated reproduction effort | Critical dependency |
|---|---|---|---|---|---|
| 1 | [CSReL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html), ICLR 2025 | **Write:** learnable/reducible-loss coreset | Native MNIST/Split CIFAR; then 0.5B text port | Medium | Trains task holdout and coreset models |
| 2 | [aL-SAR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html), ICLR 2025 | **Read:** effective-use-frequency replay | Native CIFAR small CNN; then 0.5B text port | Medium | Isolate SAR from layer freezing |
| 3 | [RDS+](https://arxiv.org/abs/2503.01807), 2025 preprint | **Pool:** representation similarity to target queries | Qwen2.5-0.5B text port | Low–medium | Query distribution and response format |
| 4 | [LEAD](https://www.vldb.org/pvldb/vol19/p426-luo.pdf), PVLDB 2025 | **Pool:** in-loop dynamic uncertainty | Qwen2.5-0.5B IDU then bandit | Medium–high | Track instance history and cluster state |
| 5 | [OASIS](https://aclanthology.org/2026.acl-long.158/), ACL 2026 | **New arrivals:** relative information and redundancy | Qwen2.5-0.5B selector port | Medium–high | Public code link was empty at check |
| 6 | [TSR](https://arxiv.org/abs/2607.15587), July 2026 preprint | **Read:** route replay from compatible earlier tasks | Published 0.5B path | High | Task signatures, era snapshots, distillation |

The effort column is a reasoned estimate, **not a measured runtime**. “Pool” refers to choosing current training examples from a candidate pool, which is related to but not identical to choosing persistent old memories. The list deliberately covers buffer **writing**, replay **reading**, online **new-example admission**, and static or iterative **current-data selection**. RDS+ and LEAD are not originally continual-learning systems, so only their selection rules can be transferred to the project's sequential setting. No published number in the six rows can be compared across papers without re-running a matched protocol.

## 1. CSReL — select examples whose loss is reducible

**Citation:** Ruilin Tong, Yuhang Liu, Javen Qinfeng Shi, and Dong Gong, *Coreset Selection via Reducible Loss in Continual Learning*, ICLR 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf) · [author code](https://github.com/RuilinTong/CSReL-Coreset-CL).

### Idea

CSReL writes a bounded replay memory by favoring samples that a model trained on the current coreset still finds hard but a model trained on the full **current task** learns well. Its score is `loss(x; coreset model) − loss(x; current-task holdout model)`. High loss on both models suggests noise or ambiguity and receives a smaller score. The paper applies this coreset construction at task boundaries and also studies streaming variants. [Paper, §§3–4](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

### Novelty

Reducible loss itself was introduced before this paper; the contribution is its greedy coreset formulation and adaptation to continual-learning memory construction. This makes the selector different from choosing merely high-loss or high-gradient examples. The paper also adds variants accounting for previous-task interference and online reservoir updates. A project that simply reuses the loss difference cannot claim that criterion as new. [Paper, §§3.2–4 and Appendix C](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

### Results

In Table 1, CSReL-CL reports final average accuracy of **38.97 ± 2.61** on Split CIFAR-10 versus **37.05 ± 3.06** for uniform sampling. On Split CIFAR-100 it reports **17.48 ± 0.21** versus **13.82 ± 1.31**. Its previous-task-aware variant is higher again in those two settings, but it changes the scoring context. These values are for the paper's class-incremental protocol, not text instruction tuning. [Paper, Table 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf).

### Limitations

The **score** uses forward passes, but obtaining the holdout model and repeatedly training coreset models is substantial work. The current-task holdout uses all of that task's training data; that is allowed at an arrival boundary but cannot be quietly extended to discarded historical data. Its task-balanced memory assumes task boundaries and labels. The author repository has ready scripts for MNIST, CIFAR, and streaming cases but notes sensitivity to random seeds. [Paper, Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/24a8d40f6656e542f3fd43bac678e71b-Paper-Conference.pdf), [code README](https://github.com/RuilinTong/CSReL-Coreset-CL).

### Relevance

CSReL is the strongest recent **memory-writing** candidate for the project's “small representative history” question. Begin with the author's Split MNIST or CIFAR script, then compare its selector with reservoir and top-loss selection while holding replay batches fixed. In the text port, train the holdout only on the newly arrived training set, never evaluation or evicted historical examples. Report total selector training cost alongside retention and stored bytes; otherwise “forward-only” can be misleading.

**First reproduction check:** Reproduce one native small-vision-model Table 1 direction at the paper's memory size. Verify the loss difference on a hand-checked batch, and log holdout training, coreset retraining, score passes, and memory separately. Then port the writer to the common 0.5B instruction stream using only current-task training data for the holdout. The text run is an adaptation.

## 2. aL-SAR — retrieve memories that have received little effective training

**Citation:** Minhyuk Seo, Hyunseo Koh, and Jonghyun Choi, *Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling*, ICLR 2025; first arXiv version October 2024. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf) · [author code](https://github.com/snumprlab/budgeted-cl).

### Idea

The paper argues that continual methods should be compared at matched computation and **total bytes**, not only memory-example count. Its similarity-aware retrieval (SAR) tracks how often stored examples have effectively been trained, discounting old uses and crediting similarity to newly trained examples. Memories with less effective exposure get higher replay probability. The full aL-SAR also freezes layers adaptively on some batches to save backward computation. [Paper, §§2–3](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

### Novelty

The reusable selection idea is **effective-use-frequency** rather than raw sample age, loss, or current gradient danger. Related class gradients estimate when training one example also teaches similar stored examples. Adaptive freezing is a separate efficiency component and must be isolated in selection experiments. The joint contribution is a practical accuracy-versus-resource comparison, which is especially close to this project's proposed impact claim. [Paper, §§2.2–3 and Appendix A.8](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

### Results

In the paper's continuous CIFAR-10 retrieval ablation, full SAR reaches **72.43 ± 0.38** final accuracy versus **70.08 ± 0.97** for random retrieval. Raw use frequency alone performs worse at **69.40 ± 0.38**, so the discounting and similarity terms matter. The combined freezing-and-SAR system changes both accuracy and total FLOPs; its number is not an isolated retrieval effect. These results come from the paper's vision protocol. [Paper, Tables 6–7](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

### Limitations

Class-gradient similarity and layer-wise Fisher information require extra statistics that depend on the model and label space. The full method is more complex than SAR-only and was not tested on a 0.5B text generator. Raw frequency without discounting can starve older memories, which the paper's ablation directly shows. The paper does not actively optimize which examples enter persistent storage. [Paper, Tables 7–8 and Appendix A.27](https://proceedings.iclr.cc/paper_files/paper/2025/file/760564ebba4797d0dcf1678e96e8cbcb-Paper-Conference.pdf).

### Relevance

Use SAR to test the **read-policy** axis under a fixed reservoir writer and fixed replay token budget. It is a timely benchmark for the project's cost claim because it accounts for computation and storage in one resource budget. Start on the author's CIFAR path, then port effective-exposure tracking to the 0.5B text model with explicit task or response-gradient similarity. Keep “SAR-only” and “SAR plus adaptive freezing” as separate rows, with auxiliary-statistic bytes charged to each.

**First reproduction check:** Run the official SAR ablation against random read at one CIFAR budget. Verify exposure counters and discounting before adding class similarity. Count both selection and training FLOPs. Only after SAR works should the adaptive-freezing component be added.

## 3. RDS+ — choose instruction examples by model representations

**Citation:** Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, and Pradeep Dasigi, *Large-Scale Data Selection for Instruction Tuning*, March 2025 preprint. [Paper](https://arxiv.org/pdf/2503.01807) · [author code](https://github.com/hamishivi/automated-instruction-selection).

### Idea

RDS+ embeds labeled query examples and candidate instruction-response pairs with the model that will be tuned. It position-weights final hidden states, pools them, and uses cosine similarity to score query-candidate pairs. A round-robin selector chooses top candidates across target queries or tasks, avoiding total domination by one query. This is a **static selection** study rather than sequential replay. [Paper, §3 and Appendix G](https://arxiv.org/pdf/2503.01807).

### Novelty

Representation-based selection was known; the paper's advance is a position-weighted RDS variant tested at much larger pool and selection sizes. It shows several fashionable selectors can underperform random sampling once the pool changes scale. The query-set and source-balance protocol are as important as the embedding formula. In this project, applying RDS+ to each incoming batch or to bounded historical candidates is an adaptation, not a published continual-learning result. [Paper, §§3–5](https://arxiv.org/pdf/2503.01807).

### Results

On its TÜLU 3 setting, Table 3 reports **76.1** average multi-task score for RDS+ versus **74.7** for balanced random selection using Llama 3.1 8B. In the TÜLU 2 scale sweep, about **326k of 5.8M** selected examples approximately match full-data performance, **50.8 versus 50.9** in the paper's aggregate. The authors note selection compute becomes worthwhile against random only around that larger subset scale in their FLOP analysis. This is evidence against assuming a gain at a tiny 1–10k pool. [Paper, Figures 3–4 and Table 3](https://arxiv.org/pdf/2503.01807).

### Limitations

The paper uses labeled query examples reflecting evaluation distributions; a project protocol must obtain these from training or a declared calibration split, never test data. Full prompt-plus-response embeddings can blend semantics with surface format and answer length. Original training is with 7–8B models and very large pools, so a 0.5B result could reverse the ordering. RDS+ also does not decide how to evict old examples in a strict stream. [Paper, §3 and Appendix G](https://arxiv.org/pdf/2503.01807).

### Relevance

RDS+ is the easiest **text-native selector logic** to port and a useful contrast to gradient/loss approaches. Store instruction, input, answer, and template separately; compare embeddings of full records, instruction-only text, and label-only text, following the paper's ablation spirit. Use a fixed training-only query set for each arrival, and report query selection and embedding costs. Its prior findings make random and balanced-random baselines mandatory even at larger budgets.

**First reproduction check:** Match the paper's position-weighted pooling and round-robin decisions on a saved candidate/query pool. Implement the causal decoder formula on Qwen2.5-0.5B-Instruct and keep query examples in a training-only calibration split. Then compare full-record, prompt-only, and target-only embeddings under identical token budgets.

## 4. LEAD — select from training-time uncertainty signals

**Citation:** Xiaotian Lin et al., *LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning*, Proceedings of the VLDB Endowment 19(3), 2025. [Paper](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) · [author code](https://github.com/HKUSTDial/LEAD).

### Idea

LEAD estimates example utility inside ordinary training rather than repeatedly scoring the entire dataset with extra model inference. Its instance-level dynamic uncertainty (IDU) combines response-token loss, a post-update loss-change approximation, and exponentially smoothed loss history. A coarse-to-fine selector first chooses difficulty/task clusters with a multi-armed bandit and then samples high-IDU examples within them. The original setting is one instruction-tuning pool, not sequential tasks with irreversible history eviction. [Paper, §§3–5](https://www.vldb.org/pvldb/vol19/p426-luo.pdf).

### Novelty

The distinctive mechanism is **using training-loop feedback** to update individual utility without repeated full-pool inference. The bandit then allocates selection to promising clusters rather than a fixed uniform mixture. Its two components should be ablated independently: IDU versus raw loss, and bandit allocation versus balanced cluster sampling. Reusing IDU as an online buffer score would be new experimental adaptation, not the paper's demonstrated use. [Paper, §4–5](https://www.vldb.org/pvldb/vol19/p426-luo.pdf).

### Results

At a 15k-example budget, 2.5% of its pool, Table 1 reports **66.62** average score for LEAD on Llama 3.1 8B, **6.31 points** above that paper's full-data row. The abstract reports 5–10× shorter training time in its evaluated configurations. These headline results use 7–8B backbones and four H800 GPUs, so they are not numeric targets for the 0.5B port. They also do not establish old-task retention after multiple arrivals. [Paper, §7 and Table 1](https://www.vldb.org/pvldb/vol19/p426-luo.pdf).

### Limitations

IDU needs per-example state; cluster features, bandit weights, and unvisited-example exploration also consume memory. The “no extra inference” claim refers to utility estimation within its own training schedule, not free selection or free clustering. A strict stream must decide what happens when a never-selected candidate is discarded, since its future utility can no longer be observed. Original evaluation is static instruction tuning rather than continual fine-tuning. [Paper, §§4–7](https://www.vldb.org/pvldb/vol19/p426-luo.pdf).

### Relevance

LEAD tests whether **dynamic learning signals** beat one-shot geometry on newly arriving records. Use response-token-normalized loss on the common 0.5B generator. Begin with IDU against raw loss and random, then add cluster bandits when instance bookkeeping is correct. Compare equal processed tokens and equal wall time to test whether selection overhead pays for itself.

**First reproduction check:** On a tiny public instruction pool, verify IDU recursion and per-record loss history with deterministic training. Run IDU-only, then cluster-bandit, with exactly equal candidate access. Only claim a full LEAD reproduction if the bandit, cluster construction, and original response-token loss are all retained.

## 5. OASIS — admit informative arrivals relative to stream history

**Citation:** Minjae Lee, Minhyuk Seo, Tingyu Qu, Tinne Tuytelaars, and Jonghyun Choi, *OASIS: Online Sample Selection for Continual Instruction Tuning*, ACL 2026; earlier visual-instruction preprint in May 2025. [Paper](https://aclanthology.org/2026.acl-long.158.pdf) · [linked repository](https://github.com/snumprlab/oasis).

### Idea

OASIS estimates each new example's informativeness with a forward-efficient approximation. Rather than selecting a fixed top-k from every batch, it normalizes scores against an exponential moving mean and variance from earlier batches. It then makes a probabilistic admission decision and adjusts selected scores to reduce similarity-driven redundancy. Its paper evaluates text and multimodal continual instruction streams. [Paper, §§3–4](https://aclanthology.org/2026.acl-long.158.pdf).

### Novelty

The useful distinction is **relative-to-stream** selection: an average-looking example in a rare high-information batch should be treated differently from one in a routine batch. A redundancy correction adds a second criterion after individual informativeness. This targets the failure of constant-per-batch top-k when batch distributions shift. It is a selection method for training examples, but the paper does not establish bounded historical storage. [Paper, §3 and Appendix A.2](https://aclanthology.org/2026.acl-long.158.pdf).

### Results

At 6.25% selection on TRACE, Table 3 reports **53.48 ± 0.17** final score for OASIS, **52.27 ± 0.03** for random selection, and **57.73 ± 0.54** for full-data training. Thus the method beats random at that budget but does not match full-data training in this reported row. The paper's broader 25% claim concerns other tested settings and should not be read as universal. The text rows use 8B models. [Paper, Table 3](https://aclanthology.org/2026.acl-long.158.pdf).

### Limitations

Appendix A.2 assumes an **infinite episodic memory** containing all encountered samples, so reduced training examples do not mean reduced historical storage. More practically, the paper-linked `snumprlab/oasis` GitHub repository was [empty when checked on 2026-09-30](https://github.com/snumprlab/oasis), despite the paper's code link. That raises independent implementation risk for a first reproduction. The score's statistical normalization also depends on a suitable warm-up and shifts in batch distribution. [Paper, Appendix A.2](https://aclanthology.org/2026.acl-long.158.pdf).

### Relevance

OASIS is the closest recent **online selection** competitor to the project's cost-saving proposal and should remain on the six-paper list despite the code issue. Implement the paper's score, prior-batch normalization, admission rule, and redundancy update from its pseudocode on a tiny stream. Run its original access regime first, then a separate strict-memory variant. Charge all historical examples to storage in the former and do not call it a fixed-memory result.

**First reproduction check:** Unit-check the paper's equations on a fabricated two-batch stream where informativeness changes sharply. Then compare fixed top-k, OASIS admission, and random at matched selected-token and forward-pass budgets on the 0.5B model. Revisit the linked repository before starting; if still empty, use the PDF as the implementation specification. The 8B headline is not a 0.5B target.

## 6. TSR — replay from earlier tasks that can help the new task

**Citation:** Yang Meng, Zhenya Liu, Zhuokai Zhao, and Yuxin Chen, *Rethinking Transfer in Continual Learning: A Replay-Based Realisation*, July 2026 preprint. [Paper](https://arxiv.org/html/2607.15587) · [author code](https://github.com/ymeng3/transfer-selective-replay).

### Idea

TSR computes a label-aware gradient signature for each arriving task before training and routes replay toward earlier tasks with similar signatures. It stores a small example subset and a frozen adapter snapshot per past task. During each update, current or routed replay data are trained with a distillation term against a selected era snapshot. The method explicitly uses replay for **forward transfer**, while distillation protects stability. [Paper, §§4–5](https://arxiv.org/html/2607.15587).

### Novelty

Task-similarity replay is not new by itself; the paper's claim is that pre-training signatures can identify helpful source tasks and that replay plus snapshot distillation separates transfer from retention. The evaluated signature is a mean gradient at shared adapter initialization, and a softmax over cosine similarities gives source weights. It directly threatens a proposal to select historical tasks solely by similarity and call that novel. Its source-compatibility/headroom analysis is also a useful test before optimizing any selector. [Paper, §§4–5](https://arxiv.org/html/2607.15587).

### Results

Unlike the other language papers here, TSR reports an official **0.5B** model setting. On TRACE-8 at 50 target examples per task, Table 3 gives **0.475 ± 0.030** final overall accuracy versus **0.451 ± 0.031** for 50%-replay ER; TSR's plasticity is **0.439 ± 0.017** versus **0.427 ± 0.024**. The paper uses five seeds and also reports 7B, NumGLUE, and order/backbone checks. Its 0.5B configuration fits the stated budget. [Paper, §6 and Table 3](https://arxiv.org/html/2607.15587).

### Limitations

Each task keeps data **and an adapter snapshot**, so total auxiliary storage grows with task count even if the example buffer is small. The signature requires gradient probes, and the full method adds distillation compute to every training step. The published 0.5B configuration should be attempted first. The paper is a 2026 preprint, so its results have not been independently validated by a peer-reviewed venue here. [Paper, Algorithm 2 and §7.2](https://arxiv.org/html/2607.15587).

### Relevance

TSR adds a different objective to the project: select history that improves **new-task learning**, not only old-task retention. Run the [author's 0.5B scripts](https://github.com/ymeng3/transfer-selective-replay) as the text reproduction anchor. Then isolate routing under a fixed writer without distillation before comparing the full snapshot-distillation recipe. Measure forward transfer and backward forgetting separately; one final average hides the tradeoff.

**First reproduction check:** Run the author's 0.5B SeqFT, ER, and TSR scripts on one TRACE-8 seed, then expand to the paper's five seeds if hardware allows. Log signature probes, replay text, and every saved adapter snapshot in the storage budget.

## Nine additional recent works: comparators and novelty boundaries

These nine are part of the 15-paper review, but are not in the first six because they require a more expensive teacher, a different task setting, a larger candidate archive, or an additional training objective. Their **paper-reported** scores use their own models and metrics. A port to Qwen2.5-0.5B-Instruct tests the mechanism and does not recreate those scores.

| No. | Work | Main selection or data question | Role at 0.5B |
|---|---|---|---|
| 7 | [Adapt-∞, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html) | Dynamic skill-cluster selection and permanent pruning | Stronger bounded-pool comparator |
| 8 | [On-Policy Replay, 2026 preprint](https://arxiv.org/abs/2605.29495) | Regenerate responses before replay | Generated-data comparator; count archive and inference |
| 9 | [KPIG, ACL 2025](https://aclanthology.org/2025.acl-long.1153/) | Replay by instruction key-part information | Format and instruction-compliance comparator |
| 10 | [SEEKR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.190/) | Select attention supervision during replay | Data versus distillation-cost boundary |
| 11 | [D3, IJCAI 2025](https://www.ijcai.org/proceedings/2025/928) | Diversity, difficulty, dependability | Static data-quality comparator; teacher-cost caveat |
| 12 | [RICo, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40732) | In-context contribution estimates | Gradient-free contribution comparator |
| 13 | [Self-Adaptive CIT, 2025 preprint](https://arxiv.org/abs/2503.15924) | Proxy-model filtering of new arrivals | Direct online systems comparator |
| 14 | [Mix-CPT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc721384c26c0bdff3ec31a7de31d8d5-Abstract-Conference.html) | Select format-alignment examples | Format-sensitive selection comparator |
| 15 | [Coresets are more than replay, 2026](https://link.springer.com/article/10.1007/s00521-026-12273-y) | Select current-task training data, not only replay memory | Native vision pilot and hypothesis check |

## 7. Adapt-∞ — dynamically select skills and prune the pool

**Citation:** Adyasha Maharana, Jaehong Yoon, Tianlong Chen, and Mohit Bansal, *Adapt-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection*, ICLR 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf) · [code](https://github.com/adymaharana/adapt-inf).

### Idea

It groups examples by gradient-derived pseudo-skills, chooses a suitable scorer for each cluster, and balances selected training examples across skills. LITE-Adapt-∞ also prunes its accumulated data pool, making it one of the closer published comparisons to a genuinely bounded historical store. The original task sequence includes visual and language instruction data. [Paper, §§4–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Novelty

Its contribution is **adaptive selection across heterogeneous skills**, not merely keeping a small replay buffer. Gradient features determine clusters, then cluster-specific importance and diversity control sample choice. Pool pruning and per-update training selection are separate budget decisions. A new project cannot claim permanent representative-memory pruning as untouched territory. [Paper, §§4–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Results

At 25,000 selected training examples per step, Table 2 reports average accuracy **50.5** versus **47.2** for random selection and forgetting rate **0.9** versus **2.1**. At 50,000, Adapt-∞ reports **51.7** average accuracy. These are multimodal-task metrics and use the paper's own forgetting definition; they are not TRACE accuracy or a 0.5B text target. [Paper, Table 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Limitations

Gradient clustering and model-dependent scoring consume compute and potentially extra retained features. The LITE experiment has a 100,000-example accessible pool and a 25,000-example training selection, so “25k” alone is not a storage bound. Visual data and specialized metrics limit direct transfer to text generation. [Paper, §§5–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Relevance

It is the strongest additional comparator for a claim about **adaptive pruning under a persistent-memory limit**. A 0.5B pilot can compare its text-compatible gradient clustering against stratified reservoir and CSReL at matched candidate access and total bytes. Defer the full multimodal selector ensemble until a simple text port produces a gain that survives its overhead.

## 8. On-Policy Replay — regenerate historical answers

**Citation:** Yan Chen et al., *On-Policy Replay for Continual Supervised Fine-Tuning*, May 2026 preprint. [Paper](https://arxiv.org/pdf/2605.29495) · [code](https://github.com/Yancey2024/OnPolicyReplay).

### Idea

The current checkpoint answers stored historical prompts; reward or self-confidence scores choose generated prompt-response pairs for later ordinary supervised replay. This shifts the replay content with the model rather than repeatedly training on fixed historical answers. Its first-stage prompt pool and final selected buffer have different sizes. [Paper, §3 and Appendix C](https://arxiv.org/pdf/2605.29495).

### Novelty

The on-policy signal enters the **data source**, not a new teacher loss. Selection operates over generated responses, so the decision includes both which prompts and which answer versions to keep. It directly overlaps any proposed method that refreshes historical exemplars before replay. [Paper, §§3.2–3.5](https://arxiv.org/pdf/2605.29495).

### Results

On Qwen2.5-7B-Instruct at 1% replay, reward-filtered OPR reports ACC **65.45** and backward transfer **−2.29**, versus **62.57** and **−4.27** for vanilla replay. At 10%, vanilla replay's ACC **66.08** slightly exceeds OPR's **65.92**; the gain is therefore metric- and budget-dependent. The paper reports three seeds. [Paper, Table 2](https://arxiv.org/pdf/2605.29495).

### Limitations

Appendix C draws candidate prompts from the historical training prompts, which is a larger access requirement than the final replay buffer indicates. Generating and scoring responses also costs inference; confidence can retain confidently wrong generations. The paper's task-order coverage is narrower than a robust deployment conclusion requires. [Paper, Appendix C and Limitations](https://arxiv.org/pdf/2605.29495).

### Relevance

At 0.5B, compare fixed-response replay with regenerated-response replay using **only the same bounded prompt store**. Preserve original and generated targets, format schemas, generation cost, and scorer outputs separately. This tests whether answer refresh adds value beyond selecting better original examples, without granting OPR a hidden archive.

## 9. KPIG — select replay by instruction key-part information

**Citation:** Yongquan He et al., *Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning*, ACL 2025. [Paper](https://aclanthology.org/2025.acl-long.1153.pdf).

### Idea

KPIG masks task-defining instruction spans and measures how much the model's predictions change. It uses low key-part information gain to choose replay targets and adds divergence regularization so the model remains sensitive to those spans. The paper also studies violations such as wrong format and excessive verbosity. [Paper, §§3–4](https://aclanthology.org/2025.acl-long.1153.pdf).

### Novelty

The selection signal is **instruction-semantic sensitivity**, not whole-example loss or embedding closeness. Its masking intervention probes whether the model actually uses critical instruction constraints. The regularizer is part of the full method and should be separated from replay selection when judging the latter. [Paper, §3](https://aclanthology.org/2025.acl-long.1153.pdf).

### Results

Evaluation covers **88 seen and 40 held-out** Super-NaturalInstructions tasks and reports improved task performance and instruction-constraint following. Appendix C reports **340 minutes** for KPIG, versus **200** for SFT and **440** for its InsCL configuration; those runs differ in other details and are not equal-cost policy comparisons. [Paper, §4 and Appendix C](https://aclanthology.org/2025.acl-long.1153.pdf).

### Limitations

Key-span extraction uses GPT-4-assisted annotations, whose quality and cost are material. Masking and divergence add evaluation/training work beyond the selected examples. A task that lacks identifiable instruction constraints gives the criterion less leverage, and published violation scores do not map directly to TRACE's overall score. [Paper, Limitations and Appendix C](https://aclanthology.org/2025.acl-long.1153.pdf).

### Relevance

KPIG is the key **formatting-aware** replay comparator. On the 0.5B testbed, use human-specified or deterministic key spans where possible, then compare key-part-based task selection with ordinary task-balanced replay. Score semantic answer correctness, valid output format, and their intersection separately; hold key-span annotation cost visible.

## 10. SEEKR — use selected internal supervision on replay examples

**Citation:** Jinghan He et al., *SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models*, EMNLP 2024. [Paper](https://aclanthology.org/2024.emnlp-main.190.pdf) · [code](https://github.com/jinghan1he/SEEKR).

### Idea

SEEKR combines a small replay sample with output and attention distillation. It selects internal attention heads according to task sensitivity and forgettability, and limits supervised layers, heads, and queries. It therefore extracts more training signal from each retained record. [Paper, §3](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Novelty

The selected object is partly **internal supervision**, not just an example ID. Attention-head choice focuses distillation where past behavior seems most fragile. Replay and output matching were known; selective head supervision is the contribution. This is a different efficiency lever from CSReL or SAR. [Paper, §3](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Results

For LLaMA-2-7B-Chat on TRACE order 1, **1% SEEKR** reports OP **54.99** and BWT **−2.61**. Ordinary **1% replay** gives **48.47** and **−9.69**; ordinary **10% replay** gives **55.67** and **−3.96**. The result supports improved retention per replay example in that protocol, not a general reduction in total compute or state. [Paper, Table 1](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Limitations

Teacher attention/output signals, head selection, and stored or recomputed supervision add memory and computation. The main TRACE evaluation uses reasoning-augmented examples and substantial multi-GPU resources. A 0.5B model has different head capacity, so the ranking of useful heads might change. [Paper, §4 and Limitations](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Relevance

SEEKR bounds the value of improving **what training extracts from the same small buffer**. Use it after a simpler writer/read-policy comparison: keep buffer IDs fixed, compare ordinary replay, output distillation, and selective attention distillation at matched total cost. This can show whether selection gains disappear when stronger replay learning is allowed.

## 11. D3 — select for diversity, difficulty, and dependability

**Citation:** Jia Zhang et al., *D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning*, IJCAI 2025. [Paper](https://www.ijcai.org/proceedings/2025/0928.pdf).

### Idea

D3 scores candidate instructions on three axes: distinctiveness, prediction difficulty corrected for generation variability, and dependability judged by an external model. It then solves a weighted coreset selection problem, potentially over multiple scoring/selection rounds. Its original setting is static instruction tuning, not a sequential replay buffer. [Paper, §§2–3](https://www.ijcai.org/proceedings/2025/0928.pdf).

### Novelty

The distinctive element is the **joint quality-diversity-dependability coreset objective** with a difficulty estimate that tries to remove variation caused merely by free-form responses. Individual diversity and loss heuristics are older; the method's combination and iterative selection are the claims to reproduce. A 0.5B port should retain separate scores to discover which term actually helps. [Paper, §3](https://www.ijcai.org/proceedings/2025/0928.pdf).

### Results

At **5% of Alpaca**, Table 1 reports D3 AlpacaEval **42.01**, versus **28.29** for random and **36.50** for instruction-following difficulty selection in the paper's setup. On the AlpacaGPT4 5% row, D3 reports **71.26** versus **60.93** random. These automated-evaluation results depend on the original model, teacher, and pool. [Paper, Table 1](https://www.ijcai.org/proceedings/2025/0928.pdf).

### Limitations

An external LLM judges dependability, creating monetary, latency, and reproducibility costs that a 0.5B learner budget does not itself cover. The weighted coreset can favor familiar task clusters, and static-pool gains do not show stability after future arrivals. Judge quality and format preference can bias which samples look dependable. [Paper, §§3–4](https://www.ijcai.org/proceedings/2025/0928.pdf).

### Relevance

D3 supplies a strong **data-quality** comparator to RDS+'s representation similarity. For a bounded experiment, score only the current arrival set, cache the teacher judgments, and include them in total cost. Compare a no-teacher diversity+difficulty ablation with full D3 before claiming the dependence score pays for itself.

## 12. RICo — estimate a sample's contribution without gradients

**Citation:** Yixin Yang et al., *RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection*, AAAI 2026. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40732/44693).

### Idea

RICo measures how a candidate example changes in-context performance on an assessment set, combining task-level and global contribution. It then trains a lightweight selector on those contribution estimates to reduce repeated expensive inference over large pools. The selection target is a static instruction-tuning subset. [Paper, method and experiments](https://ojs.aaai.org/index.php/AAAI/article/view/40732/44693).

### Novelty

Its **fine-grained in-context contribution target** replaces gradient or simple similarity proxies, while the learned selector amortizes that costly target. This puts assessment-set construction at the center of the method: labels, task balance, and response formats in that set influence what counts as valuable. It is not evidence that gradient-free selection has zero inference cost. [Paper, §§1–3](https://ojs.aaai.org/index.php/AAAI/article/view/40732/44693).

### Results

On LLaMA3.1-8B, **15%** RICo-selected Alpaca data reaches average benchmark score **43.37** versus **37.95** for full-data tuning, a **5.42-point** gain. The paper also evaluates Qwen2.5-3B and LLaMA2-7B, but the gain varies by model and subset; its LLaMA2-7B 15% row falls slightly below full data. These are aggregate static-tuning scores. [Paper, Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/40732/44693).

### Limitations

Obtaining contribution labels still needs assessment-set model calls; the inexpensive selector applies only after that supervision is collected. Any assessment examples drawn from the final test distribution could contaminate selection. The paper does not evaluate irreversible historical eviction or future task retention. [Paper, method and Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/40732/44693).

### Relevance

RICo is a demanding **utility-estimation** comparator for RDS+ and LEAD. A 0.5B pilot can label a small training-only assessment set, measure assessment-call cost, and compare the learned scorer with random and representation retrieval. Do not use final test examples as in-context probes or hide the amortization set from the storage bill.

## 13. Self-Adaptive CIT — update a proxy filter as new data arrive

**Citation:** Peiyi Lin, Fukai Zhang, Kai Niu, and Hao Fu, *Towards Automatic Continual Learning: A Self-Adaptive Framework for Continual Instruction Tuning*, March 2025 preprint. [Paper](https://arxiv.org/pdf/2503.15924).

### Idea

The framework filters newly arrived instruction data using length, diversity, and an instruction-following difficulty score computed by a small proxy; the proxy is updated as the deployed model changes. An automated loop trains a candidate adapter, evaluates it, and accepts or rolls back the update. In its experiments the proxy is **Qwen2.5-0.5B-Instruct**, while the tuned model is Qwen2.5-14B-Instruct. [Paper, §§3–4](https://arxiv.org/pdf/2503.15924).

### Novelty

The key distinction is **keeping the inexpensive selector aligned with an evolving target model** rather than freezing a one-time data ranking. The deployment loop adds acceptance and rollback, but those system features should be separated from the data-filtering gain. This directly overlaps any proposal to use a small model as an online data-selection proxy. [Paper, §§3.3–3.6](https://arxiv.org/pdf/2503.15924).

### Results

The paper reports a **66.7% reduction** in selected training data and associated update computation in its medical-domain configuration, with improved exact-diagnosis accuracy against its full-data row. It simulates five update batches from a 3M-example source and also tests IMCS; this is domain-specific system evidence, not a guarantee for TRACE or other formats. [Paper, §§4.1–4.2](https://arxiv.org/pdf/2503.15924).

### Limitations

The learner is 14B, beyond the project's ceiling; only the proxy matches 0.5B. The medical examples are generated and threshold values are fitted to their distribution, so results could shift on short, multi-format instruction tasks. The paper's cost reduction should be read alongside proxy scoring, embedding, checkpoint evaluation, and any LLM-judge calls. [Paper, §4.1](https://arxiv.org/pdf/2503.15924).

### Relevance

This is a direct recent **online incoming-data filter** baseline for OASIS. At 0.5B, the same model can act as learner and scorer, but the paper's separate-proxy advantage is then not being numerically reproduced. Compare fixed and refreshed proxy scores using only seen examples; publish acceptance rate, old/new task scores, and full update time.

## 14. Mix-CPT — spend a small selected set on format alignment

**Citation:** Jinhao Jiang et al., *Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment*, ICLR 2025; first preprint July 2024. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf).

### Idea

Mix-CPT first mixes domain documents, instruction-response pairs, and alignment pairs during continual pretraining, with logit-swap self-distillation to protect earlier capabilities. It then selects a small set of previously encountered formatted SFT/DPO examples using a format alignment score (FAS). This is domain adaptation with two training stages, not the same setup as a stream of ordinary instruction updates. [Paper, §§2–3](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf).

### Novelty

FAS compares answer loss conditioned on a formatted query with loss on the answer alone, selecting examples whose **format-conditioned behavior remains difficult**. The paper's larger contribution is separating knowledge acquisition from learning response form. It therefore preempts a simple claim that representative examples should be chosen by prompt-format difficulty. [Paper, §2.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf).

### Results

For LLaMA3-8B-Base adapted to medicine, Table 3 reports target-domain average **63.4** for Mix-CPT versus **55.7** for ordinary CPT; general-domain average is **47.6** versus **43.7**. The format-selection ablation reports favorable balance for high-FAS examples, but those full-system numbers also include mixed pretraining and distillation. They cannot be credited to FAS alone. [Paper, Table 3 and §3.3](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf).

### Limitations

Its first public preprint predates the strict two-year cutoff; the **ICLR 2025 publication** is the reason it appears here. It requires documents plus SFT and preference data, and its format scorer depends on exact rendering. The original base-to-chat training stages and 8B results do not transfer unchanged to an already instruction-tuned 0.5B model. [Paper, §§2–3](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf).

### Relevance

Use FAS as a **format-selection diagnostic**, not as a claimed reproduction of all Mix-CPT. On the common 0.5B stream, compare FAS-based replay admission with ordinary answer loss while holding record IDs and templates fixed. Report task correctness and valid serialization separately so a schema gain is not mistaken for learning new facts.

## 15. Coresets are more than replay — select current-task training data

**Citation:** Elif Ceren Gok Yildirim, Murat Onur Yildirim, and Joaquin Vanschoren, *Coresets are more than replay: a data-centric view of continual learning*, Neural Computing and Applications, June 2026. [Paper](https://link.springer.com/article/10.1007/s00521-026-12273-y) · [code](https://github.com/ElifCerenGokYildirim/Coreset-CL).

### Idea

It evaluates coreset methods for **current-task training samples**, while several continual learners may also maintain replay memory. Each new task receives a warm-up on all its examples, then the remaining epochs train on a selected subset. This tests the assumption that a continual learner must process every current example at every epoch. [Paper, §§3–4](https://link.springer.com/article/10.1007/s00521-026-12273-y).

### Novelty

The work is an empirical cross-method study, not a new coreset algorithm. Its useful distinction is that **training-data selection and memory selection are different decisions** with interacting effects. It challenges a project that optimizes only historical storage while leaving incoming-data cost untouched. [Paper, §§1 and 3](https://link.springer.com/article/10.1007/s00521-026-12273-y).

### Results

Across Split-CIFAR10/100 and Split-ImageNet100, some coreset/learner combinations improve incremental accuracy and retention; the authors also report that random subsets are a strong baseline. The paper explicitly observes greater underfitting risk on smaller-per-task CIFAR100 and ImageNet100. This conditional result is more useful for the project than a single best-row number: a selector's effect depends on data volume and learner. [Paper, §§4–5](https://link.springer.com/article/10.1007/s00521-026-12273-y).

### Limitations

The warm-up has already processed **all current-task data**, so the reported reduced later-epoch exposure is not a pure one-pass stream. The benchmark is vision classification rather than instruction generation, and no output-format behavior is measured. Some selection methods add model-scoring passes whose cost must be included before claiming end-to-end savings. [Paper, §§3–5](https://link.springer.com/article/10.1007/s00521-026-12273-y).

### Relevance

This paper motivates a **two-axis** experiment: how many arriving examples train the model and how many old examples remain accessible. On 0.5B, compare current-task random/uncertainty/representation subsets at the same warm-up and replay budgets, then test whether any benefit survives without a full-data warm-up. Its native vision setting is also a low-cost check of that interaction before the text port.

## One comparison protocol that makes the first six useful together

**Stages.** Stage A verifies available native settings: CSReL and aL-SAR on their small vision models, and TSR on its published **Qwen2.5-0.5B-Instruct** TRACE-8 setting. Stage B uses that same 0.5B text learner for a matched instruction stream and ports all six selection mechanisms. Start with TRACE-8 because TSR provides a native 0.5B anchor; run a smaller [CLINC150](https://github.com/clinc/oos-eval/blob/master/data/data_full.json) text-to-label stream if per-method debugging requires shorter answers. CSReL writes memory; aL-SAR/TSR read it; RDS+/LEAD/OASIS choose trainable examples from a candidate set. Compare policies within those axes first, then factorial combinations. Stage B is an **algorithm-port comparison** for five of the six, not numerical reproduction of 7–8B headline scores. Follow with the nine additional works only where a specific claim survives the first-six tests. [TSR paper](https://arxiv.org/html/2607.15587).

**Canonical data and formatting.** Use versioned JSONL records containing `id`, `source`, `split`, `arrival_id`, `task_id`, `instruction`, `input`, `target`, `token_count`, `template_id`, and `format_schema`. Pin tokenizer, chat template, truncation rule, and **response-only loss mask**; never let test examples become RDS+/RICo queries or selector candidates. Store canonical record IDs rather than only rendered strings so prompt-only, target-only, and full-record features can be compared without changing the actual examples. Score **semantic correctness**, **schema/format validity**, and **joint correctness** separately with a fixed parser. Include a meaning-preserving prompt paraphrase and a declared schema-change test; hold example IDs and answer semantics fixed so formatting drift is not confused with knowledge loss. [KPIG](https://aclanthology.org/2025.acl-long.1153/), [Mix-CPT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc721384c26c0bdff3ec31a7de31d8d5-Abstract-Conference.html).

**Matched resources.** Hold stream order, initialization, optimizer, LoRA rank, replay schedule, and evaluation split fixed. Compare at equal stored examples **and** equal serialized/token bytes, then at equal processed training tokens and equal total wall time. Count CSReL holdout and coreset-model training, SAR gradient/statistic updates, RDS+ embedding/query scans, LEAD cluster/IDU state, OASIS selection forwards, and TSR signature probes and adapter snapshots. For later comparators, include OPR generation, D3 external-judge calls, RICo assessment probes, and Mix-CPT's extra training stages. A method requiring access to every past candidate is a different storage regime from one updating only current arrivals and its retained buffer. Include new-only, uniform/reservoir replay, task-balanced replay, and cumulative-data training as references.

**Acceptance and novelty boundary.** Report the full stage-by-task matrix, final old/new performance, backward transfer or forgetting, forward transfer, selector time, total cost, retained bytes, semantic correctness, format validity, and their joint rate. Multiple stream seeds and at least two task orders are needed before preferring one method. The defensible project target is a better **retention–adaptation–format–storage–total-cost** frontier under a precisely bounded historical-access rule. Informative memory writing, selective replay reading, online arrival selection, dynamic pool pruning, regenerated answers, and format-aware selection all have recent precedent. Combining existing scores without a new tested mechanism or stronger empirical insight is unlikely to support algorithmic novelty.

## Why some other papers were deferred

[Goldilocks replay](https://arxiv.org/abs/2406.09935) is a useful learning-speed selector with an ICLR 2025 review version, but its first public version predates the strict two-year cutoff and it is vision-only. [InsCL](https://aclanthology.org/2024.naacl-long.37/) is a direct instruction-aware replay baseline, but its first version and NAACL publication predate the cutoff; include its method in experiment baselines without representing it as recent. [Self-Synthesized Rehearsal](https://aclanthology.org/2024.acl-long.77/) similarly provides a missing-history alternative but is older. Older reservoir, herding, and gradient-selection methods remain essential baseline algorithms even though the 15-paper narrative emphasizes newer work.

## Evidence status

This update combines targeted primary-paper reading for the first six and source-grounded review of nine additional papers, with paper-specific depth noted in their summaries. OASIS's linked repository was empty at the 2026-09-30 check; the other five first-six repositories had visible implementation material, but none was installed or executed. Code availability for the nine additions was not uniformly audited. No training, result reproduction, or independent numerical validation has been performed. All numerical results above are paper-reported and retain their native protocols and model-scale boundaries.
