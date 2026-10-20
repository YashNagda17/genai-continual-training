# Literature review: Cost-efficient incremental fine-tuning

Date and search cutoff: **2026-09-28**. Main review: **exactly 15 works**. Scope: **2023-09-28 through 2026-09-28**, including newer versions/publications where identified. Input: [Idea.md](/Idea.md). Workflow: [literature-review skill](/skills/literature-review.md), [orchestrator](/SKILLS.md), and the local general, execution, and script guidelines.

## What this means for the project

**Updating a model on new data plus a small historical sample is already an established approach, called experience replay or rehearsal.** InsCL, SEEKR, SSR, and On-Policy Replay directly overlap with that part of `Idea.md`. A replay-percentage sweep is a useful starting experiment, but by itself is unlikely to establish a new method. [InsCL](https://aclanthology.org/2024.naacl-long.37/), [SEEKR](https://aclanthology.org/2024.emnlp-main.190/), [SSR](https://aclanthology.org/2024.acl-long.77/), [OPR](https://arxiv.org/abs/2605.29495).

**Recommended project question:** Under a fixed historical-storage budget and a measured total update-cost budget, how much replay is needed to remain within a predefined tolerance of cumulative-data fine-tuning, while preserving both task correctness and output-format compliance across successive arrivals? This is a proposed empirical framing, **not a verified novelty claim**. OASIS assumes infinite episodic memory; OPR constructs its small final buffer using historical prompts; Adapt-infinity already studies permanent pruning. These are different resource assumptions that a fair comparison must expose. [OASIS, Appendix A.2](https://aclanthology.org/2026.acl-long.158.pdf), [OPR, Appendix C](https://arxiv.org/pdf/2605.29495), [Adapt-infinity, §4.4](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

The working setting is **text LLM supervised/instruction fine-tuning**. `Idea.md` does not specify a model, domain, GPU budget, dataset, or the earlier data-formatting discussion. Consequently, the formatting section below covers instruction/chat examples, raw-document conversion, chronological batches, and buffer records; these are recommendations awaiting those project choices, not decisions recovered from an earlier discussion. Raw-text continual pretraining and multimodal papers are explicitly marked as adjacent evidence.

Success criterion for this review: 15 source-verified works, five substantive paragraphs per work, identifiable empirical evidence and limitations, a comparison of resource assumptions, and concrete data/evaluation decisions that make an initial project feasible. All 15 received **targeted full-text reading of methods and empirical sections**, including limitations or analysis where available. This is not an exhaustive systematic review, a code audit, or a reproduction of any result.

## Search and evidence boundaries

Search combined the repository's OpenAlex helper with targeted searches of ACL Anthology, arXiv, proceedings, and author repositories. Query families included continual instruction tuning plus replay; incremental fine-tuning plus historical-data budgets; TRACE/CITB; self-synthesized rehearsal; instruction-format forgetting; continual-pretraining compute equivalence; online sample selection; and bounded-memory/token-budget novelty checks. The preserved API query, eight exact final-check queries, selection decisions, and publication caveats are in [search-log.md](/results/literature-review/search-log.md). Machine-readable records are in [candidates-ranked.json](/results/literature-review/candidates-ranked.json), with the final order determined manually by relevance and coverage, not an automated citation score.

The OpenAlex broad query returned 50 records heavily skewed toward September 2026 because the helper sorts by publication date. It was a discovery input, not a representative ranking. Primary-source verification supplied the selected works. Full texts and source metadata are archived under `sources/`; local IDs P01–P14 and P16 identify the selected works. P15 is a reserve paper, displaced by the closer OASIS paper to keep the requested count at 15.

Publication caveats matter: TRACE is cited here as the verified **2023 arXiv version**; a NeurIPS submission does not establish acceptance. I-LoRA's reviewed 2024 preprint and 2025 ICASSP publication have different author metadata. MER-CPT's reviewed 2025 preprint corresponds to CoLLAs 2025, with PMLR proceedings published in 2026 and a different published author list. The bibliography identifies the versions actually used rather than silently combining metadata. [I-LoRA publication](https://ieeexplore.ieee.org/document/10889361/), [MER-CPT proceedings](https://proceedings.mlr.press/v330/abbes26a.html).

## Comparison and reading priority

Numbers are **paper-specific results**, not a common leaderboard. OP/ACC, relative gain, ROUGE-L, F1, loss, and forgetting rate cannot be compared across rows. “Code linked” means an author link was located; no implementation was executed.

| Priority / ID | Work and status | Method / data | Reported evidence | Reproducibility and ablation evidence | Project role |
|---|---|---|---|---|---|
| 1 / P01 | [InsCL, NAACL 2024](https://aclanthology.org/2024.naacl-long.37/) | Instruction-aware replay; 16 tasks | Table 3 average relative gain: 93.71 vs 92.01 random | Task-order and component ablations; code availability unverified | Closest selection baseline |
| 2 / P02 | [SEEKR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.190/) | Replay plus attention distillation; TRACE/SuperNI | TRACE order 1 OP: 54.99 at 1% vs replay 48.47 | Code linked; head/layer/query and replay ablations | Strong low-replay competitor |
| 3 / P03 | [KPIG, ACL 2025](https://aclanthology.org/2025.acl-long.1153/) | Key-instruction masking and replay | SupNatInst-ST: 340 min vs SFT 200 min; extra quality objectives | Data/template/method ablations; code availability unverified | Formatting and evaluation anchor |
| 4 / P04 | [OPR, 2026 preprint](https://arxiv.org/abs/2605.29495) | Regenerated historical responses; TRACE | At 1%: BWT −2.29 vs −4.27 vanilla replay | Code linked; three seeds, budget/scorer ablations | Recent direct competitor |
| 5 / P16 | [OASIS, ACL 2026](https://aclanthology.org/2026.acl-long.158/) | Online informative/diverse selection; text and multimodal | TRACE at 6.25%: final 53.48 vs 57.73 full-data | Code linked; three seeds, t-tests, component/scale/order analyses | Compute-selection and storage boundary |
| 6 / P05 | [SSR, ACL 2024](https://aclanthology.org/2024.acl-long.77/) | Synthetic rehearsal; SuperNI | Ten-task ROUGE-L: 63.23 vs 61.49 random 10% | Code linked; synthesis/refinement/selection analyses | Missing-history alternative |
| 7 / P06 | [Adapt-infinity, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html) | Gradient clusters and pruning; multimodal stream | At 25k: average 50.5 vs random 47.2 | Code linked; selectors, representation and pool-size ablations | Strong adjacent pruning competitor |
| 8 / P07 | [CITB, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.633/) | Instruction benchmark; 19/38-task streams | Template change: AR 13.9 → 38.0 | Code/data linked; templates/order analyses, three seeds | Small-model pilot and data protocol |
| 9 / P08 | [TRACE, 2023 preprint](https://arxiv.org/abs/2310.06762) | Eight heterogeneous continual tasks | Llama-2-chat-13B GSM8K: 28.8 → 2.0 in reported setting | Code/data linked; reasoning augmentation and baseline analysis | Main heterogeneous benchmark |
| 10 / P09 | [O-LoRA, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.715/) | Orthogonal adapters; text tasks | MMLU after CL: 33.6 vs 23.3 ordinary adapter CL | Code linked; rank/scale analyses and three runs | No-example-replay comparison |
| 11 / P10 | [I-LoRA, 2024 preprint; ICASSP 2025](https://arxiv.org/abs/2402.18865) | Fast/slow adapters plus real replay | NumGLUE-cm forgetting: 9.1 vs EWC 20.6 | Code linked; interpolation and trajectory analysis | PEFT plus replay baseline |
| 12 / P11 | [Simple/scalable CPT, TMLR 2024](https://arxiv.org/abs/2403.08763) | LR scheduling plus replay; raw text | 10B average loss 1.89 vs joint 1.87 | Extensive recipe ablations; no multiple-seed training | Compute-accounting precedent |
| 13 / P12 | [Replay/gradient alignment, CoLLAs 2025](https://arxiv.org/abs/2508.01908) | MER; multilingual raw-text CPT | 6B average downstream score 77.1 vs joint 72.6 | Code linked; replay/model-scale analyses | Replay versus compute tradeoff |
| 14 / P13 | [Bonito, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.748/) | Raw documents → instruction examples | +22.1 average F1 vs −0.8 with raw-text TAPT | Code/model/data linked; data size and generator analyses | Data-conversion backend |
| 15 / P14 | [Aligned-CPT forgetting, 2024 preprint](https://arxiv.org/abs/2401.03129) | Raw-text adaptation of chat model | Chinese-prompt rep-4: 0.103 → 0.552 | Code linked; module/freezing/adapter comparisons | Output-behavior failure case |

## 1. InsCL — instruction-aware historical selection (P01)

**Citation:** Yifan Wang et al. *InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions*. NAACL 2024. DOI: 10.18653/v1/2024.naacl-long.37. [Paper](https://aclanthology.org/2024.naacl-long.37.pdf). Evidence: §3, §5, Tables 2–3, Limitations.

### Idea

InsCL allocates historical replay according to instruction similarity. It measures task distance using Wasserstein distance over instruction embeddings. InsInfo then favors informative, diverse instructions within the allocation. The evaluated learner is LLaMA-7B on a sequence of 16 instruction tasks. [Paper](https://aclanthology.org/2024.naacl-long.37.pdf).

### Novelty

The contribution is instruction-aware allocation and selection, not replay itself. Greater task distance receives more historical attention. The objective remains adaptation using current and replayed examples. Its comparison against prototype selection is particularly relevant when repetitive prompt text dominates embeddings. [Paper, §3 and §5.1](https://aclanthology.org/2024.naacl-long.37.pdf).

### Results

Table 3 reports average relative gain of 93.71 versus random replay's 92.01. No replay reaches 80.84 in that table. These are normalized relative-gain scores, not accuracy percentages. The paper's separate final-stage headline gain of 3.0 must not be substituted for the table's differently aggregated comparison. [Paper, §5](https://aclanthology.org/2024.naacl-long.37.pdf).

### Limitations

The default replay count grows as 200 times the number of previous tasks. Thus, it does not establish performance under fixed total historical storage. The authors identify sensitivity to ambiguous instructions. For this project, embedding and instruction-annotation costs also belong in the selection bill. [Paper, §5 and Limitations](https://aclanthology.org/2024.naacl-long.37.pdf).

### Relevance

Implement random replay before attempting this stronger selection baseline. Preserve instruction and instance text separately so their influence can be tested. Compare selection policies using the same retained-token and training-token budgets. **Threat: high**, because representative historical selection is already the method's central contribution.

## 2. SEEKR — extracting more retention from fewer examples (P02)

**Citation:** Jinghan He et al. *SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models*. EMNLP 2024. DOI: 10.18653/v1/2024.emnlp-main.190. [Paper](https://aclanthology.org/2024.emnlp-main.190.pdf) · [Code](https://github.com/jinghan1he/SEEKR). Evidence: §3–4, Tables 1–4, Limitations.

### Idea

SEEKR combines replay with attention and output distillation. It selects attention heads using task sensitivity and forgettability. Layer, head, and query budgets control distillation work. Historical examples consequently carry more supervision than their target answers alone. [Paper, §3](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Novelty

Its advance is selective internal distillation within replay-based learning. Ordinary rehearsal and output matching are existing components. Attention-head selection supplies the additional retention mechanism. It is therefore a stronger comparison than another random-buffer implementation. [Paper, §3](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Results

For LLaMA-2-7B-Chat on TRACE order 1, 1% SEEKR achieves OP 54.99 and BWT −2.61. Ordinary 1% replay achieves 48.47 and −9.69. Ordinary 10% replay achieves 55.67 and −3.96. This supports competitive retention with fewer examples, not equal performance on every metric. [Paper, Table 1](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Limitations

The main TRACE evaluation uses reasoning-augmented data. Attention supervision adds state and computation beyond stored text. Experiments use substantial multi-GPU resources and do not validate 70B models. A project comparison must count teacher-derived state and distinguish data savings from total cost savings. [Paper, §4 and Limitations](https://aclanthology.org/2024.emnlp-main.190.pdf).

### Relevance

Use SEEKR to test whether simpler selection leaves useful retention untapped. Reproduce its data variant before comparing published scores. Add a text-only storage measure and a separate total-state measure. **Threat: high**, because a claim that tiny replay budgets preserve performance is already directly demonstrated.

## 3. KPIG — instruction meaning and output constraints (P03)

**Citation:** Yongquan He et al. *Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning*. ACL 2025. DOI: 10.18653/v1/2025.acl-long.1153. [Paper](https://aclanthology.org/2025.acl-long.1153.pdf). Evidence: §3–4, Tables 1–4, Appendix C, Limitations.

### Idea

KPIG identifies task-defining instruction spans and compares predictions with those spans masked. Low information gain helps identify tasks for replay. Training combines ordinary supervision with divergence regularization on masked inputs. GPT-4 assists instruction diversification and key-part extraction. [Paper, §3](https://aclanthology.org/2025.acl-long.1153.pdf).

### Novelty

The method targets superficial instruction matching during continual learning. Its evaluation distinguishes task performance from instruction violations. Wrong format, out-of-scope answers, and excessive verbosity receive separate measurements. This makes formatting an evaluated capability rather than merely a preprocessing detail. [Paper, §3.4](https://aclanthology.org/2025.acl-long.1153.pdf).

### Results

The study includes 88 seen and 40 held-out SupNatInst tasks. It reports improved performance and constraint following against the compared methods. Appendix C reports 340 minutes for KPIG versus 200 for SFT and 440 for its InsCL configuration. Those timings are configuration-specific; replay counts differ across methods. [Paper, §4 and Appendix C](https://aclanthology.org/2025.acl-long.1153.pdf).

### Limitations

Key-part annotation quality is not fully evaluated against human writing. The approach adds masking, scoring, and teacher-style regularization work. Its performance and violation aggregates are not interchangeable with TRACE OP. For this project, count annotation and scoring overhead separately from optimizer time. [Paper, Limitations and Appendix C](https://aclanthology.org/2025.acl-long.1153.pdf).

### Relevance

Use its constraint categories to design format-validity checks. Keep a correct-answer score separate from valid-serialization rate. Test semantically equivalent instruction templates on the same held-out examples. **Threat: high** to claiming that instruction-format-aware replay itself is new; it remains valuable as an evaluation foundation.

## 4. On-Policy Replay — regenerate historical answers (P04)

**Citation:** Yan Chen et al. *On-Policy Replay for Continual Supervised Fine-Tuning*. arXiv:2605.29495, May 2026; **preprint**. [Paper](https://arxiv.org/pdf/2605.29495) · [Code](https://github.com/Yancey2024/OnPolicyReplay). Evidence: §3–4, Tables 1–3, Appendices B–C, Limitations.

### Idea

OPR regenerates responses to historical prompts using the latest checkpoint. It scores the generations and retains a small selection. Subsequent training uses ordinary supervised fine-tuning on these pairs plus new examples. Reward-based and self-confidence-based scorers provide two variants. [Paper, §3](https://arxiv.org/pdf/2605.29495).

### Novelty

The distinguishing change is the source of replay responses. It moves on-policy supervision into the data rather than an extra training objective. Its theoretical interpretation uses small-update distributional arguments. This is not a guarantee of retention through arbitrary long training sequences. [Paper, §3.2–3.5](https://arxiv.org/pdf/2605.29495).

### Results

At 1% on Qwen2.5-7B-Instruct, reward-filtered OPR gives ACC 65.45 and BWT −2.29. Vanilla replay gives 62.57 and −4.27. At 10%, vanilla replay's ACC 66.08 exceeds OPR's 65.92 despite worse BWT. Three seeds are reported, so both adaptation and retention should remain visible. [Paper, Table 2 and §4.1](https://arxiv.org/pdf/2605.29495).

### Limitations

Appendix C uses the historical training prompts as generation candidates. Its final buffer is capped relative to the new task's 5,000 examples. Therefore a tiny replay set does not establish tiny total retained historical data. Confidence scoring can preserve confidently wrong answers, and alternative task orders are untested. [Paper, Appendix C and Limitations](https://arxiv.org/pdf/2605.29495).

### Relevance

Treat OPR as a recent direct baseline with explicit generation cost. Store original and generated answers separately if evaluating it. Restrict its candidate prompts to the allowed memory in a bounded-storage experiment. **Threat: high**, because budget sweeps and near-zero backward degradation are already central results.

## 5. OASIS — online selection with a crucial storage assumption (P16)

**Citation:** Minjae Lee, Minhyuk Seo, Tingyu Qu, Tinne Tuytelaars, and Jonghyun Choi. *OASIS: Online Sample Selection for Continual Instruction Tuning*. ACL 2026. DOI: 10.18653/v1/2026.acl-long.158. [Paper](https://aclanthology.org/2026.acl-long.158.pdf) · [Code](https://github.com/snumprlab/oasis). Evidence: §3–4, Tables 1–5, Appendix A.2.

### Idea

OASIS selects informative examples relative to a running historical score distribution. It reduces redundant selection through gradient-similarity-based score adjustments. Its sample count can vary across batches. The method is tested on both language and multimodal continual instruction benchmarks. [Paper, §3–4](https://aclanthology.org/2026.acl-long.158.pdf).

### Novelty

Its contribution combines adaptive relative informativeness with redundancy reduction. This differs from a fixed top-k choice in every batch. The score is designed to avoid a costly full backward pass for selection. Approximation quality and the remaining forward-pass cost still matter. [Paper, §3 and Limitations](https://aclanthology.org/2026.acl-long.158.pdf).

### Results

The headline describes comparable full-data performance at 25% selection in tested settings. Table 3 also exposes limits: at 6.25% on TRACE, final accuracy is 53.48 versus 57.73 for full-data training. Adapt-infinity reaches 50.48 in that comparison. The evaluation reports three seeds and Welch tests at level 0.05. [Paper, §4.1 and Table 3](https://aclanthology.org/2026.acl-long.158.pdf).

### Limitations

Appendix A.2 explicitly assumes infinite memory storing all encountered samples. Training batches are retrieved from that episodic memory. Thus, reduced training selection does not establish reduced historical storage. Treating its full-data baseline as repeated cumulative retraining would also misdescribe the experiment. [Paper, Appendix A.2](https://aclanthology.org/2026.acl-long.158.pdf).

### Relevance

This is an essential recent comparator for an efficiency claim. Adapt its selector to the same accessible pool as other methods. Separately evaluate whether limiting its memory changes its advantage. **Threat: high** to generic efficient selection; its storage assumption leaves a concrete comparison question, not a proven novel algorithm.

## 6. Self-Synthesized Rehearsal — when historical data are unavailable (P05)

**Citation:** Jianheng Huang et al. *Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal*. ACL 2024. DOI: 10.18653/v1/2024.acl-long.77. [Paper](https://aclanthology.org/2024.acl-long.77.pdf) · [Code](https://github.com/DeepLearnXMU/SSR). Evidence: §4–5, Tables 1–5, Limitations.

### Idea

SSR synthesizes rehearsal examples using a base model and demonstrations. The latest checkpoint then refines their responses. A selection stage retains diverse synthetic pairs. New tasks are trained alongside that synthetic rehearsal set. [Paper, §4](https://aclanthology.org/2024.acl-long.77.pdf).

### Novelty

The contribution is a practical synthesis-refinement-selection pipeline for LLM rehearsal. Generative replay itself predates this paper. The stages seek to represent earlier abilities without replaying the original corpus directly. Demonstrations remain an input, so “no historical information whatsoever” would be inaccurate. [Paper, §3–4](https://aclanthology.org/2024.acl-long.77.pdf).

### Results

For ten SuperNI tasks, Llama-2-7B with SSR reaches average ROUGE-L 63.23 and BWT −1.56. Random 10% rehearsal reaches 61.49 and −4.03. Joint training reaches 64.69 average ROUGE-L. The study also examines five-task orders and synthesis components. [Paper, Tables 1–5](https://aclanthology.org/2024.acl-long.77.pdf).

### Limitations

The experiments use sampled task streams rather than real temporal updates. Synthetic answers can carry model errors and biases. SSR does not always give the best forward transfer. Generation, filtering, and storing synthetic examples must be included in any project cost comparison. [Paper, §5 and Limitations](https://aclanthology.org/2024.acl-long.77.pdf).

### Relevance

Keep this as an alternative when original examples cannot be retained. Label every synthetic record and retain its generating-checkpoint identity. Test real and synthetic replay under equal total token exposure. **Threat: high** to proposing synthetic historical rehearsal as a new mechanism, but it is optional for the first pilot.

## 7. Adapt-infinity — dynamic selection and permanent pruning (P06)

**Citation:** Adyasha Maharana, Jaehong Yoon, Tianlong Chen, and Mohit Bansal. *Adapt-infinity: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection*. ICLR 2025; arXiv:2410.10636. [Published paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf) · [Code](https://github.com/adymaharana/adapt-inf). Evidence: §4–6, Tables 2–5.

### Idea

Adapt-infinity groups examples into pseudo-skill clusters using gradient representations. It chooses scoring strategies per cluster and balances selected data across skills. LITE-Adapt-infinity additionally prunes the retained pool. The main setting is a sequence of multimodal instruction datasets. [Paper, §4](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Novelty

The advance is model-dependent, multi-selector data management over an evolving pool. Selection and permanent storage pruning are distinct operations. Semantic similarity supports pruning, while gradients support skill grouping. It already covers more than choosing representative samples once. [Paper, §4–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Results

At 25,000 training samples per step, Table 2 reports average accuracy 50.5 and forgetting rate 0.9. Random selection gives 47.2 and 2.1. At 50,000, Adapt-infinity reaches 51.7 and 0.2. Its forgetting definition differs from conventional final-stage BWT. [Paper, Table 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Limitations

The main evidence includes visual tasks and image-dependent scores. Gradient representations and scoring require additional work. LITE experiments distinguish a 100,000-example pool from a 25,000-example training budget. Consequently, neither number alone describes the method's storage-compute footprint. [Paper, §5–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/a6610efd6c767f63343a4ab28505212e-Paper-Conference.pdf).

### Relevance

Study this before claiming adaptive pruning as the contribution. Start with simpler stratified and diversity-based selectors for the text pipeline. Add gradient selection only if its total cost is justified. **Threat: high** to generic pruning novelty, with transfer to this project's text setting still requiring evidence.

## 8. CITB — a controllable instruction-stream benchmark (P07)

**Citation:** Zihan Zhang, Meng Fang, Ling Chen, and Mohammad-Reza Namazi-Rad. *CITB: A Benchmark for Continual Instruction Tuning*. Findings of EMNLP 2023. DOI: 10.18653/v1/2023.findings-emnlp.633. [Paper](https://aclanthology.org/2023.findings-emnlp.633.pdf) · [Code/data](https://github.com/hyintell/CITB). Evidence: §4–6, Tables 1–4, Limitations.

### Idea

CITB separates initial instruction tuning, sequential learning, and held-out-task evaluation. It constructs 19-task and 38-task streams from SuperNI. Task instructions and input-output examples share a text-to-text interface. This supports studying retention and generalization together. [Paper, §4](https://aclanthology.org/2023.findings-emnlp.633.pdf).

### Novelty

The contribution is a benchmark and protocol rather than a replay algorithm. It tests whether instruction tuning changes conclusions from conventional continual learning. Previously learned tasks and unseen instructions are evaluated separately. Instruction-template ablations are especially useful for this project's data questions. [Paper, §4–6](https://aclanthology.org/2023.findings-emnlp.633.pdf).

### Results

On InstrDialog++, Table 3 gives FT-init average ROUGE-L 13.9 using task IDs. Replacing them with task definitions raises it to 38.0. Adding two positive examples gives 34.8 while improving BWT from −6.9 to −2.8 relative to definitions alone. Formatting choices therefore change different objectives differently. [Paper, Table 3](https://aclanthology.org/2023.findings-emnlp.633.pdf).

### Limitations

Experiments use LM-adapted T5-small and English tasks. The authors caution against generalizing directly to larger models. ROUGE-L is an imperfect aggregate for heterogeneous tasks. Modern chat checkpoints may require harder tests and model-specific rendering. [Paper, Limitations](https://aclanthology.org/2023.findings-emnlp.633.pdf).

### Relevance

Use its smaller setting to debug the arrival and evaluation pipeline. Preserve its official held-out task separation in reproductions. Treat template variants as controlled experimental conditions rather than accidental differences. **Threat: moderate**, because it already establishes the benchmark questions but supplies a useful starting protocol.

## 9. TRACE — evaluate more than old-task accuracy (P08)

**Citation:** Xiao Wang et al. *TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models*. arXiv:2310.06762v1, 2023; publication acceptance **not verified**. [Paper](https://arxiv.org/pdf/2310.06762) · [Code/data](https://github.com/BeyonderXX/TRACE). Evidence: §4–5, Tables 1–4.

### Idea

TRACE combines eight tasks spanning domains, languages, code, and numerical reasoning. It evaluates learned-task performance alongside changes in general capabilities. Instruction following and response safety are also assessed. This broadens retention beyond the replayed task distribution. [Paper, §4](https://arxiv.org/pdf/2310.06762).

### Novelty

The contribution is an aligned-LLM benchmark and extended evaluation. Its reasoning-augmented variant adds task-specific rationales. That changes the training targets as well as their length. Original and augmented versions must therefore be distinguished in comparisons. [Paper, §4–5](https://arxiv.org/pdf/2310.06762).

### Results

The reviewed version specifies 40,000 training and 16,000 testing examples in its balanced construction. It reports Llama-2-chat-13B GSM8K accuracy falling from 28.8 to 2.0 after the evaluated training sequence. Replay improves sequential-task retention in its comparisons. These observations are version- and checkpoint-specific, not universal predictions. [Paper, abstract and §4](https://arxiv.org/pdf/2310.06762).

### Limitations

The task stream is a constructed benchmark rather than observed production arrivals. Later checkpoints may have encountered related public data. Reasoning augmentation adds generation cost and possible answer artifacts. Mixed task metrics require explicit normalization and per-task reporting. [Paper, §4–5](https://arxiv.org/pdf/2310.06762).

### Relevance

Use TRACE as the main comparable heterogeneous benchmark if compute allows. Pin its revision, sample IDs, order, and answer parser. Preserve general-capability tests outside the replay-selection process. **Threat: moderate**, because benchmark construction is already done and a new project should build on it rather than rename it.

## 10. O-LoRA — a baseline without stored examples (P09)

**Citation:** Xiao Wang et al. *Orthogonal Subspace Learning for Language Model Continual Learning*. Findings of EMNLP 2023. DOI: 10.18653/v1/2023.findings-emnlp.715. [Paper](https://aclanthology.org/2023.findings-emnlp.715.pdf) · [Code](https://github.com/cmnfriend/O-LoRA). Evidence: §3–4, Tables 2–5, Limitations.

### Idea

O-LoRA learns successive tasks with low-rank updates constrained toward orthogonal subspaces. Earlier LoRA parameters are frozen. This seeks to reduce interference without storing replay examples. Tasks use a common definition/options/text/answer schema. [Paper, §3](https://aclanthology.org/2023.findings-emnlp.715.pdf).

### Novelty

Low-rank adapter subspaces approximate historical gradient directions. Orthogonal learning and LoRA are adapted rather than introduced independently. The combination avoids storing full historical gradients. It shifts the resource tradeoff from examples toward parameter and subspace state. [Paper, §3](https://aclanthology.org/2023.findings-emnlp.715.pdf).

### Results

Table 3 reports post-CL MMLU 33.6 for Alpaca-O-LoRA versus 23.3 for ordinary Alpaca-LoRA-CL. The corresponding CL scores are 76.8 and 46.7. Initial Alpaca-LoRA MMLU is 37.5, so preservation is incomplete. Results include three runs and rank/scale analyses. [Paper, §4 and Table 3](https://aclanthology.org/2023.findings-emnlp.715.pdf).

### Limitations

Task identification is required during training. The paper does not establish performance over hundreds of tasks. No-example replay should not be interpreted as zero additional memory. A fair budget must include retained adapters and constraint state. [Paper, Limitations](https://aclanthology.org/2023.findings-emnlp.715.pdf).

### Relevance

Include a no-example-replay comparator if storage is the central constraint. Keep trainable-parameter count distinct from total resident state. Report adaptation quality alongside retention because restrictive updates can underlearn. **Threat: moderate**, since it challenges the assumption that historical examples are always necessary.

## 11. I-LoRA — parameter-efficient replay with fast and slow learners (P10)

**Citation:** Weijieying Ren et al. *Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning*. Reviewed version: arXiv:2402.18865v1, 2024. [Paper](https://arxiv.org/pdf/2402.18865) · [Code](https://github.com/which47/LLMCL) · [Later ICASSP 2025 publication](https://ieeexplore.ieee.org/document/10889361/). Evidence: §4–5, Tables 1–3, Limitations.

### Idea

I-LoRA maintains a fast adapter and a slow exponential-average adapter. Current data are mixed with an episodic replay set. Representation consistency on historical examples supplements task supervision. The design aims to balance adaptation with stability. [Paper, §4](https://arxiv.org/pdf/2402.18865).

### Novelty

The motivation comes from low-loss connections between successive fine-tuning solutions. Dual-memory replay and adapter interpolation operationalize that observation. This is not evidence that ordinary LoRA alone prevents forgetting. The method adds both replay and a consistency mechanism. [Paper, §3–4](https://arxiv.org/pdf/2402.18865).

### Results

The reviewed setup uses Llama-2-7B, rank-eight adapters, and two RTX 4090 GPUs. It samples 5,000 training and 500 evaluation examples per dataset. The results discussion reports NumGLUE-cm forgetting of 9.1 versus EWC's 20.6. The preprint's abstract and table captions differ on task counts, so reproduce the actual task list. [Paper, §5](https://arxiv.org/pdf/2402.18865).

### Limitations

The evidence is concentrated on one foundational model and selected domains. The authors leave stronger theoretical justification open. Additional adapters and replay-derived representations have storage costs. The later publication's metadata must not be spliced into the reviewed preprint citation. [Paper, Limitations](https://arxiv.org/pdf/2402.18865).

### Relevance

Use it after a simple LoRA-plus-random-replay baseline works. Test whether added state improves the retention-cost frontier. Keep adapter rank and target modules consistent where possible. **Threat: high** to presenting LoRA plus small replay as a new combination, but useful for a feasible baseline implementation.

## 12. Simple and scalable continual pretraining — account for the starting checkpoint (P11)

**Citation:** Adam Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*. TMLR 2024; arXiv:2403.08763. [Paper](https://arxiv.org/pdf/2403.08763). Evidence: §4–6, Tables 2–4, Limitations. **Adjacent setting: raw-text pretraining.**

### Idea

The recipe combines learning-rate re-warming, re-decay, and old-data replay. It continues a model already trained on an earlier corpus. Replay tokens replace new tokens under its compute-equivalent comparison. The study considers English-to-English and English-to-German shifts. [Paper, §4–6](https://arxiv.org/pdf/2403.08763).

### Novelty

Its contribution is empirical evidence that a simple recipe scales. It studies learning schedules and replay jointly. Full retraining starts from random initialization on both corpora. That comparison differs from restarting supervised fine-tuning from a pretrained base. [Paper, §5](https://arxiv.org/pdf/2403.08763).

### Results

For the 10B model, 5% replay yields average validation loss 1.89. Joint training reaches 1.87. The continuation stage uses 300B tokens versus 600B for the new joint-training run. The earlier 300B-token checkpoint already exists, so the saving concerns incremental update work. [Paper, Table 4](https://arxiv.org/pdf/2403.08763).

### Limitations

Most experiments emphasize two-corpus transitions. The authors report no multiple-seed training and possible German train-validation overlap. Loss agreement is not a guarantee of instruction-format preservation. These findings therefore motivate controls, not an automatic replay percentage for SFT. [Paper, Limitations](https://arxiv.org/pdf/2403.08763).

### Relevance

Use its compute-equivalent accounting in the project design. State which checkpoint each baseline starts from. Report both per-update costs and cumulative lifecycle costs. **Threat: high** to the broad claim of approaching retraining with less computation, although direct SFT evidence remains a different experiment.

## 13. Replay and gradient alignment — replay competes with model scaling (P12)

**Citation:** Istabrak Abbes et al. *Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models*. Reviewed 2025 arXiv version; CoLLAs 2025, [PMLR 330 proceedings, 2026](https://proceedings.mlr.press/v330/abbes26a.html). [Reviewed paper](https://arxiv.org/pdf/2508.01908) · [Code](https://github.com/chandar-lab/continual-pretraining). Evidence: §3–5, Tables 2–4, Limitations. **Adjacent setting: raw-text pretraining.**

### Idea

The work combines experience replay with periodic Reptile-style meta-updates. It uses mixed batches of incoming and historical data. The resulting MER implementation encourages compatible updates across tasks. Experiments use multilingual corpora and models from 99M to 6B parameters. [Paper, §3–4](https://arxiv.org/pdf/2508.01908).

### Novelty

Replay and meta-learning mechanisms are established components. The contribution is their implementation and scaling study in LLM pretraining. It compares spending compute on replay with increasing model size. That expands efficiency analysis beyond training fewer examples on a fixed model. [Paper, §3 and §5.4](https://arxiv.org/pdf/2508.01908).

### Results

The reviewed 6B results give average downstream score 77.1 for 50% replay plus Reptile versus joint training's 72.6. The 25% combination reaches 76.8. Its compute accounting assigns 25% and 50% replay approximately 1.33× and 2× overhead per incoming token. Those replay rates describe batch mixture, not fractions of retained history. [Paper, §5.2–5.4](https://arxiv.org/pdf/2508.01908).

### Limitations

Most experiments use three language stages in a fixed order. A five-stage extension covers one scale. This is not a small supervised-instruction deployment. Also, copied forgetting formulas require care because the printed sign convention and prose are not fully aligned. [Paper, §4.3 and Limitations](https://arxiv.org/pdf/2508.01908).

### Relevance

Treat optimization and replay budget as interacting choices. Keep the pilot model fixed before introducing a scaling comparison. Include optimizer-state reset or continuation in the run manifest. **Threat: moderate** to this SFT project, and high to generic claims that replay-cost optimization is unexplored.

## 14. Bonito — turn documents into useful supervised examples (P13)

**Citation:** Nihal Nayak, Yiyang Nan, Avi Trost, and Stephen Bach. *Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation*. Findings of ACL 2024. DOI: 10.18653/v1/2024.findings-acl.748. [Paper](https://aclanthology.org/2024.findings-acl.748.pdf) · [Code/model/data](https://github.com/BatsResearch/bonito). Evidence: §4–6, Table 3, Limitations. **Enabling work, not a continual-learning benchmark.**

### Idea

Bonito generates instruction-response pairs from unannotated text and a task attribute. It is trained using remixed instruction templates. The resulting examples support domain adaptation without manually labeled target tasks. This addresses what to do when new arrivals are documents rather than supervised pairs. [Paper, §4](https://aclanthology.org/2024.findings-acl.748.pdf).

### Novelty

The contribution is conditional task generation and its supporting training dataset. Raw-text adaptation is replaced by task-shaped supervision. This changes the learning signal, not merely the JSON container. It does not itself supply a historical-memory policy. [Paper, §4](https://aclanthology.org/2024.findings-acl.748.pdf).

### Results

The generator's training set contains 1.65 million examples. Evaluation covers seven datasets across three task types. Instruction-tuned models gain an average 22.1 F1 points with generated tasks, while raw-text TAPT loses 0.8. These are adaptation results, not retention guarantees over future arrivals. [Paper, §4 and §5.3](https://aclanthology.org/2024.findings-acl.748.pdf).

### Limitations

Small unannotated corpora can produce weaker adaptation. The reported downstream scope covers only three task types. Generated answers can be factually incorrect. In the project, source-grounded verification and generation costs must be evaluated independently of replay quality. [Paper, Limitations](https://aclanthology.org/2024.findings-acl.748.pdf).

### Relevance

Use Bonito only if the chosen arrival source lacks labels. Retain the source-document identity for every generated example. Split documents before generating related questions to prevent leakage. **Threat: low** to replay-budget novelty, but high to claiming document-to-instruction conversion itself as new.

## 15. Forgetting in aligned continual pretraining — output behavior can regress (P14)

**Citation:** Chen-An Li and Hung-Yi Lee. *Examining Forgetting in Continual Pre-training of Aligned Large Language Models*. arXiv:2401.03129, 2024; **preprint**. [Paper](https://arxiv.org/pdf/2401.03129) · [Code](https://github.com/lca0503/Llama_tw). Evidence: §2–4, Tables 1–4, Limitations. **Adjacent setting: raw-text adaptation of a chat model.**

### Idea

The study continues pretraining an aligned Llama-2-7B-chat model on Traditional Chinese text. It examines output language, repetition, knowledge, and reliability. Freezing modules and parameter-efficient updates are compared. The question is whether domain/language adaptation preserves useful chat behavior. [Paper, §2–4](https://arxiv.org/pdf/2401.03129).

### Novelty

Its contribution is a diagnostic empirical study. It separates output behavior from benchmark knowledge. Repetition is measurable even when some task scores improve. It therefore cautions against using one accuracy aggregate as evidence of successful adaptation. [Paper, §4](https://arxiv.org/pdf/2401.03129).

### Results

The continued-pretraining corpus contains one billion tokens. With Chinese prompts, the rep-4 score rises from 0.103 to 0.552 after ordinary continuation. The evaluated LoRA variant yields 0.244. These values quantify repetition in this experiment, not a universal forgetting rate. [Paper, §2.1 and Table 2](https://arxiv.org/pdf/2401.03129).

### Limitations

The study centers on one model family and one language corpus. Learning rate changes materially affect adapter behavior. It does not establish the minimal replay needed to avoid the observed problems. Generalizing the result to all raw-text updates would exceed the evidence. [Paper, §4 and Limitations](https://arxiv.org/pdf/2401.03129).

### Relevance

Add output-language and repetition checks if the project uses multilingual documents. Track them separately from task correctness. Preserve the checkpoint's expected conversation template during evaluation. **Threat: low** to a replay method, but substantial to an evaluation that ignores changes in generated behavior.

## What is already covered, and what remains a project hypothesis

| Proposed component | Closest evidence | Assessment | Consequence |
|---|---|---|---|
| New data plus a little history | InsCL, SEEKR, SSR, OPR | Already done | Treat as baseline pipeline |
| Sweep replay budgets to preserve old tasks | SEEKR, OPR, Simple-CPT | Already done | A sweep needs a sharper setting or reproducible practical result |
| Use task similarity or representative samples | InsCL, Adapt-infinity, OASIS | Already done | Compare to random/stratified selection before adding complexity |
| Reduce retained history permanently | LITE-Adapt-infinity | Already done in an adjacent setting | Fixed storage alone is not an unqualified novelty claim |
| Preserve answer formats and instruction meaning | CITB, KPIG, aligned-CPT study | Already studied | Integrate separate correctness/format tests |
| Measure the smallest feasible replay under fixed stored tokens, full cost accounting, and controlled format shifts | Partial overlap across the above works | Worth testing; novelty unconfirmed | Establish a Pareto frontier and failure conditions before proposing an algorithm |

**Assessment:** this is an active, problem-driven area with competing methods and incompatible resource definitions. The mature part is replay as a retention mechanism. The unresolved project decision is which deployment constraint and evaluation setting will make the study useful. This classification is a synthesis of the reviewed works, not a bibliometric finding.

Evidence from at least seven reviewed methods/benchmarks—InsCL, SEEKR, OPR, SSR, CITB, TRACE, I-LoRA—supports replay as a useful retention baseline **in their evaluated settings**. There is no universal “1% is enough” result: InsCL grows memory with tasks, OPR fixes a final buffer relative to new-task size, and CPT papers count replayed tokens differently. Model size, initialization, task ordering, and supervision change the outcome. Those qualifications prevent an experiment on one benchmark from becoming a blanket recommendation.

The strongest practical impact would be a reproducible decision rule: *given a storage ceiling and tolerated regression, choose a replay policy and budget, or report that no tested configuration meets the target*. A positive result must improve both old-task retention and new-task usefulness at a defensible total cost. A negative result can also be valuable if it identifies when random replay is sufficient, when selection overhead dominates, or when apparent benefits disappear after fixing templates and token exposure. These are proposed deliverables, not claims that this review proves them achievable.

Citation counts were queried from OpenAlex on the review date for each selected DOI/version. The two highest observed version-specific velocities were O-LoRA (71 citations; approximately 25.13/year) and SSR (20; approximately 9.27/year). These are **not Semantic Scholar counts, not merged across versions, and not evidence of globally highest impact**; snapshot records are stored alongside the papers. Use **InsCL and SEEKR as methodological anchors**, with **OASIS and OPR as the recent novelty check**. [O-LoRA count source](https://api.openalex.org/works/https://doi.org/10.18653/v1/2023.findings-emnlp.715), [SSR count source](https://api.openalex.org/works/https://doi.org/10.18653/v1/2024.acl-long.77).

## Data-formatting decisions before training

### 1. Choose the learning target before the file format

| Arrival type | Suitable initial representation | Learning/evaluation implication |
|---|---|---|
| Labeled classification, QA, extraction | Instruction + input/context + target | Direct supervised tuning; preserve label choices and answer parser |
| Single/multi-turn chat | Ordered role/content messages | Preserve role boundaries and score the intended assistant turns |
| Raw documents | Document text and provenance; optionally generated instruction pairs | CPT and document-to-SFT are different experimental conditions |
| Corrections to previously valid facts | Versioned source and supersession relation | Retention tests must not reward an obsolete answer |

Bonito supports considering document-to-instruction conversion; aligned-CPT provides a reason to evaluate raw-text adaptation separately. Neither justifies manufacturing unverified labels. Keep the raw-source record and derived training example linked. The versioned-correction row is a proposed project control, because “preserve all historical answers” is not a coherent objective when ground truth changes. [Bonito](https://aclanthology.org/2024.findings-acl.748/), [Aligned-CPT](https://arxiv.org/abs/2401.03129).

### 2. Store a canonical record; render the model template later

The following is a **proposed project schema**, not a dataset format copied from a paper. JSONL is convenient for inspection; Parquet is also acceptable for storage. File container choice is not a research contribution. Keep experiment metadata out of the model-visible prompt unless it is intentionally part of the task.

```json
{
  "example_id": "sourceA-v2-00017",
  "source_id": "sourceA-document-42",
  "source_version": "v2",
  "arrival_batch": 3,
  "observed_at": "2026-09-01T00:00:00Z",
  "split": "train",
  "task_id": "sentiment",
  "domain": "product-reviews",
  "language": "en",
  "instruction": "Classify the review. Answer positive or negative.",
  "input": "The replacement arrived quickly and works well.",
  "target": "positive",
  "allowed_outputs": ["positive", "negative"],
  "template_version": "sentiment-v1",
  "group_id": "review-thread-42",
  "synthetic": false,
  "supersedes": null
}
```

For chat, replace the instruction/input/target payload with `messages`; retain the provenance envelope. Convert the canonical payload into the selected model's native chat template only in preprocessing. Pin tokenizer and template revisions, EOS handling, truncation policy, and maximum sequence length. Hugging Face's documentation distinguishes standard/prompt-completion/conversational representations and warns that chat control tokens are model-specific. [TRL SFT formats](https://huggingface.co/docs/trl/sft_trainer), [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating).

For the proposed SFT pilot, compute loss on target/assistant tokens, with padding and prompt tokens excluded, and verify the mask on a decoded sample. TRL's `assistant_only_loss` requires a compatible template; completion-only behavior depends on the dataset representation. Do not accidentally mix different loss masks between replayed and new examples. If using packing, verify example boundaries and attention behavior in the pinned implementation. [TRL loss masking and packing](https://huggingface.co/docs/trl/sft_trainer).

### 3. Make the arrival process and split policy explicit

Use disjoint training/validation/test groups, keeping a source document, conversation, or near-duplicate family within one group. If timestamps exist, construct actual chronological arrivals and never select replay using future data. If timestamps do not exist, call the task sequence **simulated incremental arrivals**. Freeze test sets before buffer selection and before any synthetic-question generation.

For a benchmark reproduction, retain the official split and task order first. For the project's deployment study, add an independent chronological or simulated stream with a documented rationale. There is **no identical split shared by five reviewed papers**: even SuperNI-based papers use different subsets and sampling counts. Do not conflate equal source-dataset names with equal evaluation protocols. [CITB, §4.4](https://aclanthology.org/2023.findings-emnlp.633.pdf), [SEEKR, §4.1](https://aclanthology.org/2024.emnlp-main.190.pdf), [SSR, §5.1](https://aclanthology.org/2024.acl-long.77.pdf), [TRACE, §4.1](https://arxiv.org/pdf/2310.06762).

### 4. Separate three meanings of budget

Let `H_t` denote previously arrived training data, `D_t` the new arrival, and `M_t` the retained historical store available before training stage `t`.

| Quantity | Definition for the proposed study | Why it must be logged |
|---|---|---|
| Retained storage | Serialized bytes and unique model-input/target tokens in `M_t` | Measures persistent historical data, not repeated use |
| Retained fraction | `unique_tokens(M_t) / unique_tokens(H_t)` | A constant fraction grows with history; denominator must be named |
| Training replay share | `old_token_exposures / (old_token_exposures + new_token_exposures)` | Repeated epochs/sampling can make a tiny store expensive |
| Old:new exposure ratio | `old_token_exposures / new_token_exposures` | This is `r/(1-r)` when replay share is `r`, not `r` |
| Auxiliary state | Embeddings, scores, cached logits, attention targets, adapters, optimizer state | Separates data storage from the complete method footprint |
| Selection access | Which historical records may be inspected when updating `M_t` | A small selected set is not necessarily a bounded-memory method |

Count both non-padding processed tokens and supervised target tokens. Long prompts still consume compute when their loss is masked, and long rationales increase target exposure. Report padded work or measured wall time as well; equal semantic token counts do not guarantee equal runtime. These are proposed accounting rules informed by the differing budgets in SEEKR, OPR, Adapt-infinity, OASIS, and the CPT studies.

For a strict-storage experiment, update memory from **only `M_t ∪ D_t`** after evaluation and selection; evicted examples must not silently remain accessible through an archive, embedding store, or generation-prompt pool. A separate archive-access experiment is legitimate if labeled. Preserve memory manifests with IDs, token counts, selection seed, selected stage, and selection score. Never put validation or test targets into that store.

### 5. Distinguish semantic errors from output-format errors

Use fixed answer parsers and log strict accuracy, permitted normalized accuracy, valid-format rate, and out-of-label-set rate. For JSON targets, check parseability and required schema independently of field correctness. For classification, distinguish the correct label surrounded by unwanted prose from an incorrect label. For generation, add task-native metrics and inspect a fixed sample of errors rather than treating ROUGE as a complete measure.

Run paired template controls: canonical instruction; a meaning-preserving paraphrase; and an intentional output-schema change where relevant. Keep examples, targets, checkpoints, and token budgets matched as closely as the intervention permits. If a template makes prompts longer, report the difference rather than calling that comparison token-matched. CITB and KPIG motivate these controls; any benefit observed here remains a result to establish experimentally. [CITB, Tables 3–4](https://aclanthology.org/2023.findings-emnlp.633.pdf), [KPIG, §3.4](https://aclanthology.org/2025.acl-long.1153.pdf).

## A feasible first study

This is a recommended starting protocol, not an implemented or approved experiment plan. GPU capacity and the application domain remain unspecified. Start with a model that fits the available hardware, one pinned training configuration, and a small benchmark subset. Scale to the complete benchmark and additional seeds after the data and accounting checks work.

1. **Reproduce the simple baselines.** Measure the starting checkpoint; new-data-only sequential tuning; random replay; task/domain-stratified replay; and one diversity selector. A reservoir sampler is a useful streaming baseline, but its standard example-uniform guarantee does not automatically apply to a variable-length token cap; document the chosen admission/eviction policy.
2. **Define the full-data reference precisely.** At stage `t`, cumulative restart means fine-tuning the same original base checkpoint on all `D_1...D_t`. Cumulative warm-start instead continues the preceding checkpoint on that union. Report these separately if both are used; neither is pretraining a language model from random initialization.
3. **Sweep storage before inventing a selector.** A pilot can retain token budgets corresponding to 0, 0.5, 1, 2, 5, and 10% of the final simulated historical pool, fixed in absolute tokens for the experiment. This is an offline budget-setting convention, not permission to inspect future content. Choose realistic fixed token ceilings instead when deployment storage is known.
4. **Control replay exposure independently.** Compare policies at equal total processed-token budgets, then measure achieved quality at equal wall-time budgets. Keep epochs, learning-rate schedules, quantization, LoRA rank, and optimizer reset policy explicit. Include selection, embedding, generation, training, and required evaluation time in total update cost.
5. **Add one strong competitor.** InsCL is the natural selection comparison; SEEKR tests added distillation; OASIS tests online selection under matched accessible memory. OPR is relevant if response regeneration is acceptable. Reproduction should precede an adapted variant, whose changed memory assumptions must be stated.
6. **Evaluate every checkpoint on old, new, and held-out capabilities.** Use at least three seeds/orders for the substantive result, including a difficult order; chronological experiments additionally need repeated sampling seeds without shuffling time. Add a second model/domain before generalizing. Report uncertainty for the actual experimental unit, not only variation across tasks.

For higher-is-better task scores `A[t,j]` after stage `t` on task `j`, report final macro average, current-task score, worst old-task regression, and backward transfer:

```text
BWT(T) = mean over j < T of (A[T,j] - A[j,j])
Forgetting(T) = mean over j < T of
                (max over k = j,...,T of A[k,j] - A[T,j])
```

This forgetting definition is nonnegative and distinct from BWT, which can be positive. For losses, reverse the comparison appropriately and publish the formula. Keep task-native metrics alongside any normalized aggregate. Do not average BLEU, classification accuracy, and F1 without a declared convention and per-task breakdown.

Predefine “close to full-data tuning,” for example an absolute tolerance on old-task macro score plus separate limits for worst-task regression and format validity. Those tolerances are **project choices**, not literature constants. Measure the smallest tested memory budget satisfying all constraints on validation data; use the held-out test set only for final assessment. If no budget meets them, report that outcome instead of weakening the target after seeing results.

## Impact, open conditions, and limits of the recommendation

**Operational value to demonstrate:** reduced GPU-hours per accepted update; reduced persistent historical bytes; preserved new-task quality; and bounded regressions of previously supported tasks/formats. Report both marginal update cost and total cost across the stream, including initial setup and selection infrastructure. A lower replay-example count with unchanged or increased GPU-hours is a data-saving result, not a compute-saving result. Whether this matters economically depends on the eventual arrival frequency, model, and data source.

**Condition that has changed:** recent work evaluates on-policy replay and online selection on newer instruction checkpoints, providing concrete modern comparators. Adjacent experiments now worth attempting include matching OPR's candidate access to a fixed store and evaluating OASIS with a finite historical pool. Neither experiment requires asserting a new algorithm. [OPR](https://arxiv.org/abs/2605.29495), [OASIS](https://aclanthology.org/2026.acl-long.158/).

**Condition that has changed:** data-conversion and pruning tools make more preparation choices explicit. Adjacent experiments include real versus generated replay under equal total cost, and canonical versus changing instruction formats with separate validity measurements. These are newly accessible combinations for this project, not claims that such experiments were historically impossible. [Bonito](https://aclanthology.org/2024.findings-acl.748/), [Adapt-infinity](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html), [KPIG](https://aclanthology.org/2025.acl-long.1153/).

No common primary metric or universal benchmark split is established across these 15 works. Current benchmark saturation is **Unknown — needs verification for the chosen model and pinned dataset version**; a normalized relative gain above 90 or 100 is not evidence that an accuracy benchmark is saturated. Code execution, resource estimates on the user's hardware, dataset licensing, and citation aggregation across versions were not validated in this review.

**Final prior-art verdict:** replay, representative selection, synthetic rehearsal, parameter-efficient continual tuning, budget sweeps, and formatting-aware evaluation are already covered. A useful project can start by reproducing random replay and one strong method, then measuring a rigorously defined storage–cost–retention frontier with format controls. The reviewed work does not establish that this combined study is publishably novel, and permanent pruning already has adjacent precedent. OASIS, OPR, InsCL, SEEKR, and LITE-Adapt-infinity are the comparisons that any stronger contribution claim must survive.

Next action: choose the arrival source, model/GPU budget, whether discarded history remains accessible, and the permitted performance regressions; then use the repository's challenger/domain/experiment workflow to turn this evidence into a small, falsifiable study.
