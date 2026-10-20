# Six data-selection papers to reproduce first

> **Superseded as the first-six recommendation.** The user prefers work from the last two years. Use the [revised recent-paper shortlist](/results/literature-review/small-model-selection/six-recent-selection-papers-2026-09-30.md) for project prioritization; retain this report as a foundational-baseline reference.

**Date:** 2026-09-30. **Primary output:** this report. **Input:** [Idea.md](/Idea.md) and the [15-work review](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md). **Decision:** prioritize six complementary selection mechanisms that can be checked on networks with millions of parameters: tiny-buffer experience replay, iCaRL herding, GSS, MIR, ASER, and EL2N/GraNd data pruning. All six are **Foundational (Older than 3 years)** relative to today; the older selection is intentional because the user's first constraint is a feasible small-model reproduction. InsCL, OASIS, and other recent LLM papers remain essential later comparisons, but their published large-model systems and extra assumptions make them less suitable as the first six implementations. This is a decision and reproduction report, not a claim that the six were originally tested on text instruction tuning.

**What “reproduce” means here:** first implement the published selector and confirm one reported direction on its native classification setup; then transplant the selector into one shared small-model text pipeline. The latter is a **new adaptation** whose results cannot be called a numerical reproduction of the original paper. Paper-specific details and numerical comparisons below come from the archived primary PDFs; author code was inspected for availability and structure, but no training has been run. The report is complete when each candidate has an algorithm, a genuine novelty statement, empirical anchor, practical obstacles, and a comparable project experiment.

## Decision matrix

| Order | Paper / year | Core question | Selection operation | Native small-model path | Relative effort* | Why it earns a slot |
|---|---|---|---|---|---|---|
| 1 | [Tiny Episodic Memories](https://arxiv.org/abs/1902.10486), 2019 | How much does ordinary replay buy? | **Write**: reservoir or class-balanced buffer; random read | Small classifier, Split CIFAR or MNIST | Low | Essential cost/retention reference |
| 2 | [iCaRL](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html), 2017 | Which examples represent each old class? | **Write**: feature-space herding | Small CNN, incremental CIFAR | Low–medium | Representative-coverage mechanism |
| 3 | [GSS](https://proceedings.neurips.cc/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html), 2019 | Which examples give distinct update directions? | **Write**: gradient-diverse memory | Two-layer MLP or small CNN, MNIST/CIFAR | Medium | Optimization-aware diversity |
| 4 | [MIR](https://arxiv.org/abs/1908.04742), 2019 | Which stored examples is the next update likely to harm? | **Read**: interference-based replay | MLP or small CNN, Split CIFAR | Medium | Selection conditioned on the impending update |
| 5 | [ASER](https://ojs.aaai.org/index.php/AAAI/article/view/17159), 2021 | Which memories protect old and new class boundaries? | **Read and write**: approximate KNN Shapley value | Small CNN, Split CIFAR | Medium–high | Boundary-aware valuation |
| 6 | [Deep Learning on a Data Diet](https://papers.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html), 2021 | Which new training examples can be omitted? | **Current-data pruning**: early EL2N/GraNd scores | ResNet18, CIFAR-10 | Medium | Extends cost savings beyond old-data replay |

*Effort is a **qualitative implementation estimate**, not a measured runtime. Original setups vary in task order, training epochs, head/evaluation rules, and memory accounting. Their published scores must never be put in one cross-paper ranking. “Write” changes what survives in persistent memory; “read” changes which survivors are rehearsed in a particular step. These are separate choices and can later be combined.

## 1. Tiny Episodic Memories — the indispensable replay reference

**Citation:** Arslan Chaudhry et al., *On Tiny Episodic Memories in Continual Learning*, 2019 arXiv / ICML workshop. [Primary PDF](https://arxiv.org/pdf/1902.10486). [Shared PyTorch implementation of related ER strategies](https://github.com/RaptorMai/online-continual-learning). This is foundational work.

### Idea

The learner sees a task stream and keeps a small memory of previous examples. Each update trains on the new batch together with samples drawn from that memory. The paper tests reservoir, ring-buffer, feature-mean, and clustering strategies for filling the memory. Its setting is supervised classification on MNIST and split vision benchmarks, not generated language. [Primary PDF, §§3–5](https://arxiv.org/pdf/1902.10486).

### Novelty

The paper's principal contribution is an empirical demonstration that direct rehearsal remains strong even with very small memories. Reservoir sampling itself predates the paper, so copying reservoir sampling is not an algorithmic novelty claim. Its practical additional idea is a hybrid that begins with reservoir sampling and switches toward class balancing when earlier classes risk vanishing from memory. This gives us an honest reference against which any more elaborate selector must justify both accuracy and extra work. [Primary PDF, §§4–5](https://arxiv.org/pdf/1902.10486).

### Results

The authors report gains of 7–17 percentage points for some ER variants when memory holds one example per class; those gains depend on benchmark and comparator. In their Split CIFAR experiment at one example per class, reservoir ER trails ER K-Means by 3.5 points. Their tiny-memory table reports forgetting of 0.13 for ER-Ringbuffer versus 0.27 for fine-tuning on Split CIFAR. These are native image-classification results, not predicted text-task outcomes. [Primary PDF, Fig. 1 and Tables 1–2](https://arxiv.org/pdf/1902.10486).

### Limitations

With a tiny fixed reservoir, older classes can disappear as later examples arrive. Class balancing assumes labels or task categories that may be absent or ambiguous in a real incoming-data stream. The reported wall times are tied to the original small models and hardware. Selection is simple enough that later methods must beat it under matched budgets, rather than citing their own paper's larger-model number. [Primary PDF, §5](https://arxiv.org/pdf/1902.10486).

### Relevance

For incremental fine-tuning, start with new-data-only, then ER-Reservoir and ER-Class-Balanced as two reference curves. Store exactly K serialized training records, and use the same replay retrieval policy for this first comparison. Record both persistent bytes and repeated example/token exposures because a tiny memory can still consume many optimizer steps. **Threat to the proposed novelty: high**; updating with a small historical sample has long been studied.

**Minimal algorithm:** On the n-th observed training example, if fewer than K records are stored, append it; otherwise draw an integer uniformly from 1…n and replace that memory slot only when the draw is at most K. Sample old examples uniformly from the current memory for each training step. For the hybrid, switch to class quotas only after a predeclared minimum coverage rule is breached; document this as the paper's hybrid variant and check how its dynamic quotas handle newly seen classes.

**First check:** Reproduce a Split CIFAR-10 or MNIST qualitative result with an MLP/small CNN and one example per class, then sweep 1, 5, and 20 per class plus an equal-size reservoir. Log the retained class histogram after every arrival. No author environment is required for this baseline; the shared [PyTorch repository](https://github.com/RaptorMai/online-continual-learning) has ER, random retrieval, and reservoir components, but its settings still need matching to the paper.

## 2. iCaRL — representative examples in feature space

**Citation:** Sylvestre-Alvise Rebuffi et al., *iCaRL: Incremental Classifier and Representation Learning*, CVPR 2017. [Primary PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf). [Author code](https://github.com/srebuffi/iCaRL). This is foundational work.

### Idea

iCaRL learns a representation while adding new classes and retaining a fixed number of old-class exemplars. It selects each class's exemplars greedily so that their mean feature vector approximates the class's full-data mean. When new classes arrive, the per-class quota shrinks and the ordered exemplar lists can be truncated. The full method also uses old-output distillation and a nearest-exemplar-mean classifier. [Primary PDF, Algorithms 1–5](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf).

### Novelty

Herding as a general selection idea precedes iCaRL. The paper integrates ordered exemplar selection with incremental representation learning and classification across newly arriving classes. For this project, the reusable data-selection component is the exemplar construction rule; distillation and the classifier are separate method components. A herding-only comparison must therefore be labeled an ablation/adaptation, not a reproduction of all iCaRL claims. [Primary PDF, §§2.3–2.4](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf).

### Results

On iCIFAR-100, the paper's Table 1 gives 64.1 average incremental accuracy for iCaRL with 10 classes per increment. Some component ablations in that row report 59.3 and 59.9, but they change more than exemplar selection alone. Table 1 also shows that replacing the exemplar-mean classifier with a full-class-mean classifier yields 64.5, a small gain in that setting. None of these figures gives an isolated expected gain for herding on text. [Primary PDF, Table 1](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf).

### Limitations

The rule needs a stable label or grouping key and a feature extractor evaluated over the new class's candidate set. Its use of all new-class examples at the current boundary is legitimate for batch arrivals but differs from a strict one-pass stream. Feature drift after later updates can make old selected exemplars less representative. The [author repository](https://github.com/srebuffi/iCaRL) explicitly says its TensorFlow/Theano code is outdated; the shared [PyTorch implementation](https://github.com/RaptorMai/online-continual-learning) includes an iCaRL agent but its public example uses a random buffer update, so that command alone does **not** verify herding.

### Relevance

For incremental fine-tuning, herding tests whether coverage of the old data distribution matters more than selecting hard cases. Use a fixed representation checkpoint at each arrival to select exemplars, then store the raw examples and recompute features only when the protocol permits. Compare herding-only to random memory under one common replay learner before testing full iCaRL. **Threat to a “representative examples” claim: high**, because selecting class prototypes is established.

**Minimal algorithm:** For each class or task group c, compute the current candidate mean feature vector μc. Choose exemplar k that minimizes the distance between μc and the average feature vector of the already chosen exemplars plus that candidate. Repeat until the group's quota is filled, preserving the chosen order so later quota reductions truncate from the end; enforce unique chosen records in the implementation. Unlike a simple nearest-centroid top-K, the greedy rule accounts for exemplars already chosen. [Primary PDF, Algorithm 4](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf).

**First check:** On small Split CIFAR-10, implement the exact ordered herding selector and compare against random and closest-to-mean top-K at equal K; separately run the full iCaRL training/classification recipe if validating its published score is important. Log feature checkpoint, group quota, selection time, and whether all current-group examples were scanned. On text, use labels or task IDs only when they are truly available at selection time.

## 3. GSS — gradient-direction diversity

**Citation:** Rahaf Aljundi et al., *Gradient based sample selection for online continual learning*, NeurIPS 2019. [Primary PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf). [Author code](https://github.com/rahafaljundi/Gradient-based-Sample-Selection). This is foundational work.

### Idea

GSS views replay memory as a reduced set of gradient constraints from old examples. A useful memory spans distinct gradient directions rather than filling up with similar examples. The paper derives a diversity surrogate and offers an integer quadratic program plus a cheaper online greedy rule. The proposed first reproduction is **GSS-Greedy**, which changes what is stored and needs no task boundary in its original online setting. [Primary PDF, §§3.3–3.4](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf).

### Novelty

The novel step is treating memory selection as a constraint-reduction problem in parameter-gradient space. Diversity of image pixels or hidden features is an existing comparison; GSS asks whether gradients provide a more relevant geometry. Its greedy score estimates a new example's maximum cosine similarity to a small memory sample and preferentially evicts redundant stored examples. GSS-IQP is the more exact optimization but is costly enough that it is a poor first implementation target. [Primary PDF, §3.4 and Tables 1–3](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf).

### Results

For a disjoint-MNIST buffer of 300 examples, Table 1 reports 82.6 ± 2.9 average test accuracy for GSS-Greedy versus 37.5 ± 1.3 for that table's random policy. The large gap partly reflects the paper's severe online/task-agnostic setting and the random buffer's poor retention of early tasks; it must not be treated as a general expected gain over a balanced reservoir. On five deliberately imbalanced MNIST sequences, Table 4 reports mean 77.96 for GSS-Greedy versus 69.12 for reservoir. The paper also reports that GSS-Greedy can outperform more expensive GSS variants in its tested settings. [Primary PDF, Tables 1 and 4](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf).

### Limitations

Per-example gradient computation is an additional backward-work cost, even when the buffer is small. The score depends on current parameters, so stored scores can become stale as the model changes. In a text model, full-gradient vectors may be large; a last-layer gradient substitute would be an adaptation, not an exact paper reproduction. The original work is classification on MNIST/CIFAR and says nothing directly about language output format. [Primary PDF, §§3.4–4](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf).

### Relevance

GSS tests whether the project's representative historical data should cover update directions rather than input semantics. Start on the native small MLP where full gradients are feasible, then test a head-only or projected-gradient approximation as a separate text-port ablation. Measure retention per selection second, not just accuracy per retained example. **Threat to novelty: high** for any generic proposal to store gradient-diverse samples.

**Minimal algorithm:** For a new example, compute its parameter-loss gradient and a small sample of old memory gradients. Shift the maximum cosine similarity into a nonnegative score; less similar new examples are more valuable. When memory is full, sample an eviction candidate biased toward high redundancy and apply the paper's score-dependent replacement probability rather than simply replacing the closest point. The [shared PyTorch repository](https://github.com/RaptorMai/online-continual-learning) exposes GSS-Greedy as a buffer update, which is a useful implementation reference but has not been executed here.

**First check:** Replicate the disjoint-MNIST direction on a 2-layer MLP, then compare GSS-Greedy with reservoir and class-balanced replay on the same Split CIFAR stream. Pin gradient layer scope, memory-gradient sample count, when scores are refreshed, and whether a candidate is scored before or after the learner update. Use identical training batches/optimizer for every buffer-writing policy.

## 4. MIR — retrieve the memories endangered by the next step

**Citation:** Rahaf Aljundi et al., *Online Continual Learning with Maximally Interfered Retrieval*, NeurIPS 2019. [Primary PDF](https://arxiv.org/pdf/1908.04742). [Author code](https://github.com/optimass/Maximally_Interfered_Retrieval). This is foundational work.

### Idea

MIR keeps a historical memory and chooses which memories to replay **at the current optimizer step**. It first estimates the parameter update that would follow from the new batch alone. For candidate old examples, it compares loss after that virtual update with loss before it and selects those with the largest increase. The ER-MIR variant uses reservoir for persistent storage, so its published retrieval gain does not imply a better stored subset. [Primary PDF, §3.1](https://arxiv.org/pdf/1908.04742).

### Novelty

The distinctive selection signal is predicted interference with the imminent update. Ordinary random replay ignores whether a particular old example is currently at risk. The paper also studies generative and hybrid replay, but those are separate systems and unnecessary for the first comparison. To isolate MIR, hold the written memory and learner fixed and change only the retrieval score. [Primary PDF, §§3–4](https://arxiv.org/pdf/1908.04742).

### Results

On its Split CIFAR-10 setup with 50 memories per class, Table 2 reports final accuracy 40.0 ± 1.1 for ER-MIR versus 33.1 ± 1.7 for ER. The same row reports forgetting 30.2 ± 2.3 for ER-MIR versus 35.4 ± 2.0 for ER. At 100 memories per class, the final accuracies are 47.6 versus 41.3. Those values are for the paper's training schedule and cannot be compared numerically to ASER's separate CIFAR-10 table. [Primary PDF, Table 2](https://arxiv.org/pdf/1908.04742).

### Limitations

MIR evaluates candidate losses twice and forms a virtual update; that selection bill grows with candidate-pool size. Current high loss alone is not the same as increased loss due to the next update, so shortcutting to hard-example mining changes the method. The paper includes optional historical best-loss scoring, which adds state and should be kept separate from the base score. It does not solve which old examples survive a strict fixed-storage eviction policy. [Primary PDF, §3.1](https://arxiv.org/pdf/1908.04742).

### Relevance

MIR asks whether dynamic retrieval can protect exactly the older behaviors threatened by each new arrival. Pair it with the same reservoir used in the ER reference so only retrieval changes. Record candidate-pool size and extra forward/backward time because a quality gain may not reduce total update cost. **Threat to novelty: high** for the general idea of replaying samples most likely to be forgotten.

**Minimal algorithm:** Starting from parameters θ, compute a one-step virtual θ′ using only the new batch, without mutating the actual model/optimizer state. For each of C randomly drawn stored candidates, score loss(θ′, x, y) − loss(θ, x, y); take the top R to replay. Restore exactly θ and optimizer state, then perform the real update on the new batch plus R memories. The temporary scores use only previously seen training examples, never validation/test examples. [Primary PDF, Algorithm 1](https://arxiv.org/pdf/1908.04742).

**First check:** Use the author ER script or the shared [PyTorch selector](https://github.com/RaptorMai/online-continual-learning) on Split CIFAR-10. Hold reservoir K fixed, compare uniform retrieval to MIR at R fixed, and sweep C/R to expose cost. The author repository pins old Python/PyTorch versions, so a modern port is likely easier than installing its historical stack unchanged. [Author README](https://github.com/optimass/Maximally_Interfered_Retrieval).

## 5. ASER — value memories by old/new decision boundaries

**Citation:** Dongsub Shim et al., *Online Class-Incremental Continual Learning with Adversarial Shapley Value*, AAAI 2021. DOI: 10.1609/aaai.v35i11.17159. [Primary PDF](https://ojs.aaai.org/index.php/AAAI/article/download/17159/16966). [Original TensorFlow code](https://github.com/RaptorMai/ASER) and [shared PyTorch port](https://github.com/RaptorMai/online-continual-learning). This is foundational work.

### Idea

ASER scores candidates using approximate K-nearest-neighbor Shapley values in the model's current feature space. A stored example can support an older class boundary and interfere with a new-class boundary. The method uses that cooperative/adversarial relationship to decide which memories to replay. It also applies a related valuation to memory updates, so both reading and writing can change. [Primary PDF, §4](https://ojs.aaai.org/index.php/AAAI/article/download/17159/16966).

### Novelty

The contribution is a class-boundary-aware value score built from KNN-Shapley approximations, not a new optimizer. The score compares a candidate's effect on an old balanced subset and on the arriving batch. The paper defines a hard max/min version and a softer mean-based ASER-μ variant. This differs from MIR's loss change after a virtual gradient step and iCaRL's centroid coverage. [Primary PDF, Eqs. 5–6 and Algorithm 2](https://ojs.aaai.org/index.php/AAAI/article/download/17159/16966).

### Results

Table 1 reports 14.0 ± 0.4 final average accuracy for ASER-μ on Split CIFAR-100 with M = 1,000, compared with 11.2 ± 0.4 for ER. On Split CIFAR-10 with M = 500, ASER-μ reaches 36.3 ± 1.2 while ER reaches 32.2 ± 1.4. The paper averages 15 runs, and its setup is much harder than a standard offline supervised classifier. These are different datasets and cannot be collapsed into one universal advantage. [Primary PDF, Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/17159/16966).

### Limitations

The algorithm uses class labels, feature-space distances, KNN statistics, and a balanced old evaluation subset. It is the most implementation-heavy of the six, and its valuation can be unstable with rare classes or drifting embeddings. The [author README](https://github.com/RaptorMai/online-continual-learning) says its PyTorch ASER port is more efficient and gives better performance than its original TensorFlow code, so the two should not be treated as identical published-number reproductions. The paper also discusses an implementation mismatch in MIR comparisons caused by filtering same-task memory, making explicit task-ID use important. [Primary PDF, Appendix F](https://arxiv.org/pdf/2009.00093).

### Relevance

ASER offers a meaningful contrast to both distribution coverage and one-step interference: it directly emphasizes fragile decision boundaries. Reproduce it on the labeled native vision benchmark before reusing it for labeled text tasks. In the project comparison, test read-only ASER and combined read/write ASER separately against ER. **Threat to novelty: high** for claiming that decision-boundary-sensitive replay valuation is unexplored.

**Minimal algorithm:** Embed a balanced old evaluation subset, a candidate pool from memory, and the new batch with the same current feature extractor. Compute each candidate's approximate KNN-Shapley contribution to old and new evaluation sets; for ASER-μ, subtract mean new-set contribution from mean old-set contribution. Replay the highest-valued candidates; for the full method, apply the paper's KNN-Shapley memory update as well. With heterogeneous text tasks, label namespaces must be explicit or same-named labels from different tasks can create false neighbors. [Primary PDF, §4 and Algorithm 2](https://ojs.aaai.org/index.php/AAAI/article/download/17159/16966).

**First check:** Run the official TensorFlow configuration on a small Split CIFAR setting only if its archived environment works; otherwise port the exact scoring equations and compare component decisions against the public PyTorch implementation. Begin with ASER-μ because its mean-based score is easier to audit, then add ASER's max/min variant. Record KNN k, balanced subset size, candidate count, embedding checkpoint, feature distances, and both retrieval and update time.

## 6. Data Diet — score new examples early, then prune

**Citation:** Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite, *Deep Learning on a Data Diet: Finding Important Examples Early in Training*, NeurIPS 2021. [Primary PDF](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf). [Author JAX/Flax code](https://github.com/mansheej/data_diet). This is foundational work and **not originally a continual-learning replay paper**.

### Idea

The paper estimates example importance from a checkpoint early in training. GraNd uses the norm of each example's parameter-loss gradient, averaged over multiple random initializations. EL2N uses the L2 norm of predicted class probabilities minus the one-hot label vector and is cheaper to calculate. The authors use these scores to rank and prune current training examples before longer training. [Primary PDF, §§2–3](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf).

### Novelty

The paper shows that useful pruning signals can appear near the beginning of training rather than requiring completed runs or late forgetting statistics. Its specific algorithmic contribution is the GraNd score and its low-cost EL2N proxy. High-scoring samples are not always best: the authors analyze extreme scores and use a sliding score interval to avoid noisy outliers. Applying EL2N to sequential replay is a project adaptation and requires a new retention experiment. [Primary PDF, §§2–4](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf).

### Results

The paper reports that EL2N can remove about 50% of CIFAR-10 training data while slightly improving the reported test result in its evaluated setup. Its text also reports removing 25% of CIFAR-100 with little or no loss in the cited setting. Results average four independent runs in the paper's principal pruning experiments. These are static-data, warm-up-plus-retraining findings, not evidence that EL2N retains old tasks after later updates. [Primary PDF, abstract and Fig. 1](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf).

### Limitations

The author code recommends at least 10 scoring runs, which can erase some savings if scoring costs are charged to one small update. Very high scores can signal label noise or unusual examples; naively keeping only the hardest cases loses coverage. EL2N assumes comparable class-probability vectors, so a generated-text version would need a different definition and could incur much higher token-level cost. The paper's original pruning run scans the full current dataset; it does not prove a bounded historical store. [Primary PDF, §§3–4](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf), [author README](https://github.com/mansheej/data_diet).

### Relevance

Data Diet adds a different optimization axis to the user's project: process fewer **new** examples as well as fewer old examples. Begin by pruning only the currently arriving batch, which is available under the project premise, and keep historical-memory policy fixed. Compare one scoring run with multiple scoring runs and report amortized costs across future updates. **Threat to a generic “select informative data early” novelty claim: high**; continual retention under a fixed historical store remains to be tested.

**Minimal algorithm:** At a pinned early checkpoint, compute p(y|x) over each candidate and assign EL2N(x,y) = norm₂(p − onehot(y)); average over the chosen scoring runs if matching the full recipe. Retain either top-scoring examples or a predeclared score-percentile window, then train from an explicitly defined checkpoint on the selected subset. Include scoring, warm-up, and any restart in the cost total. GraNd can be added as an ablation but requires per-example gradients. [Primary PDF, §§2–3](https://papers.nips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf).

**First check:** On native CIFAR-10, confirm that high EL2N scores and a sensible score window retain performance better than a matched random subset at one early checkpoint. The [author code](https://github.com/mansheej/data_diet) documents full-data training, score extraction, and subset training, but it uses JAX/Flax and recommends multiple runs. For text adaptation, compute scores only from the new training batch or the already retained historical buffer; accessing evicted history would change the storage setting.

## Shared small-model experiment after native checks

**Native sanity checks first.** ER, iCaRL, GSS, MIR, and ASER can start on Split CIFAR-10 with a small CNN/ResNet family; Data Diet starts on static CIFAR-10. Keep each paper's original data order, optimizer, batch size, evaluation rule, and memory size for one sanity check. These checks are about implementation fidelity; a smaller learner or fewer training epochs may reproduce **direction**, not the paper's exact score. The public [online-continual-learning framework](https://github.com/RaptorMai/online-continual-learning) exposes ER, MIR, GSS and ASER under a common vision harness and also provides an iCaRL agent, but paper-specific features must be checked against their primary algorithms.

**Common text port.** A defensible inexpensive stage is [CLINC150 data_full](https://archive-beta.ics.uci.edu/dataset/570/clinc150) with ten sequential domain arrivals: 150 intent labels, 100 train / 20 validation / 30 test examples per intent. Use a pinned approximately 4.4M-parameter [BERT-tiny checkpoint](https://huggingface.co/prajjwal1/bert-tiny) with one global classifier head, and report the actual parameter count after loading. This is a **proposed experiment**, not one of the six papers' original results. A global label space avoids task-ID-at-test leakage; the unseen-label training policy and evaluation on only seen intents at each stage must be specified. Keep CLINC's official split, exclude the out-of-scope category for the first run, and add it as a later stress test. See the dataset's [official files](https://github.com/clinc/oos-eval/blob/master/data/data_full.json) and [UCI metadata](https://archive-beta.ics.uci.edu/dataset/570/clinc150).

At each arrival, record only a versioned training record with example ID, source/split, arrival index, group/label, raw text, target label, token count, and optional template version. Render prompts from that record if a later instruction model is used; keep tokenizer/template revision and loss mask pinned. For BERT classification, output-format compliance is inapplicable because the model predicts a class ID; the optional [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) extension can test generated label strings and strict formatting separately. Use fixed intent-name strings and an exact-format validator in that extension, so valid format and correct intent are reported separately. The six selectors' image-to-text transfer, and any results from it, must be reported as new evidence.

**Matched controls and budget.** Keep the same checkpoint initialization, sequence, examples, optimizer updates, replay batch size, and memory-size ceiling across write-policy comparisons. For read-policy comparisons, hold the writer fixed to reservoir; then factorially test promising reader–writer pairs. Sweep K as **both examples and serialized/token bytes** because text lengths differ, and fix total processed tokens or provide an equal-update comparison alongside an equal-wall-time comparison. Record selection forward/backward passes, candidate scans, buffer bytes, auxiliary embeddings/scores, peak memory, and total GPU/CPU time. A 1,000-example memory selected from all previously evicted examples is a different access regime from a strict buffer that updates only from current arrivals and its existing contents.

**Outputs and acceptance.** Report the full stage-by-intent accuracy matrix, final macro old-intent accuracy, new-intent accuracy, worst old-intent regression, backward transfer, nonnegative forgetting, memory retention by age/class, and elapsed update time. Use the prior [metric definitions](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md) consistently. Predefine a retention tolerance against cumulative-data fine-tuning and a cost target before looking at final test results. Run multiple stream seeds and at least two domain orders for the substantive comparison; reuse one pilot order only for debugging. A selector is interesting here only if it improves the **retention–storage–total-cost frontier**, not solely accuracy at equal K.

## Why these six, and the next novelty boundary

ER supplies the indispensable simple baseline; iCaRL tests feature coverage; GSS tests gradient-direction coverage; MIR tests at-risk retrieval; ASER tests old/new decision-boundary value; EL2N tests whether training the new arrival itself can be reduced. That spans **storage selection, retrieval selection, and new-data pruning** with reusable code and small native models. It also avoids treating methods that change losses or model architecture, such as SEEKR and O-LoRA, as pure selection comparisons. Native paper results remain visual-classification evidence; the text experiment is the project's contribution-testing environment.

Recent direct competitors from the earlier review still constrain claims: [InsCL](https://aclanthology.org/2024.naacl-long.37/) already does instruction-aware historical allocation and sampling but uses GPT-4 intention tagging and grows its replay count with tasks; [OASIS](https://aclanthology.org/2026.acl-long.158/) uses forward-efficient online selection but explicitly assumes an infinite historical memory in its experiments; [OPR](https://arxiv.org/abs/2605.29495) selects a tiny replay set after generating from a larger prompt pool. [GCR](https://openaccess.thecvf.com/content/CVPR2022/html/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.html) is a useful gradient-approximation follow-up, but a [public repository issue](https://github.com/google-research/google-research/issues/3412) reports a missing dependency in the released code, adding reproduction risk. None of the six selector mechanisms can be claimed as a new algorithm by reimplementation.

**Practical novelty test:** after reproducing the six, measure which selection *axis* matters under a strict historical token store and total update-cost cap on text arrivals. A combination such as feature coverage plus MIR retrieval is already an obvious composition; it would need a distinct mechanism or a convincing, previously unmeasured constraint/result to count as an algorithmic advance. If the chosen model later generates answers, test semantic correctness and format validity separately and compare against InsCL/OASIS on the same attainable setting. Decide future optimization only after the six native sanity checks and the first matched small-text pilot show where random replay actually fails.

## Evidence and verification status

The five older replay papers and the Data Diet paper were read from official/arXiv PDFs saved as [source files](/results/literature-review/small-model-selection/sources/), and the public code READMEs/selected implementation files were inspected. OASIS/InsCL/OPR details come from the [earlier full-text review](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md). No repository was installed, no dataset/model downloaded, and no experiment was run in preparing this decision report. Qualitative effort rankings may change once the actual hardware and software environment are known.
