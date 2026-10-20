---

## [2026-09-28] Literature Review — Cost-Efficient Incremental Fine-Tuning

**Stage**: literature-review
**Lifecycle position**: Active
**Field orientation**: Problem-first

**Files**:
- Report: [literature-review/literature-review-incremental-fine-tuning-2026-09-28.md](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md)
- Ranked candidates: [literature-review/candidates-ranked.json](/results/literature-review/candidates-ranked.json)
- BibTeX: [literature-review/references.bib](/results/literature-review/references.bib)
- Search/evidence audit: [literature-review/search-log.md](/results/literature-review/search-log.md)

**Coverage**: 15 works, each with Idea/Novelty/Results/Limitations/Relevance paragraphs, based on targeted full-text reading. Includes 2026 OPR and OASIS. Numerical findings are reported results; no training or reproduction was performed. Text instruction/SFT is the explicit working interpretation, with raw-text and multimodal papers as adjacent evidence.

**Key gaps and current claim**:
- New arrivals plus selected historical replay is established prior art; the broad idea is not a novel algorithm.
- A useful project question is the storage–total-update-cost–retention tradeoff under explicitly bounded access to history, with independent output-format checks. This is a candidate empirical contribution, not a verified novelty claim.
- Distinguish persistent historical storage, repeated replay exposure, auxiliary state, and historical candidate access. OPR's small selected buffer uses a larger historical prompt pool; OASIS describes infinite memory; LITE-Adapt-infinity already studies permanent pool pruning.
- Canonical records should preserve provenance and payload separately; pin templates/tokenizers and loss masks, prevent split leakage, and score semantic correctness separately from valid formatting.

**Anchor papers**: InsCL and SEEKR for the closest established methods; OPR and OASIS for recent direct competition; CITB/TRACE for protocols; KPIG/Bonito for instruction constraints and conversion. O-LoRA and SSR have the highest recorded citation velocities among the matched OpenAlex version records, but these fragmented counts do not establish field-wide influence.

**Condition shifts and adjacent possibilities**: Newer instruction checkpoints and online/on-policy selection methods enable matched-access comparisons; available data-generation and pruning tools enable equal-cost real-versus-synthetic replay and paired template-control experiments. These combinations are feasible project directions, not proof of being unprecedented.

**Next action**: Fix the arrival source, model/GPU budget, historical-access policy, and allowed quality regressions. Then run the challenger/domain/experiment stages around a pilot comparing new-only, random/stratified replay, one stronger method, and a precisely defined cumulative-data reference.

---

## [2026-09-30] Literature Review — Six Small-Model Data-Selection Reproductions

**Stage**: literature-review
**Lifecycle position**: Active
**Field orientation**: Problem-first

**Files**:
- Report: [literature-review/small-model-selection/six-selection-papers-2026-09-30.md](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md)
- Six-paper registry: [literature-review/small-model-selection/candidates-six.json](/results/literature-review/small-model-selection/candidates-six.json)
- BibTeX: [literature-review/small-model-selection/references-six.bib](/results/literature-review/small-model-selection/references-six.bib)

**Coverage**: Six foundational papers were selected for distinct data-selection mechanisms that have small-model classification starting points: tiny-memory ER, iCaRL herding, GSS gradient diversity, MIR interference-aware retrieval, ASER KNN-Shapley valuation, and Data Diet EL2N/GraNd current-data pruning. Each has a five-paragraph source-grounded analysis, algorithm card, empirical anchor, and first reproduction check. Native vision/classification checks and a proposed 4.4M-parameter BERT-tiny text adaptation are explicitly separate evidence stages; no experiment was run.

**Key gaps and current claim**:
- Plain historical replay, representative exemplars, gradient-diverse storage, interference-aware retrieval, Shapley valuation, and early difficulty pruning are established; implementing any one alone is not a novelty claim.
- Small-model comparisons should isolate buffer writing, buffer reading, and new-data pruning under a shared memory and total-cost budget.
- Bounded historical access, serialized/token storage, auxiliary state, repeated replay exposure, task-label availability, and output-format validity need explicit accounting.

**Anchor papers**: ER as the essential baseline; GSS, MIR, and ASER as stronger selectors. iCaRL and Data Diet test complementary coverage and current-data pruning. InsCL, OASIS, and OPR remain recent direct comparators from the preceding 15-paper review, but their full published systems are deferred from the first six small-model reproductions.

**Next action**: Implement native algorithm sanity checks in the report's order, then run a matched CLINC150/BERT-tiny pilot with fixed writing and reading controls; optimize only after the retention–storage–total-cost frontier is measured.

---

## [2026-09-30] Literature Review Revision — Fifteen Recent Selection Papers

**Stage**: literature-review
**Lifecycle position**: Active
**Field orientation**: Problem-first

**Files**:
- Revised report: [literature-review/small-model-selection/six-recent-selection-papers-2026-09-30.md](/results/literature-review/small-model-selection/six-recent-selection-papers-2026-09-30.md)
- Candidate registry: [literature-review/small-model-selection/candidates-fifteen-recent.json](/results/literature-review/small-model-selection/candidates-fifteen-recent.json)
- BibTeX for first six: [literature-review/small-model-selection/references-six-recent.bib](/results/literature-review/small-model-selection/references-six-recent.bib)

**Decision**: The user prefers work from the last two years and allows up to 1B parameters. The report now covers 15 works from 2024–2026, with CSReL, aL-SAR, RDS+, LEAD, OASIS, and TSR as the six first reproductions. Added Adapt-∞, On-Policy Replay, KPIG, SEEKR, D3, RICo, Self-Adaptive CIT, Mix-CPT, and Coresets are more than replay as comparators and novelty boundaries. Use Qwen2.5-0.5B-Instruct as the common text learner.

**Evidence boundary**: CSReL and aL-SAR have native small-vision-model paths, while TSR reports a 0.5B text experiment. Other LLM headline results generally use 7–14B models, so 0.5B comparisons are algorithmic adaptations. Mix-CPT first appeared in July 2024 but was published at ICLR 2025. OASIS's paper-linked code repository was empty on 2026-09-30. No implementation was installed or run.

**Next action**: Reproduce the published TSR 0.5B TRACE-8 setting and the native small-vision CSReL/SAR ablations. Port the six selectors to one fixed Qwen2.5-0.5B-Instruct stream. Compare semantic correctness, format validity, retained bytes, and total selection-plus-training cost before proposing optimization.

---

## [2026-10-02] Literature Review Refinement — Five Linked Complete Reproductions

**Stage:** literature-review
**Lifecycle position:** Active
**Field orientation:** Problem-first

**Files:**
- Focused report: [literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md)
- Five-paper registry: [literature-review/small-model-selection/candidates-five-linked-2026-10-02.json](/results/literature-review/small-model-selection/candidates-five-linked-2026-10-02.json)
- Verification/search log: [literature-review/small-model-selection/search-log-five-linked-2026-10-02.md](/results/literature-review/small-model-selection/search-log-five-linked-2026-10-02.md)

**Decision:** The user's objective is five actual complete-method reproductions linked by a story. Select RDS+, S2L, LEAD, TSR, and OPR: target similarity, proxy learning trajectories, adaptive training utility, compatible-source replay, and refreshed replay responses. This replaces the prior first-six reproduction order while preserving the broader 15-paper report.

**Evidence boundary:** S2L publishes a Pythia-70M proxy/Pythia-410M learner setting; TSR publishes Qwen2.5-0.5B-Instruct. RDS+, LEAD, and OPR require scaled full-algorithm reproductions for the <=1B budget. LEAD's official repository remains marked under construction. Official repositories were inspected but not installed or run; no reproduction result exists yet.

**Current claim:** The course contribution is initially a coherent reproduction study. A candidate empirical extension asks whether one-shot data-selection winners remain good bounded replay memories under format changes, measuring semantic correctness, format validity, retention, historical access, auxiliary storage, and total cost. Each selector/replay mechanism already exists; no novel algorithm or priority claim is established.

**Next action:** Run native TSR 0.5B and S2L 70M/410M checks, bring up all five complete pipelines, compare the three selectors and two replay methods in matched settings, then run the explicitly labeled bounded-access extension. Implement optimizations only after reproductions reveal a concrete failure.

---

## [2026-10-20] Implementation Design — Incremental Fine-Tuning

**Stage:** implementation-design
**Blueprint:** [design/README.md](/design/README.md), indexing 14 component documents.
**Inputs:** current [Idea.md](/Idea.md), [code-structure.md](/code-structure.md), project workflow/rules, and the four local paper PDFs with recorded checksums and primary-source links.

**Current scope:** The current idea selects CSReL, aL-SAR, OASIS, and TSR, superseding the older implementation shortlists above. The blueprint defines dataset/model/pipeline contracts, raw data and weight storage, shared records and measurements, tests, combined orchestration, and the entry point. Source code belongs in `src/`; the requested design documents belong in `design/`.

**Design decisions:** Keep standalone paper implementations distinct from the task-chunk hybrid. Enforce causal training access, one learner optimizer owner, explicit memory/auxiliary budgets, coherent data/teacher routing, complete checkpoint recovery, and total-cost accounting. Document OASIS source ambiguities rather than hiding implementation choices. Model identities, datasets, numeric budgets, and experiment acceptance margins remain unspecified.

**Open boundaries:** Full SAR needs valid class/group semantics for language data; native TSR requires its fixed-initialization LoRA-B signature; CSReL task-end selection requires current-task candidate access. OASIS equation/pseudocode choices need comparison against a pinned official implementation before reproduction claims.

**Validation:** Checked all 14 documents for index coverage, 28 local links, four source PDF checksums, and whitespace integrity. No implementation, model execution, or training was performed.

**Next action:** Use the blueprint when implementing components; supply model/dataset choices and a fixed experiment protocol before scientific runs. Validate paper algorithms independently before composing the hybrid.

---

## [2026-10-20] Design Refinement — Resume, SLM Limits, and Optimization Ideas

**Stage:** implementation-design
**Contracts:** [scope](/design/README.md), [resume and weight storage](/design/model-weights.md), [run recovery](/design/main.md), and [agent rules](/AGENTS.md).

**Decisions:** Limit every learner, teacher, probe, and auxiliary model to 700M total parameters; this supersedes earlier size suggestions. Require periodic phase-aware resume, including CSReL auxiliary training, pending batches, RNG/policy state, and atomic replay/teacher commits. Retain the latest recovery checkpoint plus one fallback during a run, prune after verified commits, and retain only final recovery state plus algorithm-required dependencies on completion. Load teachers/auxiliary models on demand within RAM/VRAM and disk budgets; do not archive all task/epoch models.

**Idea log:** Root [optimizations.md](/optimizations.md) is the single canonical file. Agents must record candidate algorithmic improvements in six short lines with a concrete example, minimal test, and unverified-novelty/risk statement. No candidate novelty is claimed or implemented by this update.

**Validation:** Documentation integrity checks passed across 14 design documents plus `AGENTS.md` and `optimizations.md`, covering 41 local links and the six-line template. Reviewed resume/retention interactions and added required future checks for interrupted auxiliary training, dependency-safe pruning, bounded residency, and the parameter ceiling. Runtime behavior remains unimplemented and untested.

**Next action:** Implement these contracts with the later supplied model/data/runtime configuration; verify interruption recovery before long experiments. Exact model identities and byte budgets remain open.

---

## [2026-10-20] Paper References — CSReL, aL-SAR, OASIS, and TSR

**Stage:** literature-review / implementation-reference
**Lifecycle position:** Active
**Field orientation:** Problem-first

**Files:** [Paper index](/Papers/README.md), [CSReL](/Papers/CSReL-summary.md), [aL-SAR](/Papers/aL-SAR-summary.md), [OASIS](/Papers/OASIS-summary.md), and [TSR](/Papers/TSR-summary.md).

**Coverage:** One detailed reference per distinct local paper, with duplicate PDFs mapped in the index. Each reference explains the problem, core formulas, execution sequence, published evidence, native and engineering optimizations, limitations, pipeline ownership, and resume/resource implications. Related-work tables link the existing reviews and primary sources, distinguishing a shared mechanism family from direct algorithmic lineage. Original settings are evidence only; model and dataset choices remain open under the 700M ceiling.

**Source findings:** OASIS Appendix A.2 uses an unbounded historical pool and memory-only training; current-arrival-only selection in our hybrid is an adaptation. Its main equations, appendix normalization, pseudocode, and printed thresholds contain distinctions requiring pinned-code reconciliation. Numerical integration confirms that the three Appendix A.7 threshold values do not solve the stated standard-normal expectation. aL-SAR also needs explicit treatment of class-exposure history on eviction and expected-square versus squared-EMA gradient normalization. TSR requires coherent era-teacher routing and follow-up on the stated 50-example training / 100-example buffer access convention and mixed evaluation metrics.

**Integration:** Linked these references from agent rules and component contracts. Clarified the affected OASIS/aL-SAR design boundaries without changing the hybrid's declared roles. Preserved historical literature reports; their earlier shortlists, size allowances, and repository-availability notes remain dated background. No candidate algorithmic novelty was established or added to `optimizations.md`.

**Validation:** Checked 21 documents, 102 local links, four PDF checksums, three duplicate-file groups, math delimiter balance, and whitespace. Cross-checked reported result cells and mechanism equations against the local PDFs. Recomputed OASIS's calibration integral with deterministic quadrature and bisection. These are source/document checks, not runtime tests or reproduced training results; released implementation behavior has not been audited.

**Next action:** Future coding agents should read the method reference with its design contract, pin the author implementation when resolving ambiguities, and validate mechanism fidelity and interruption recovery before scientific runs.

---

## [2026-10-20] Documentation compression

Trimmed repeated overviews, caveats, generic folder descriptions, and duplicated checks across `Papers/`, `design/`, agent rules, idea log, `Idea.md`, and `code-structure.md` (about 23% fewer words). Shared constraints remain in the design index, recovery/storage in `model-weights.md`, and validation in `tests.md`. Preserved equations, related-work comparisons, source ambiguities, and implementation semantics. Documentation-only; no new research claim. Next: implement against the compact contracts using the later supplied experiment configuration.

---

## [2026-10-20] Contract and portability review

Resolved branch/exhaustion and skipped-update semantics, SAR cold start/decay/profile choices, missing-teacher failures, replay eviction versus recovery retention, task commit/evaluation ordering, and isolated attempt outputs. Initial execution is one process/device; aL accumulation needs a separate aggregation contract. Model/data/protocol inputs and paper-code reconciliation remain explicit prerequisites.

Converted authored Markdown links/templates to repository-root paths, repaired missing-rule references, and removed the machine-specific README link. Checked 232 concrete local links across 49 documents, all 14 design entries, 28 unchanged display equations, and four PDF hashes; normalization is idempotent. Runtime/training remains unimplemented; required behavioral checks are in `design/tests.md`. Resume, retention, the 700M ceiling, and the six-line `optimizations.md` rule remain required.
