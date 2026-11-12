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

**Files:** [Paper index](/Papers/README.md), [CSReL](/Papers/CSReL-summary.md), [aL-SAR published paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html), [OASIS published paper](https://aclanthology.org/2026.acl-long.158/), and [TSR](/Papers/TSR-summary.md).

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

---

## [2026-10-20] Code Generation — Pipeline Algorithms

**Stage:** code-generator.
**Code/API:** [pipeline modules](/src/pipelines/), [combined pipeline](/src/combined_pipeline.py), and [adapter guide](/src/README.md).
**Inputs:** current [design contracts](/design/README.md), [paper references](/Papers/README.md), and the request for pipeline algorithms with injected model/data implementations.

**Implemented:** OASIS exact equation profile; TSR probing, source/teacher routing and KD objective delegation; native/hybrid aL-SAR retrieval and adaptive freezing; CSReL-CL/CL-Prv repeated auxiliary fitting and coreset construction; one composed learner lifecycle. Added abstract model/dataset methods, typed records, replay validation, total-parameter checks, phase-aware resume, bounded recovery retention, teacher dependencies, and on-demand numerical role residency.

**Artifacts:** each attempt writes resolved configuration/identity/status JSON, long-form metrics CSV, committed event diagnostics, verified recovery state and required teachers. The configuration records explicit scientific inputs; no production model or dataset was selected.

**Validation completed:** 32 deterministic tests passed in 103.762 seconds, including two-task standalone/hybrid runs and interruption/recovery comparisons. Syntax/import compilation, whitespace checks and 85 repository-root links passed. The test executor is a small squared-error model; these checks establish policy/orchestration behavior, not transformer or paper reproduction results.

**Open boundaries:** concrete model/dataset adapters must implement numerical training, actual gradient representations, masking, optimizer freezing, KD, isolated RNG, hardware measurements and external asset accounting. GPU execution, author-code equivalence and scientific training remain unverified. The experiment matrix/CLI and optional CSReL-RS/KD variants remain outside this implementation.

**Current claim:** the selected pipeline algorithms are callable and tested with injected numerical/data interfaces. No algorithmic novelty or scientific performance result is claimed; the sole idea log is unchanged.

**Next action:** implement concrete adapters against the base classes, then validate backend gradient/loss/resume semantics before running the supplied experimental protocol.

---

## [2026-10-20] Code Generation — Modular Pipeline Steps

**Stage:** code-generator.
**Code/API:** [composition guide](/src/README.md), [pipeline classes](/src/pipelines/), and [model interface](/src/models/model_base.py).

**Implemented:** separate public paper steps; shared learner replacement; detached weight and file handoff; independent policy/RNG snapshots. Moved generic lifecycle execution into utilities and consolidated replay/checkpoint storage and numerical/metric helpers. Removed explicit algorithm validation and exception handling from pipeline, model and dataset modules for the requested logic-review stage. Updated the contracts before implementing these interface changes.

**Validation completed:** 38 tests passed in 108.144 seconds, including equations, handoff, two-task standalone/hybrid execution, recovery and retention. Python compilation passed. The squared-error test adapter does not validate transformer, LoRA/KD or GPU behavior. No scientific result or novel algorithm is claimed.

**Next action:** review the named steps and implement concrete adapters; full backend validation remains necessary before experiments.

**Review documentation follow-up:** added short helper comments and fuller algorithm/state explanations in the pipeline, adapter and utility methods. Executable ASTs are unchanged across 14 modules; compilation, whitespace and 18 equation/composition tests passed.

---

## [2026-10-20] Native Torch and Package Cleanup

**Stage:** code-generator.
**Code/API:** [adapter guide](/src/README.md), [model base](/src/models/model_base.py), and [numerical helpers](/src/utils/numerics.py).

**Implemented:** native Torch policy calculations and model execution contracts; CUDA/ROCm and Intel XPU routing with CPU fallback; nested tensor transfer preserving dtypes/autograd; deterministic weighted sampling from saved policy RNG; Torch version/device in resume identity. Weight snapshots, shared learner identity and abstract concrete-adapter obligations remain available. Torch is declared in `requirements.txt`; package installation is deferred.

**Cleanup:** removed the two requested dated Codegen reports and unnecessary package initializers. Agent rules now require namespace imports and explicitly needed initialization only. Training and inference use native Torch.

**Validation:** the full suite ran 44 tests in 110.163 seconds: 43 passed and the real accelerator test skipped because no CUDA/ROCm or XPU device was available. The tiny Torch model exercised autograd, weight handoff, standalone/hybrid execution and recovery; routing mocks covered all requested backends. Compilation, namespace CLI imports, 83 local documentation links and scoped whitespace checks passed. GPU execution and production transformer/LoRA/KD behavior remain unverified; no scientific result is claimed.

---

## [2026-10-30] Concrete Torch Model Adapters

**Stage:** code-generator.
**Code/API:** [Qwen3](/src/models/model_qwen3.py), [LFM2.5](/src/models/model_lfm25.py), [SmolLM2](/src/models/model_smollm2.py), [shared executor](/src/models/model_causal_lm.py), and [usage guide](/src/README.md).

**Implemented:** the three requested architecture profiles with shared native Torch training and inference. The executor supplies full fine-tuning and native LoRA, response-masked causal losses, head/subset/LoRA-B gradients, exact masked teacher KL, supervised Fisher/feature statistics, adaptive freezing, generation and evaluation. Transformers supplies architectures and tokenizers; external inference backends are not used.

**State integration:** lazy role construction, unique parameter counts and the 700M ceiling, isolated numerical roles, GPU routing with CPU fallback, bounded disk offload, immutable base references, full resume state, and independent weight handoff. Replacing a learner binds the receiving runtime's dataset and storage budget. Exact resume records Torch/device identity; frozen weight transfer retains architecture compatibility.

**Non-thinking inputs:** Qwen templates always use `enable_thinking=False`; preprocessing drops reasoning-tagged targets, masks prompt/padding labels and records the tokenizer/template identity. The same preprocessing contract applies during training and evaluation.

**Execution boundary:** no tests or verification commands ran in this continuation, as requested. No test files were added, packages installed or production weights downloaded. Production architecture and GPU execution remain unverified. No scientific performance result or novel algorithm is claimed.
---

## [2026-11-03] Three-task CSReL development pilot preparation

Implemented `--protocol pilot`: seeded class-balanced CLINC training and validation views, explicit intent-label instructions, coverage/ID artifacts, prediction files per stage/task, generation-length reservation, and a cumulative full-pass budget guard. Comparison Markdown includes measured auxiliary fitting and scoring components. Model initial snapshots and attempt matching include configuration identities. Existing smoke results are preserved; pilot output stays under the ignored CSReL output directory.

Validation: syntax parsing of five changed Python modules and `git diff --check` passed. Added regression cases for balanced sampling, prefix rejection, and cumulative budget rejection. Tests and training have not been executed in this continuation: the user will run the supplied commands. Pilot performance, CUDA training for this configuration, and repeated real-model selection remain unverified.

Next: user runs fixture tests, then the three-task pilot; inspect coverage, generated predictions, task-stage metrics, and selection costs before selecting a scientific protocol.


---

## [2026-11-04] 600-example audit and learner-order ablation

Audit: [pilot-600-audit](/results/experiments/pilot-600-audit.md). All four arms completed; random 81.56%, CSReL 54.44% final validation exact match. Prepared opt-in deterministic learner shuffling; auxiliary order unchanged. Added RNG/membership and resume tests. Syntax checked; user runs tests and training. Existing results preserved. No scientific superiority or convergence claim.

---

## [2026-11-04] Pre-push verification

Saved portable one-seed pilot metrics and audit under results/experiments/. Verified Python syntax for eight changed modules, staged whitespace integrity, and exclusion of dataset/model/raw-run outputs. Ran `python -m unittest src.tests.test_csrel_baselines src.tests.test_pipelines`: 42 tests in 236.858 seconds, OK with one skipped. These are fixture/orchestration checks; shuffled real-model performance and Kaggle execution remain unverified. No language-model training was launched during this verification. Kaggle notebook creation is pending and is not included in this commit.

---

## [2026-11-08] Summarization Pipeline Components

**Stage:** code-generator.
**Code report:** [summarization pipeline implementation](/results/code/Codegen-summarization-pipelines-2026-11-08.md).
**Source:** [revision](/src/pipelines/revision.py), [adapter room](/src/pipelines/adapter_room.py), [generation shrink](/src/pipelines/generation_shrink.py), and the [callable guide](/src/README.md).
**Inputs:** [Idea_Refined_2.md](/Idea_Refined_2.md), [revision contract](/design/revision.md), [experiment contract](/design/summarization-experiments.md), and [summarization guardrails](/design/summarization-plan.md).

**Implemented:** English validation checkpoint selection and immutable past-summary pool; fixed-pass Hindi revision, loss/drift ranking, past-text filtering, G3 same-forward token weights and retirement, full-English and extra-random controls; C1 maps/stability, C2/E plans, and F's bidirectional generation-tested shrink search. Shared metrics, split checks, role isolation, complete numerical recovery, frozen-adapter storage estimates, and atomic pinned exports support these components. Components remain independent and use the existing lifecycle.

**Outputs:** attempt configuration/status/compute JSON, metrics CSV, event JSON, guarded/unguarded predictions, location-map JSON, shrink-table JSON, and immutable English/pool/Hindi/shrunk artifacts. Scientific comparison status remains unassessed. No publication PDF or scientific-result claim is produced.

**Validation:** 77 tests ran in 122.588 seconds; all executed tests passed and one setup test skipped. The final focused rerun passed all 6 pipeline tests in 41.668 seconds after identity/retention changes. Local tiny mT5 tests cover English→Hindi execution, maps, both shrink-search directions, same-forward weighted-loss equivalence, zero-probe function preservation, interrupted ranking/past-summary writing/stop regeneration, and a torn pinned write. Python compilation and whitespace checks passed. No model/data download or production GPU run was performed.

**Remaining:** scientific matrix CLI, pre-registration and C1-prediction gates, D/E winner composition, serving benchmark, seed aggregation, manual-check sheets, and publication reports. Interrupted probe/shrink analysis jobs and CUDA/XPU execution remain unverified. E's illustrative 659,456 versus 638,976 parameter counts exceed the required 3% match and are rejected.
**Next action:** implement the experiment runner against these tested pipeline APIs, then run the real pilot before any scientific arm.

---

<<<<<<< Updated upstream
<<<<<<< Updated upstream
## [2026-11-11] Imported Phase 1 results and exploratory Phase 2 handoff

**Results:** [Phase 1 training report](/results/code/summarization-passes5/phase1/training-report.md),
[CSV](/results/code/summarization-passes5/phase1/training-report.csv), and
[JSON](/results/code/summarization-passes5/phase1/training-report.json).
The imported run completed four English epochs and four Hindi epochs on each
of seeds 1, 2 and 3. Validation English chrF fell from 20.7666 to 14.2607 after
Hindi training; Hindi rose from 13.5387 to 22.0398. English lead-1 scores 22.0174,
so the scientific gate remains `fix_english`. Serving was interrupted; the
source Phase 1 is not declared fully completed.

The user explicitly authorized exploratory Phase 2 with this checkpoint.
[The handoff entry point](/src/phase2_from_checkpoint.py) verifies artifacts,
prepared subsets, prediction IDs/references and chrF, rebases source paths in
memory, and prepares a separate pilot output root. Every Phase 2 arm uses Hindi
seed 1 and four epochs; English training and Phase 1 serving are not repeated.
The failed source gate remains in the handoff and Phase 2 reports. Past-summary
replay has 400 eligible pool examples. The local base configuration matches all
four imported adapters' tensor names/shapes; values are finite. Source base
revision is only `main`, not an immutable commit.

**Validation:** 13 CPU handoff/rule tests passed. CUDA availability was checked
on the RTX 3050 Ti; no GPU model, generation or training was launched. The
preparation-only command succeeded. Run instructions and resume behavior are in
[src/README.md](/src/README.md).

---

## [2026-11-11] Exploratory Phase 2 completed

All five seed-1 arms completed four Hindi epochs (500 updates each), reusing the
imported English checkpoint. [Readable results](/results/code/phase2-exploratory/phase2/summary.md)
and full Markdown/JSON/CSV reports, predictions and five final adapter exports
are saved under `results/code/phase2-exploratory`. Random replay has the highest
English/Hindi validation chrF (21.2132/22.4333); loss (20.3709/22.0211), drift
(19.4991/21.6819), past-summary (19.1196/21.7659), and repair-stop
(20.8830/21.7392) do not improve both validation scores over random. Repair-stop
used the loss selector and retired zero examples. Summed arm wall time was
149.0824 minutes; this is not GPU busy time.

The source scientific gate remains `fix_english`; these are single-seed
exploratory results. Test scores are report-only. The final adapters are
inference exports; exact-resume checkpoints and verbose logs stay local.

**Validation:** 13 CPU handoff/rule tests passed; all five final adapter checksum
manifests and chrF across all 14,000 saved prediction rows were verified.
=======
=======
>>>>>>> Stashed changes
## [2026-11-11] Project Report - Multilingual Summarization (Idea_Refined_2)

**Stage**: paper writing (user-specified report format, not the high-level-paper-flow template)
**Report**: [paper/report-multilingual-summarization-2026-11-11.md](/results/paper/report-multilingual-summarization-2026-11-11.md)
**Inputs used**: [Idea_Refined_2.md](/Idea_Refined_2.md), [CSReL summary](/Papers/CSReL-summary.md), [TSR summary](/Papers/TSR-summary.md), [revision contract](/design/revision.md), [experiment contract](/design/summarization-experiments.md)

**Contents**: Abstract, Introduction, Literature Review (CSReL and TSR only), Idea (Phases 0-6 with motivation, first-order theory, and changes from each paper), Experiment Setup (blank), Results (blank), Conclusion and Future Work. No citations, by request. Every number in a worked example is labelled illustrative.

**Current paper state**: method and pre-registered predictions are written; no scientific run exists, so no result is claimed.
**Main framing**: the model's own writing decides what to revise (G1), with what (G2), when to stop (G3), and how small to shrink (F); G5/G4 split wrong-language from lost-content forgetting.
**Next action**: fill Sections 4 and 5 after the pilot and the scientific runs.

---

## [2026-11-12] Small cached-distillation comparison

User requested a small implementation and will execute the real run on GPU.
Added [pilot runner](/src/distillation_pilot.py), [isolated model extension](/src/models/model_distillation.py), and [protocol](/design/distillation-pilot.md).
Replay and replay-plus-KL use identical starting mT5 English weights, examples and seeds. Outputs include validation predictions, scores, cache costs and resumable state.
CPU numerical and tiny-model integration checks exercise the implementation; real-weight quality results remain pending. [Run command and scope](/results/code/Codegen-distillation-pilot-2026-11-12.md).

---

## [2026-11-12] Weaker distillation GPU comparison completed

Ran strength 0.25 versus identical ordinary replay on RTX 3050 Ti: seeds 1/2/3,
20 updates each, 32 validation articles per language. [Report](/results/code/distillation-strength025-gpu/report.md).
Mean KD-minus-replay chrF: English -0.9374, Hindi +0.5046. All three seeds fail the
predeclared English-retention success rule. Mean training times 6.86s replay,
7.69s KD plus 1.50s cache. Verified IDs and recomputed metrics for 384 predictions.
Keep ordinary replay as default; distillation stays experimental. Small tuning
run with overlapping validation data; no independent confirmation or scientific
improvement claim. Baseline English chrF does not consistently forget at 20 updates.
