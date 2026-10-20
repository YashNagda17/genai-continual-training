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
