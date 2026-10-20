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
