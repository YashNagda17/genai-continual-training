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
