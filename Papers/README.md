# Paper implementation references

Source review: 2026-10-20; no training reproduction or released-code audit. Read the relevant summary with its [design contract](/design/README.md).

| Paper | Detailed reference | Local source and duplicate copies | Project role |
|---|---|---|---|
| CSReL, ICLR 2025 | [CSReL-summary.md](/Papers/CSReL-summary.md) | [Conference PDF](/Papers/ICLR-2025-coreset-selection-via-reducible-loss-in-continual-learning-Paper-Conference.pdf); `CSReL` is identical | Task-end memory construction and reselection |
| TSR, July 2026 preprint | [TSR-summary.md](/Papers/TSR-summary.md) | [TSR.pdf](/Papers/TSR.pdf); `Replay.pdf` is identical | Historical-task routing and era-matched distillation |

Duplicates were verified by SHA-256; checksums are in the summaries.

OASIS and aL-SAR local PDFs, summaries and dedicated designs were removed at the
user's request. Mixed literature reviews retain their historical discussion and
link to the published sources. CSReL and TSR baselines are implemented under
`src/new/pipelines/`; this does not establish faithful paper reproduction.

**Reading conventions:** Results and reference settings belong to the papers, not our future experiment configuration. Engineering optimizations preserve behavior; adaptations change it. “Same family” means a shared mechanism, not direct lineage. Project scope, resume, and storage rules live in `design/`; untested algorithmic ideas go in [optimizations.md](/optimizations.md).

## Existing review map

Older shortlists, model-size allowances, and repository-availability observations are dated background. Foundational works below are older than three years.

| Review | Useful sections |
|---|---|
| [Incremental fine-tuning](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md) | §1 InsCL, §2 SEEKR, §4 OPR, §5 OASIS, §6 SSR, §7 Adapt-∞ |
| [Foundational selection](/results/literature-review/small-model-selection/six-selection-papers-2026-09-30.md) | §1 ER, §2 iCaRL, §3 GSS, §4 MIR, §5 ASER, §6 Data Diet |
| [Recent selection](/results/literature-review/small-model-selection/six-recent-selection-papers-2026-09-30.md) | §1 CSReL, §2 aL-SAR, §3 RDS+, §4 LEAD, §5 OASIS, §6 TSR |
| [Linked reproductions](/results/literature-review/small-model-selection/five-linked-reproductions-2026-10-02.md) | §1 RDS+, §2 S2L, §3 LEAD, §4 TSR, §5 OPR |
