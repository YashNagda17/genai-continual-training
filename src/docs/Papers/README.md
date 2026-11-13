# Paper implementation references

Source review: 2026-10-20; no training reproduction or released-code audit. Read the relevant summary with its [design contract](/design/README.md).

| Paper | Detailed reference | Local source and duplicate copies | Project role |
|---|---|---|---|
| CSReL, ICLR 2025 | [CSReL-summary.md](/src/docs/Papers/CSReL-summary.md) | [Conference PDF](/src/docs/Papers/ICLR-2025-coreset-selection-via-reducible-loss-in-continual-learning-Paper-Conference.pdf); `CSReL` is identical | Task-end memory construction and reselection |
| TSR, July 2026 preprint | [TSR-summary.md](/src/docs/Papers/TSR-summary.md) | [TSR.pdf](/src/docs/Papers/TSR.pdf); `Replay.pdf` is identical | Historical-task routing and era-matched distillation |

Duplicates were verified by SHA-256; checksums are in the summaries.

OASIS and aL-SAR local PDFs, summaries and dedicated designs were removed at the
user's request. The retained summaries link directly to published sources.
CSReL and TSR baselines are implemented under
`src/pipelines/csrel.py` and `src/pipelines/tsr.py`; this does not establish faithful paper reproduction.

**Reading conventions:** Results and reference settings belong to the papers, not our future experiment configuration. Engineering optimizations preserve behavior; adaptations change it. “Same family” means a shared mechanism, not direct lineage. Project scope and storage rules live in `design/`; untested algorithmic ideas go in [optimizations.md](/optimizations.md).

Related work is cited directly in the CSReL and TSR summaries above. Historical local review artifacts are no longer retained.
