# Search log — pipeline upgrades

Date: 2026-11-01. The `src/litreview` helpers named in [skills/scripts.md](/skills/scripts.md) are not in the tree, so candidates were collected with web search, arXiv HTML, conference PDFs, and an OpenAlex work lookup. Raw JSON from those calls was not retained; this log is the audit trail. Prior reviews under `results/literature-review/` were not overwritten.

## Topic

Alternate papers that can raise accuracy or cut cost in the current task-chunk hybrid: OASIS admission, TSR task routing, aL-SAR retrieval and freezing, CSReL task-end memory. Cross-domain sources were CFD goal-oriented sampling and biology of replay, tagging, and synaptic plasticity.

Window: first public date on or after 2023-11-01. Today is 2026-11-01.

## Queries

- Online gradient data selection, continual instruction tuning, replay budget, 2024–2026.
- Task-similarity routing, LoRA continual learning, distillation, gradient alignment.
- Adaptive layer freezing, Fisher information, LISA, rank-selective LoRA.
- Coreset selection, reducible loss, rehearsal memory.
- Goal-oriented adjoint and dual-weighted-residual sampling for PINNs.
- Complementary learning systems, synaptic tagging, awake replay.
- Title checks: TreeLoRA, InfLoRA, SD-LoRA, LESS, GREATS, BCSR, SHARP, RoD, ASO-LoRA, CoDyRA, arXiv:2604.01835.

## Screening decisions

Accepted for the ten-paper deep read: GREATS, LESS, Replay on Demand, TreeLoRA, ASO-LoRA, SD-LoRA, CoDyRA, LISA, Govoeyi and Richter (DWR sampling), van der Meer and Bendor (awake replay).

Rejected or held out:

- BCSR (NeurIPS 2023) is a bilevel memory writer that CSReL already positions itself against. It is a predecessor, not an upgrade of the current writer.
- GCR (2021) is older than three years.
- SHARP first appeared as arXiv:2305.18563 in May 2023. The ICDL 2024 copy is outside the first-appearance window used here.
- InfLoRA is the main rehearsal-free baseline inside the SD-LoRA tables, so it is not a separate deep read.
- DRDF-LoRA had an abstract and no result table in the sources retrieved.
- LoDA (arXiv:2603.00191) and TORA (arXiv:2610.01702) are real routing neighbors. They were read at abstract and method level and were not promoted ahead of TreeLoRA and ASO-LoRA.
- FLoE is Fisher layer selection for sparse LoRA experts. It overlaps aL-SAR without a continual instruction result in the pages read.

## Citation counts

OpenAlex matches on 2026-11-01 are version records, not a field ranking. GREATS's NeurIPS record showed 6 citations. LESS's arXiv record showed 19. Awake replay showed 13. The other matched records showed 0, including LISA, which is not credible as a field count. These numbers are not used as influence ranks.

## Round 2 (2026-11-01) — length of the admission gradient

The refined question was whether OASIS energy, and the logged probe-inner-product alternative, already contain a completion-length bias, and which published scores would duplicate a one-line rescaling of that energy.

Queries: LESS sequence-level gradient norm versus completion length; token-level influence; Grad-Mimic weight-space projection; LeaF gradient-guided token pruning. arXiv title search confirmed `2501.06708` and `2506.07851`.

Accepted into the same review: Grad-Mimic and LeaF, as boundaries. LESS Table 13 and its sequence-gradient appendix were read from the existing LESS HTML and folded into the LESS section. No OpenAlex counts were collected for the new papers.

Held out of the five-paragraph set: NICE (policy-gradient selection of a non-differentiable metric). The same loss-versus-accuracy split is already stated in LESS Appendix J, and NICE's result table was not extracted. IProX and SkMM stay out; they are proxy-model selectors already screened as a second model.

## Round 3 (2026-11-01) — token excess and the replay multiplier

The question was whether the admission gradient should change with response length, or whether the mean over tokens and the fixed replay probability are the quantities theory actually names.

Queries: selective language modeling excess loss; primal-dual continual learning Lagrangian; adaptive replay scheduling for LLM fine-tuning; conditional token selection in chain-of-thought. arXiv title search confirmed `2404.07965`, `2310.00154`, `2603.09892`, and `2505.17827`. HTML for those four was read. No OpenAlex counts were collected.

Accepted into the same review: Rho-1, PDCL, MSSR, and conditional token selection.

Held out: MSL, a recommender tuning paper returned by the same title query. It is a different task. GEM and A-GEM are older than the window and appear inside the PDCL related work as the constraint predecessors.
