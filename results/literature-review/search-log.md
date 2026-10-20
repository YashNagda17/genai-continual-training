# Literature-search audit — 2026-09-28

Topic: the incremental fine-tuning proposal in `Idea.md`, with emphasis on useful impact, historical-data selection, replay costs, and data formatting. The report contains exactly **15 selected works**. The selection is a curated starting review, not an exhaustive systematic review or a claim to identify a universal state of the art.

## Scope and reading evidence

- Date window: 2023-09-28 through 2026-09-28. Conference dates and first-preprint dates are distinguished in the report and candidate metadata.
- Primary sources: ACL Anthology, arXiv, official ICLR proceedings, PMLR/CoLLAs, and author repositories. Third-party search hits were discovery aids; reported numerical results were checked against downloaded primary PDFs.
- Sixteen primary PDFs and extracted text files are retained in `sources/`. Fifteen received the five-paragraph treatment in the main report. P15, the implicit-inference paper, was read as background but not selected: its first preprint precedes the window and OASIS is a closer recent comparator.
- Reading level: targeted full-text inspection of methods, result tables, relevant appendices, and limitations. This does not mean all proofs were checked, code was executed, or results were reproduced.
- Text-LLM supervised/instruction tuning is the working interpretation. Raw-text continual pretraining and multimodal selection appear as explicitly adjacent evidence. The application, GPU capacity, model, and concrete incoming-data format remain unspecified.

## Search and selection record

The initial search used combinations of continual/incremental fine-tuning, instruction tuning, catastrophic forgetting, representative replay, data selection, replay budgets, output formatting, and compute-efficient continual pretraining. Named-paper follow-ups located InsCL, SEEKR, KPIG, OPR, SSR, Adapt-infinity, CITB, TRACE, O-LoRA, I-LoRA, the two CPT studies, Bonito, aligned-CPT forgetting, and OASIS. Exact initial browser-query history is not reconstructed here.

The preserved API query was:

```bash
PYTHONPATH=src python -m litreview.openalex_search 'continual instruction tuning replay' --limit 50 --out results/literature-review/oa-results.json
```

It returned 50 records ordered toward recent publication dates, many of limited relevance. Therefore it is an audit/discovery input, not the basis of the final reading priority. Manual primary-source verification produced `primary-candidates.json` with 16 records.

The repository utilities were then run to preserve a reproducible lexical-screening audit:

```bash
PYTHONPATH=src python -m litreview.merge_candidates results/literature-review/primary-candidates.json results/literature-review/oa-results.json --out results/literature-review/candidates-merged.json
PYTHONPATH=src python -m litreview.abstract_screen results/literature-review/candidates-merged.json 'continual instruction tuning replay forgetting language models data selection' --min-score 0.1 --out results/literature-review/candidates-screened.json --rejected-out results/literature-review/candidates-rejected.json
PYTHONPATH=src python -m litreview.rank_candidates results/literature-review/candidates-screened.json 'continual instruction tuning replay forgetting language models data selection' --top 30 --out results/literature-review/candidates-auto-ranked.json
```

Counts: **66 merged records; 59 lexical accepts; 7 lexical rejects; 30 automatic ranking entries; 15 manually selected works.** These are record counts, not 66 deeply reviewed papers. The permissive lexical screen does not establish relevance. The manual decision prioritized direct overlap with the idea, concrete evaluation/data protocols, and useful adjacent mechanisms. `candidates-ranked.json` is that manual reading order; `candidates-auto-ranked.json` preserves the heuristic ranking separately.

## Final supplementary novelty check

These exact eight web queries were run on the review date. Their returned search text is saved in `search-final-1.json` and `search-final-2.json`; those files include discovery snippets, which are not independent factual validation.

1. `"continual instruction tuning" "replay" "data selection"`
2. `"continual supervised fine-tuning" "replay" 2026`
3. `"continual instruction tuning" "fixed" "memory"`
4. `"continual instruction tuning" "format"`
5. `"continual pre-training" "replay" "compute"`
6. `"instruction tuning" "historical" "token budget"`
7. `"continual instruction tuning" "OASIS" "OPR"`
8. `"continual instruction tuning" "format" "buffer" 2025 2026`

The supplementary search reconfirmed OPR, CITB, and replay/gradient-alignment CPT. It also surfaced broader multimodal or federated leads, including [ProtoAda](https://arxiv.org/abs/2606.02576), [Federated Continual Instruction Tuning](https://arxiv.org/abs/2503.12897), and [Continual Instruction Tuning for Large Multimodal Models](https://arxiv.org/abs/2311.16206). These are **search/abstract-screened leads, not additional full reviews**, and are not counted among the 15. ProtoAda's abstract explicitly concerns output structure in routing, so a future broad claim that format-aware continual adaptation is new would need to compare against it. The present report makes no such novelty claim. If the project becomes multimodal or federated, these leads require a dedicated review before deciding its contribution.

## Metadata and reproducibility boundaries

- ACL/ICLR publication status was taken from primary proceedings. TRACE remains labeled a preprint because acceptance was not verified; a submission page or aggregator venue tag does not prove acceptance.
- I-LoRA's reviewed 2024 preprint and later ICASSP 2025 publication have differing author metadata. The reviewed version is identified explicitly.
- The replay/gradient-alignment work has a 2025 preprint/CoLLAs conference and 2026 PMLR proceedings record. The report and bibliography identify the reviewed preprint rather than silently combining author lists.
- Citation counts are DOI-specific OpenAlex snapshots, with raw responses in `sources/Pxx-openalex.json`. Citation velocity divides that record's count by age; month-only dates are normalized to the first of the month for this rough calculation. Counts are not aggregated across preprint/proceedings versions, are not Semantic Scholar counts, and should not be used to declare overall influence.
- An author code link establishes a reported implementation, not a successful installation or reproduction. Dataset downloads, licenses, and hardware feasibility were not audited.
- The canonical report is [literature-review-incremental-fine-tuning-2026-09-28.md](/results/literature-review/literature-review-incremental-fine-tuning-2026-09-28.md). Bibliography: [references.bib](/results/literature-review/references.bib). Selection metadata: [candidates-ranked.json](/results/literature-review/candidates-ranked.json).
