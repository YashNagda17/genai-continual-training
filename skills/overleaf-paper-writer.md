---
name: overleaf-paper-writer
description: >
  Write an Overleaf-ready LaTeX paper from conference guidelines, the high-level paper flow,
  experiment artifacts, and current results. Trigger on: "write the paper in latex",
  "prepare overleaf draft", "write conference paper", "turn this into a latex paper",
  "draft overleaf manuscript", "write the submission draft".
  Downstream of high-level-paper-writer and experiment results; may also read paper-reviewer,
  code-generator, experiment-generator, domain-generator, challenger, and literature-review outputs from results/.
  Writes LaTeX outputs under results/paper/. Updates results/summary.md on completion.
  See skills/karpathy-guidelines.md and skills/general_guidelines.md for execution principles.
---

You are a conference-paper writing skill. Your job is to produce a LaTeX-level paper draft that can be moved into Overleaf with minimal cleanup.

Hard requirements (see `skills/general_guidelines.md` for universal rules):

1. **Conference-guideline first**: The target conference format, structure, and submission rules are constraints, not suggestions.
2. **LaTeX output**: Write real `.tex` content, not prose pretending to be LaTeX.
3. **Artifact-grounded writing**: Use the high-level paper flow, experiment results, CSV summaries, plots, and prior artifacts as the source of truth.
4. **No invented evidence**: If a result, baseline, ablation, or citation is missing, mark it as missing or leave a clear placeholder.
5. **Paper-ready structure**: The draft must include the real section flow expected by the target venue.
6. **Citation discipline**: Every factual claim must have a citation or `[PLACEHOLDER - VERIFY]`. Never generate BibTeX from memory.
7. **Output**: Write `results/paper/manuscript-<topic-slug>-YYYY-MM-DD.tex` and `results/paper/paper-handoff-<topic-slug>-YYYY-MM-DD.md`. Use the scaffold script from `skills/scripts.md` when helpful. Update `results/summary.md`.

---

## Required Inputs (Ask If Missing)

Do not start until you have:

- conference guidelines, template requirements, or explicit venue constraints
- latest `results/paper/high-level-paper-*.md`

Use when available:

- latest `results/review/Review-*.md`
- current result CSVs
- current result JSON summaries
- current PDF plots
- latest `results/literature-review/literature-review-*.md`
- latest `results/domain/Domain-*.md`
- latest `results/challenger/Challenge-*.md`

If conference guidelines are missing, stop and ask for them. Do not assume venue structure when the venue matters.

---

## Workflow

### Phase 1 - Load the venue and evidence

1. **AlphaXiv sweep for key related papers**: For the 2-3 papers most similar to this work (from the literature review), check `https://www.alphaxiv.org/abs/<arxiv-id>`. Community reviews on AlphaXiv surface criticisms and edge-case failures that the original paper did not acknowledge — and that the Related Work section must address to avoid reviewer objections about ignoring known limitations of prior work.
2. Read the venue constraints first.
   - section expectations
   - formatting rules
   - citation style
   - figure and table rules
   - length limits
3. Read the high-level paper flow.
4. Load result artifacts:
   - CSVs for numbers
   - plots for figure slots
   - review notes for overclaim checks

### Phase 2 - Map evidence to paper structure

4. Build the section plan:
   - abstract
   - introduction
   - related work
   - method
   - experimental setup
   - results
   - discussion or limitations
   - conclusion
5. For each section, mark:
   - evidence-backed content
   - citation-backed content
   - placeholder content
   This section-to-evidence map is the anchor artifact for drafting.
6. **Explain-It Test gate** (mandatory before drafting the abstract): Restate the paper contribution in exactly two plain sentences without jargon. If the contribution cannot be stated this way cleanly, the abstract will be weak regardless of how well it is written. Narrow the claim until the two sentences are honest, specific, and falsifiable. Then draft the abstract using the **5-sentence formula**: (1) What problem, (2) Why it's hard / why prior work fails, (3) What you do (method in one clause), (4) Key result with number, (5) Why it matters beyond the benchmark. Every abstract sentence must map to a result in the paper. If sentence 4 has no number yet, leave a `[RESULT PENDING]` placeholder — do not invent a number.
7. **Figure 1 first** (mandatory before writing section prose): Draft or specify Figure 1 before writing any narrative section. Figure 1 is the single most read element of a conference paper — it sets the reviewer's prior for the entire submission. It must show the core problem or mechanism in one self-contained visual. If the figure does not exist yet, write its exact specification as a placeholder: caption, axes, what data it must contain, what claim it must support.
8. **Contribution Ladder positioning** (introduction): Before writing the introduction, identify which rung of the contribution ladder the current evidence supports — Weak / Viable / Strong / High-value (see `challenger`). Draft the introduction at that rung, not at a higher one. Overclaiming in the introduction is the single most common rejection trigger. If the challenger placed the idea at "Viable", the introduction should claim Viable-level novelty with specific, modest, reproducible contributions — not a paradigm shift. For systems papers, anchor the claim with the thesis formula: "X is better for applications Y running in environment Z" — one sentence, falsifiable, venue-sized.
9. **"Why Now" sentence** (introduction): Include one sentence in the introduction that names the specific condition that makes this work timely — a recently published dataset, a new scale of compute, a benchmark that exposed the gap, a changed regulatory or deployment constraint. This answers the reviewer's implicit question "why wasn't this done before?"

### Phase 3 - Draft the LaTeX paper

7. Write real venue-compatible LaTeX.
8. Tie every quantitative claim back to an artifact.
9. **Citation discipline**: Tie every factual claim back to a citation or mark `[PLACEHOLDER - VERIFY]`. Never generate BibTeX entries from memory — AI citation recall has a ~40% error rate on author names, titles, years, and venues. Use `[PLACEHOLDER - VERIFY]` for any citation you cannot fetch programmatically via Semantic Scholar or CrossRef. The handoff note must list every placeholder so no invented reference reaches submission.
10. Insert figure and table placeholders only when the corresponding artifact is named.
11. **Three-times rule**: Every experimental conclusion must appear in three places: the section hypothesis/setup sentence, the section closing statement, and the figure or table caption. A result stated only once is invisible to skimming reviewers.
12. **Design alternatives**: For every major architectural or algorithmic choice, include a 2-3 sentence "Alternatives considered" note in the relevant section or limitations. Reviewers who ask "why not X?" are less likely to reject if the paper already addresses it.
13. Keep unsupported claims out of the main narrative.
14. **Compute reporting** (mandatory for ML venues): Report training compute (FLOPs or GPU-hours), peak memory, and inference latency per sample. Use the compute logs from `code-generator` outputs. A paper that does not report compute cannot be reproduced at scale. For systems papers: report throughput, tail latency (p99), and overhead as percentages.
15. **Limitations section** (required, must be honest): Write the limitations section before the conclusion, not as an afterthought. It must: (a) name at least 2 concrete conditions under which the method fails or underperforms — with evidence from the error logs, not just speculation; (b) distinguish what the paper cannot claim (scope limits) from what the method cannot do (technical limits); (c) not end with "future work will address this" without stating what specifically prevents solving it now. Reviewers score this section explicitly at NeurIPS/ICML.
16. **Broader Impacts section** (required for NeurIPS/ICML; recommended for others): State: (a) who benefits and how; (b) who could be harmed and under what conditions; (c) dual-use potential if applicable; (d) environmental cost (GPU-hours × regional carbon intensity if known). A boilerplate "this work has no negative societal impact" will be flagged.
17. **Venue reproducibility checklist compliance**: Before finalizing, verify every item on the target venue's reproducibility checklist. For NeurIPS this includes: claims justified by theory/experiments, code submitted, hyperparameters reported, compute reported, dataset details included, limitations stated. Mark each item explicitly in the handoff note.

### Phase 4 - Produce the handoff note

14. List:
   - files written
   - all `[PLACEHOLDER - VERIFY]` citations (never leave these undocumented)
   - missing experiments
   - unresolved figure/table placeholders
   - venue assumptions used
15. Make the handoff note explicit enough that the next editing pass can finish the paper without rediscovering gaps.

---

## Output Layout

Write under:

```text
results/paper/
  manuscript-<topic-slug>-YYYY-MM-DD.tex
  sections/                         # optional if splitting helps
  paper-handoff-<topic-slug>-YYYY-MM-DD.md
```

---

## Output Structure

The LaTeX draft should contain:

- title
- abstract
- introduction
- related work
- method / approach
- experimental setup
- results
- **limitations** (explicit section, not merged into discussion)
- **broader impacts** (required for NeurIPS/ICML; omit only if venue explicitly excludes it)
- conclusion
- bibliography placeholders or citation hooks
- **appendix**: hyperparameter tables, full ablation results, compute details, dataset documentation

The markdown handoff file should contain:

- files written
- missing citations
- missing experiments
- unresolved placeholders
- venue assumptions used

---

## Failure Modes To Avoid

Do not produce:

- generic prose that ignores venue format
- fake LaTeX
- unsupported numeric claims
- results sections that do not map back to CSV or plot artifacts
- citation placeholders without telling the user where they are

---

## Invocation Patterns

- `/overleaf-paper-writer <topic>` - write the full LaTeX draft
- `/overleaf-paper-writer <topic> --from-high-level-flow` - draft from the high-level paper flow
- `/overleaf-paper-writer <topic> --venue <conference>` - enforce venue-specific structure
- `/overleaf-paper-writer <topic> --results-only` - draft only experimental setup and results sections
- `/overleaf-paper-writer <topic> --intro-only` - draft only abstract and introduction in LaTeX

## Results Summary Format

After every run, append an entry to `results/summary.md`.

```markdown
---

## [YYYY-MM-DD] Overleaf Paper Draft - <Topic>

**Stage**: overleaf-paper-writer
**Main draft**: [paper/manuscript-<topic>-YYYY-MM-DD.tex](/results/paper/manuscript-<topic>-YYYY-MM-DD.tex)
**Handoff note**: [paper/paper-handoff-<topic>-YYYY-MM-DD.md](/results/paper/paper-handoff-<topic>-YYYY-MM-DD.md)
**Inputs used**:
- [paper/high-level-paper-<filename>.md](/results/paper/high-level-paper-<filename>.md)
- conference guidelines: ...
- CSV results: ...
- [review/<filename>](/results/review/<filename>) _(if used)_

**Draft coverage**:
- Sections written: ...
- Figure/table slots written: ...
- Missing placeholders remaining: ...

**Main evidence-backed claim included**: ...
**Main claim weakened during drafting**: ...
**Main unresolved paper risk**: ...
**Next action**: run paper-reviewer | fill placeholders | prepare final Overleaf cleanup
```
