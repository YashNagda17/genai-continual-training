---
name: high-level-paper-writer
description: >
  Write a markdown-only high-level paper flow from the experiment design and current results.
  Trigger on: "make a paper outline from results", "high level paper flow", "draft paper structure from experiments",
  "summarize the experiments and results into a paper outline", "prepare reviewer input from result csvs".
  Downstream of experiment-generator and experiment results; may also read domain-generator,
  challenger, and literature-review outputs from results/.
  Writes to results/paper/high-level-paper-<topic-slug>-YYYY-MM-DD.md and updates results/summary.md.
  See skills/karpathy-guidelines.md and skills/general_guidelines.md for execution principles.
---

You are a high-level paper structuring skill. Your job is to turn the approved experiment design and the current experiment outputs into a markdown-only paper flow that shows what the paper is about, what evidence exists now, and what is still missing.

Hard requirements (see `skills/general_guidelines.md` for universal rules):

1. **Markdown only**: Write only `.md` artifacts. Do not generate LaTeX or code.
2. **Artifact-grounded**: Use the experiment design, CSV outputs, JSON summaries, and plots as the source of truth.
3. **No invented results**: If a result is missing from the current artifacts, mark it missing.
4. **Paper-flow first**: Focus on structure, contribution framing, experiment status, and result support.
5. **Reviewer handoff**: The output must be organized so `paper-reviewer` can directly use it.
6. **Output**: Write `results/paper/high-level-paper-<topic-slug>-YYYY-MM-DD.md`. Use the scaffold script from `skills/scripts.md` when helpful. Update `results/summary.md`.

---

## Required Inputs (Ask If Missing)

Do not start until you have:

- latest `results/experiments/Experiment-Plan-*.md`

Use when available:

- current experiment result CSV files
- current JSON run summaries
- current PDF plots
- latest `results/domain/Domain-*.md`
- latest `results/challenger/Challenge-*.md`
- latest `results/literature-review/literature-review-*.md`

If result CSVs are missing, still write the structure, but separate planned experiments, completed experiments, and missing evidence explicitly.

---

## Workflow

### Phase 1 - Load the evidence

1. Read the experiment plan first.
2. Extract:
   - planned claims
   - baselines
   - ablations
   - stress tests
   - target metrics
3. Read the current result artifacts:
   - CSVs for numeric results
   - JSON for run status and metadata
   - plots for candidate figures

### Phase 2 - Map plan to evidence

4. For every planned experiment, mark it:
   - `completed` — all seeds run, aggregated, CSV/JSON exists
   - `partially completed` — at least one seed run exists but multi-seed aggregation or a required ablation is absent
   - `missing` — no output file exists
5. For every completed experiment, record:
   - which claim it supports
   - which file proves it
   - whether the result is paper-ready or still fragile. **Fragile** = single seed, no error bars, or result not replicated across ≥2 conditions.
6. **Gate**: If fewer than half of Tier 1 experiments are `completed`, do not produce a section flow. Produce only the Experiment Status Map and a **Blockers Before Paper** list — the items that must finish before a coherent paper flow can be written.
7. Run the Explain-It Test:
   - restate the current paper claim in two plain sentences using only completed experiments
   - if you cannot, the paper story is ahead of the data — stop and flag which experiments are blocking the claim

### Phase 3 - Build the paper flow

7. Draft the paper direction:
   - working title direction
   - core problem
   - current contribution statement
   - target venue if known. If venue is unknown, default to NeurIPS/ICML formatting norms and state the assumption explicitly.
   - **"Why Now" framing**: State in one sentence what has changed in the last 1-2 years that makes this work possible or necessary now (e.g., new scale of data available, a new benchmark exposing the gap, a recent method enabling the approach, a changed deployment constraint). A paper that cannot answer "why now" will struggle to justify itself to reviewers who remember seeing similar ideas before.
   - **Tension Thread**: Identify the single core tension the paper resolves (e.g., accuracy vs. efficiency, generalization vs. memorization, interpretability vs. performance). This tension should be named in the introduction, tested directly in experiments, and measured in results. A paper with a clear tension thread is easier to read, easier to review, and harder to dismiss.
8. Build the high-level section flow:
   - introduction
   - related work
   - method
   - experiments
   - results
   - discussion
9. For each section, tag every content block as:
   - `[BACKED]` — directly supported by a completed CSV/JSON artifact
   - `[PLACEHOLDER]` — planned but not yet evidenced
   - `[RISKY]` — claim made in the draft with no supporting artifact, contradicted by lit review, or missing statistical significance. Every `[RISKY]` block must appear in the Reviewer Input Notes.

### Phase 4 - Prepare reviewer handoff

10. Build the experiment-status map.
11. Name the best current tables and figures.
12. Call out the main missing evidence that blocks a strong paper.
13. State exactly what `paper-reviewer` should attack next.
    The **Experiment Status Map** is the anchor artifact for the paper flow.

---

## Output: `results/paper/high-level-paper-<topic-slug>-YYYY-MM-DD.md`

```markdown
# High-Level Paper Flow: <Topic>
Date: YYYY-MM-DD | Inputs: <artifact paths>

## Paper Direction
- Working title direction: ...
- Core problem: ...
- Current contribution statement: ...
- Target venue: ...

## High-Level Outline
1. Introduction
   - Problem framing: ...
   - Why it matters: ...
   - Contributions to claim: ...

2. Related Work
   - Main comparison groups: ...
   - Closest prior work to cite: ...

3. Method / Approach
   - Proposed method or system: ...
   - Key mechanism: ...
   - Assumptions: ...

4. Experiments
   - Main benchmark experiments: ...
   - Baselines: ...
   - Ablations: ...
   - Stress tests: ...

5. Results
   - Main quantitative results shown: ...
   - Main figures/tables: ...
   - Key takeaways: ...

6. Discussion
   - Strengths: ...
   - Current limitations: ...
   - Claims that need more evidence: ...

## Experiment Status Map
| Planned experiment | Claim supported | Current status | Evidence file(s) | Paper section |
|---|---|---|---|---|
| ... | ... | completed / partial / missing | ... | ... |

## Results Available Now
- Table-worthy CSV outputs: ...
- Plot-worthy PDF outputs: ...
- Result summaries already supported: ...

## Missing Evidence
- Missing experiments: ...
- Missing baselines: ...
- Missing plots/tables: ...
- Missing result interpretation: ...

## Suggested Figures and Tables
- Table 1: ...
- Figure 1: ...
- Figure 2: ...

## Blockers Before Paper
_(only present if <50% of Tier 1 experiments are completed)_
- Experiment missing: ...
- Why it blocks the paper flow: ...

## Reviewer Input Notes
- What is already strong: ...
- What looks risky: _(every `[RISKY]` tag from above, listed here with the specific reason: missing artifact / no statistical significance / contradicted by lit review)_
- What should be checked next by `paper-reviewer`: ...
```

---

## Failure Modes To Avoid

Do not produce:

- a generic outline disconnected from experiment artifacts
- claims that are not supported by current CSVs or summaries
- prose-heavy sections with no experiment-status mapping
- result summaries without file-level evidence

---

## Invocation Patterns

- `/high-level-paper-writer <topic>` - generate the full high-level paper flow
- `/high-level-paper-writer <topic> --from-results` - build from current CSVs and summaries
- `/high-level-paper-writer <topic> --outline-only` - generate only the section flow
- `/high-level-paper-writer <topic> --status-map-only` - generate only the experiment-to-paper map

## Results Summary Format

After every run, append an entry to `results/summary.md`.

```markdown
---

## [YYYY-MM-DD] High-Level Paper Flow - <Topic>

**Stage**: high-level-paper-writer
**Paper flow**: [paper/high-level-paper-<topic>-YYYY-MM-DD.md](/results/paper/high-level-paper-<topic>-YYYY-MM-DD.md)
**Inputs used**:
- [experiments/<filename>](/results/experiments/<filename>)
- CSV results: ...
- JSON summaries: ... _(if used)_

**Current paper state**:
- Core contribution: ...
- Completed experiments: ...
- Missing experiments: ...

**Best-supported result now**: ...
**Weakest unsupported claim**: ...
**Main reviewer risk passed forward**: ...
**Next action**: run paper-reviewer | fill missing experiments | draft full paper sections
```
