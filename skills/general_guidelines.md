---
name: general_guidelines
description: >
  Shared guidelines for all skills in this project. Every skill must follow these conventions.
  Referenced from: literature-review, challenger, domain-generator, experiment-generator,
  code-generator, high-level-paper-writer, overleaf-paper-writer, paper-reviewer, karpathy-guidelines.
  Do not repeat these guidelines inside individual skill files — reference this file instead.
---

## 1. Output Directory Structure

All skill outputs go under `results/`. Never write to `resources/` (legacy path, no longer used).

Project overrides: implementation code belongs in `src/`, contracts in `design/`, and the four paper references in `Papers/`; see [design/README.md](/design/README.md). Use repository-root local links (`/design/README.md`), following [GitHub's repository Markdown convention](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes#relative-links-and-image-paths-in-readme-files).

```
results/
  summary.md                    ← living journal; every skill appends an entry here
  literature-review/            ← lit review artifacts
    literature-review-<slug>-YYYY-MM-DD.md
    arxiv-results.json
    s2-results.json
    oa-results.json
    candidates-merged.json
    candidates-screened.json
    candidates-ranked.json
    references.bib
  challenger/                   ← challenge reports
    Challenge-<slug>-YYYY-MM-DD.md
  domain/                       ← theory-grounding reports
    Domain-<slug>-YYYY-MM-DD.md
  experiments/                  ← experiment plans
    Experiment-Plan-<slug>-YYYY-MM-DD.md
  code/                         ← generated code + manifest
    <topic-slug>/
    Codegen-<slug>-YYYY-MM-DD.md
  paper/                        ← drafted paper sections and bibliography
    high-level-paper-<slug>-YYYY-MM-DD.md
    manuscript-<slug>-YYYY-MM-DD.tex
    paper-handoff-<slug>-YYYY-MM-DD.md
    references.bib
  review/                       ← reviewer-style paper critique
    Review-<slug>-YYYY-MM-DD.md
```

Rules:
- All scripts get `--out results/<subdir>/<filename>` explicitly — never rely on script defaults.
- Subdirectory is created by the first script invocation if missing.
- All files are date-stamped (YYYY-MM-DD) and topic-slugged where applicable.
- Each skill should name its exact primary output path near the top of the file.

## 2. results/summary.md — Living Research Journal

`results/summary.md` is the single entry point for understanding the state of all research in this project.

**When to update**: Every skill appends an entry immediately after it writes its output file.
**How to update**: Append a new `---` separated entry. Never overwrite or remove existing entries.
**Format**: Each entry uses the template defined in that skill's "Results Summary Format" section.

The summary must always answer: stage, open gaps, current claim, and next action.

## 3. Required Skill File Shape

Every workflow skill should use this structure unless there is a strong reason not to:

1. frontmatter with name, trigger patterns, upstream dependencies, output location, and references to `karpathy-guidelines.md` and `general_guidelines.md`
2. one short role paragraph saying what the skill is for
3. `Hard requirements`
4. `Required Inputs (Ask If Missing)`
5. `What This Skill Does` (optional — include only when the skill's scope is easily confused with an adjacent skill, e.g., domain-generator vs. challenger; omit when the frontmatter description is sufficient)
6. `Workflow` with named phases
7. `Output: <path>`
8. `Failure Modes To Avoid`
9. `Invocation Patterns`
10. `Results Summary Format`

Do not leave the output format implicit. The artifact path and section structure should be visible in the skill file itself.

## 4. Invocation Checklist (Every Skill)

Before starting any work:
- [ ] Read `results/summary.md` if it exists — orient on prior work, don't repeat it.
- [ ] State your interpretation of the task in 1-2 sentences. If ambiguous, list interpretations and ask.
- [ ] State the success criterion: "I'll consider this done when X."
- [ ] Confirm the required inputs exist, or explicitly say what is missing and whether you will stop or proceed at lower confidence.

Before writing any output:
- [ ] Verify every non-trivial factual claim is cite-backed or marked `Unknown`.
- [ ] Run the Explain-It Test: can the core contribution be stated in two sentences without jargon?
- [ ] Ensure the output is grounded in the upstream artifacts for that stage rather than reconstructed from memory.
- [ ] Update `results/summary.md` with a new entry.

## 5. Input and Scope Discipline

- Every skill should tell the user what inputs are mandatory and what inputs are optional.
- Use the latest relevant artifacts under `results/` as the source of truth for downstream stages.
- If a required upstream artifact is missing, either stop and ask for it or proceed with an explicit lower-confidence warning.
- If the request names a specific section, artifact, or stage, touch only that scope.
- If you notice issues outside the requested scope, flag them explicitly. Never silently broaden the task.

## 6. Citation and Factuality Rules

- Every non-trivial claim cites a paper, official doc, benchmark spec, or primary source.
- If you cannot cite it, rephrase as an open question. Never state unverified claims as fact.
- If a detail is unknown, write `Unknown — needs verification` and explain what is needed to confirm.
- Do not hallucinate paper details.

## 7. Artifact-Grounded Execution

- Upstream artifacts are the default source of truth for downstream skills.
- Do not invent results, baselines, claims, citations, numbers, or missing files.
- If evidence is missing, mark it as missing, unsupported, placeholder, or unknown.
- Favor anchor artifacts such as:
  - gap-coverage map
  - claim-to-theory map
  - claim-to-experiment map
  - experiment-status map
  - claim-to-result audit
- Each skill should make one such anchor artifact or table central enough that later stages can reuse it.

## 8. Workflow Design Rules

- Use named phases in the workflow instead of an undifferentiated list.
- Early phases should load and interpret inputs before doing synthesis.
- Mid phases should build the anchor map, table, or audit that drives the rest of the skill.
- Final phases should convert that analysis into the final artifact and next action.
- If a skill only critiques without producing a usable downstream artifact, it is incomplete.

## 9. File Naming Conventions

| Artifact | Pattern |
|---|---|
| Lit review report | `results/literature-review/literature-review-<topic-slug>-YYYY-MM-DD.md` |
| Challenge report | `results/challenger/Challenge-<topic-slug>-YYYY-MM-DD.md` |
| Domain report | `results/domain/Domain-<topic-slug>-YYYY-MM-DD.md` |
| Experiment plan | `results/experiments/Experiment-Plan-<topic-slug>-YYYY-MM-DD.md` |
| Code manifest | `results/code/Codegen-<topic-slug>-YYYY-MM-DD.md` |
| Code root | `results/code/<topic-slug>/` |
| High-level paper flow | `results/paper/high-level-paper-<topic-slug>-YYYY-MM-DD.md` |
| Manuscript draft | `results/paper/manuscript-<topic-slug>-YYYY-MM-DD.tex` |
| Paper handoff note | `results/paper/paper-handoff-<topic-slug>-YYYY-MM-DD.md` |
| Review report | `results/review/Review-<topic-slug>-YYYY-MM-DD.md` |
| BibTeX (lit review) | `results/literature-review/references.bib` |
| BibTeX (paper) | `results/paper/references.bib` |
| JSON candidates | `results/literature-review/<source>-results.json` |
| Summary journal | `results/summary.md` |

Topic slug: lowercase, hyphens, no spaces, max 5 words. Example: `rag-eval-long-context`.

## 10. Output Contract

- Every skill must name one primary artifact path and one summary entry.
- The output artifact should be specific enough for the next stage to consume directly.
- The skill file should show the expected markdown or file structure for that artifact.
- Later stages should not need to guess what fields, sections, or tables the earlier stage produced.

## 11. Correctness Over Comprehensiveness

- A shorter, accurate output is better than a longer, padded one.
- Every sentence must carry information. Delete hedges, filler, and vague praise.
- Prefer concrete numbers over qualitative comparators.
- If you are uncertain, say so — don't obscure uncertainty with confident-sounding prose.

## 12. Failure-Mode Discipline

Every skill should contain a `Failure Modes To Avoid` section that names the most likely low-quality outputs for that stage.

These lists should be specific to the stage, but the shared default is:

- outputs disconnected from the actual upstream artifacts
- unsupported claims presented as facts
- over-broad scope changes
- generic filler instead of decision-useful structure
- critique without a usable downstream artifact
