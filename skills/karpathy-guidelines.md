---
name: karpathy-guidelines
description: >
  Execution principles for all research writing and literature review tasks.
  Adapted from Andrej Karpathy's four core principles for LLM-assisted work.
  Referenced by: literature-review, challenger, overleaf-paper-writer, paper-reviewer skills.
  Core problem: models make wrong assumptions and run with them, overcomplicate output,
  and change things that were not asked.
---

## 1. Think Before Acting

State your interpretation before doing any work.

- Before searching: restate the topic in your own words. If 2+ plausible interpretations exist, list them and ask which is intended — never silently pick one.
- Before writing: state which section you're drafting, what context files you've loaded, and what the contribution claim is.
- Before critiquing: state what you expect to find (your prior) so the user can correct it before you spend effort on the wrong framing.
- If a paper's abstract is ambiguous about its scope, say so rather than assuming the favorable interpretation.

## 2. Simplicity First

- Lit review asked for 10 papers → return exactly 10, not 15 "just to be safe".
- Section revision asked → revise that section, not the whole paper.
- Abstract screening asked → screen, don't also run ranking unless asked.
- No padding: every sentence in written output must carry information. Hedges like "This is an important problem" are deleted.
- If a 5-paper scan answers the question, don't run 10.

## 3. Surgical Changes

Edit only what the request names. Treat all other content as read-only.

- When revising Related Work, don't touch Introduction or Methodology.
- When adding a paper to the review, don't restructure the synthesis section.
- When fixing a citation, don't rewrite the surrounding paragraph.
- If you notice issues outside the scope of the request: flag them explicitly ("I also noticed X in §3 — want me to fix that?"). Never silently fix them.
- Match the existing writing style and terminology. Don't rename concepts or change notation without being asked.

## 4. Goal-Driven Execution

| Skill | Checkpoints |
|---|---|
| literature-review | candidates collected ✓ → Tier 1 screened ✓ → top 10 selected ✓ → Tier 3 done ✓ → synthesis written ✓ |
| overleaf-paper-writer | context loaded ✓ → draft written ✓ → placeholders marked ✓ → file written ✓ |
| paper-reviewer | draft loaded ✓ → overlap checked ✓ → experiment gaps identified ✓ → review written ✓ |
| challenger | lit review read ✓ → Gap-Coverage Map built ✓ → Destroy phase (novelty / impact / assumption / failure / confound audits) ✓ → Forge phase (TRIZ negation + cross-domain blends + contribution ladder) ✓ → Multi-Lens Validation (theorist / practitioner / skeptic / Janusian) ✓ → report written ✓ |

- A checkpoint must be verifiable.
- If a checkpoint fails (e.g., only 6 relevant papers exist), report the failure explicitly with the reason rather than silently substituting lower-quality papers.
- On ambiguous tasks, state the success criterion before starting: "I'll consider this done when X."
