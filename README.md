## PaperWriting Skills

This repository stores reusable skills, shared instructions, and helper scripts for an AI-assisted industrial research workflow.

The research workflow is in [ARCHITECTURE.md](/ARCHITECTURE.md); the current implementation blueprint is [design/README.md](/design/README.md).

## Implemented Markdown Files

The repository currently contains these implemented workflow skill files under `skills/`:

- [skills/literature-review.md](/skills/literature-review.md): literature search and review workflow
- [skills/challenger.md](/skills/challenger.md): adversarial idea refinement from literature evidence
- [skills/domain-generator.md](/skills/domain-generator.md): theory-backed domain grounding and assumption correction
- [skills/experiment-generator.md](/skills/experiment-generator.md): high-impact experiment planning from grounded claims
- [skills/code-generator.md](/skills/code-generator.md): minimal reproducible code generation for approved experiments
- [skills/high-level-paper-writer.md](/skills/high-level-paper-writer.md): markdown-only paper flow from experiment plans and result artifacts
- [skills/overleaf-paper-writer.md](/skills/overleaf-paper-writer.md): conference-guideline-driven LaTeX paper drafting
- [skills/paper-reviewer.md](/skills/paper-reviewer.md): reviewer-style novelty, impact, and experiment-gap review

The repository also contains these implemented support markdown files and utilities:

- [skills/scripts.md](/skills/scripts.md): canonical script registry
- [skills/general_guidelines.md](/skills/general_guidelines.md): shared repo conventions
- [skills/karpathy-guidelines.md](/skills/karpathy-guidelines.md): execution rules for research tasks
- [src/litreview/](/src/litreview/): helper modules for search, merging, ranking, scaffolding, and PDF extraction

Current pipeline:

`Literature Review -> Challenger -> Domain Generator -> Experiment Generator -> Code Generator -> High-Level Paper Writer -> Overleaf Paper Writer -> Paper Reviewer`

## Output Convention

The repo’s shared conventions in [skills/general_guidelines.md](/skills/general_guidelines.md) treat `results/` as the canonical research-output root; executable code follows [code-structure.md](/code-structure.md):

- `results/literature-review/`
- `results/challenger/`
- `results/domain/`
- `results/experiments/`
- `results/code/`
- `results/paper/`
- `results/review/`
- `results/summary.md`

There are still legacy sample artifacts under `resources/lit-review/`. Those are older examples, not the target architecture.

## Planned Additions

To match the full architecture, the next missing first-class components are:

- experiment execution / results aggregation workflow
- final LaTeX-draft preparation workflow
