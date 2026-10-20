# Project Architecture

This project is a staged research system that turns an initial idea into a final paper draft.

## End-to-End Workflow

| Stage | Input | Output | Repo support |
|---|---|---|---|
| 1. Initial Idea Intake | user idea | problem statement, first search terms | planned |
| 2. Literature Review 1 | initial idea | prior work, gaps, open questions | [skills/literature-review.md](/skills/literature-review.md) |
| 3. Challenge and Refine | idea + review | refined idea, objections, reviewer risks | [skills/challenger.md](/skills/challenger.md) |
| 4. Literature Review 2 | refined idea | tighter novelty map | [skills/literature-review.md](/skills/literature-review.md) |
| 5. Domain Grounding | refined idea + review + challenger | theory-backed idea, corrected assumptions, falsifiable claims | [skills/domain-generator.md](/skills/domain-generator.md) 
| 6. Experiment Generation | domain report + challenger + literature review | prioritized experiment plan | [skills/experiment-generator.md](/skills/experiment-generator.md)
| 7. Code Generation | experiment plan | runnable code, configs, output contract | [skills/code-generator.md](/skills/code-generator.md)
| 8. Experiment Execution and Results | code + data + compute | runs, tables, plots, result summaries | planned |
| 9. High-Level Paper Flow | experiment plan + result artifacts | markdown paper flow, experiment status map, reviewer input | [skills/high-level-paper-writer.md](/skills/high-level-paper-writer.md)
| 10. Conference Paper Drafting | high-level paper flow + venue guidelines + results | LaTeX draft and handoff notes | [skills/overleaf-paper-writer.md](/skills/overleaf-paper-writer.md)
| 11. Paper Review and Final Draft | draft + reviews + updated lit | reviewer notes, added experiments, final LaTeX draft | [skills/paper-reviewer.md](/skills/paper-reviewer.md)

## Architecture Flow

```text
User Idea
  -> Literature Review 1
  -> Challenger / Refinement
  -> Literature Review 2
  -> Domain Generator
  -> Experiment Generator
  -> Code Generator
  -> Experiment Results
  -> High-Level Paper Writer
  -> Overleaf Paper Writer
  -> Paper Reviewer
  -> Final LaTeX Draft
```

## Implemented Markdown Files

Workflow skills:
- [skills/literature-review.md](/skills/literature-review.md)
- [skills/challenger.md](/skills/challenger.md)
- [skills/domain-generator.md](/skills/domain-generator.md)
- [skills/experiment-generator.md](/skills/experiment-generator.md)
- [skills/code-generator.md](/skills/code-generator.md)
- [skills/high-level-paper-writer.md](/skills/high-level-paper-writer.md)
- [skills/overleaf-paper-writer.md](/skills/overleaf-paper-writer.md)
- [skills/paper-reviewer.md](/skills/paper-reviewer.md)

Support markdown files and utilities:
- [skills/scripts.md](/skills/scripts.md)
- [src/litreview/](/src/litreview/)
- [skills/general_guidelines.md](/skills/general_guidelines.md): output, naming, and summary conventions
- [skills/karpathy-guidelines.md](/skills/karpathy-guidelines.md): execution discipline across skills
- [README.md](/README.md): top-level overview of implemented files and pipeline
- [ARCHITECTURE.md](/ARCHITECTURE.md): stage-by-stage architecture and artifact flow

Planned additions:
- experiment execution / results workflow
- final-latex-draft workflow

## Artifact Flow

Target artifact model:

```text
results/
  summary.md
  literature-review/
  challenger/
  domain/
  experiments/
  code/
  paper/
  review/
```

- `results/summary.md` is the running state of the whole project
- each stage writes dated artifacts into its own subdirectory
- later stages should read earlier-stage artifacts instead of re-deriving them from scratch

## Current Gap Between Target and Repo

- the intended architecture uses `results/`
- some existing scripts and sample artifacts still reference `resources/lit-review/`
Maintain the architecture against the `results/` model; keep `resources/` only as legacy examples until the migration is complete.
