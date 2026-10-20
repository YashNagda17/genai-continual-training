# PaperWriting Skills Orchestrator

Read this file first when a new agent starts work in this repository.

Its job is to give one compact view of:

- what this repo does
- what each skill is for
- when to invoke each skill
- what each skill depends on
- what each skill produces
- how skills compose with each other

This file is the orchestrator entrypoint.
It is not the place for Python CLI details.
For runnable scripts, always use [skills/scripts.md](/skills/scripts.md)

## Purpose

This repository is a staged research workflow that turns an initial idea into:

1. a literature-backed problem framing
2. a challenged and refined contribution
3. a theory-grounded claim
4. an experiment plan
5. runnable code
6. a paper flow
7. an Overleaf-ready draft
8. a reviewer-style critique and revision plan

The workflow is artifact-driven.
Later skills should read prior outputs from `results/` instead of re-deriving them from scratch.

## First Read

When a new agent starts here, it should read these in order:

1. `SKILLS.md` for repo-wide orchestration
2. `results/summary.md` if it exists
3. the specific skill file it is about to use
4. `skills/scripts.md` if helper scripts are relevant
5. `skills/general_guidelines.md` and `skills/karpathy-guidelines.md`

## Canonical Architecture

```text
User Idea
  -> Literature Review 1
  -> Challenger / Refinement
  -> Literature Review 2 (optional but often useful)
  -> Domain Generator
  -> Experiment Generator
  -> Code Generator
  -> Experiment Results (planned external stage)
  -> High-Level Paper Writer
  -> Overleaf Paper Writer
  -> Paper Reviewer
  -> Iterate on experiments / draft / positioning as needed
```

Current implemented workflow skills:

- [skills/literature-review.md](/skills/literature-review.md)
- [skills/challenger.md](/skills/challenger.md)
- [skills/domain-generator.md](/skills/domain-generator.md)
- [skills/experiment-generator.md](/skills/experiment-generator.md)
- [skills/code-generator.md](/skills/code-generator.md)
- [skills/high-level-paper-writer.md](/skills/high-level-paper-writer.md)
- [skills/overleaf-paper-writer.md](/skills/overleaf-paper-writer.md)
- [skills/paper-reviewer.md](/skills/paper-reviewer.md)

Supporting files:

- [skills/scripts.md](/skills/scripts.md)
- [skills/general_guidelines.md](/skills/general_guidelines.md)
- [skills/karpathy-guidelines.md](/skills/karpathy-guidelines.md)
- [README.md](/README.md)
- [ARCHITECTURE.md](/ARCHITECTURE.md)

## Artifact Model

All canonical outputs go under `results/`:

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

Core rule:

- every stage should produce a reusable artifact
- every stage should append a short entry to `results/summary.md`
- every downstream stage should prefer the latest artifact under `results/`

## Skill Map

### 1. Literature Review

File:
- [skills/literature-review.md](/skills/literature-review.md)

Use it when:
- the user asks for related work, prior work, recent papers, survey, state of the art
- novelty is still unclear
- a draft or idea needs a fresh targeted search

Triggers:
- `literature review`, `related work`, `prior work`
- `state of the art`, `survey`, `find papers`
- `recent papers`, `last 3 years`
- `review this paper`, `assess paper`

Primary job:
- find and analyze exactly 10 relevant works from the last 3 years
- produce a hard-nosed synthesis of gaps, overlap, and threats

Reads:
- user idea, topic, domain, draft section, or paper URL

Writes:
- `results/literature-review/literature-review-<topic>-YYYY-MM-DD.md`
- candidate JSON files
- optional BibTeX

Next likely skills:
- `challenger`
- `domain-generator`
- `paper-reviewer`

Notes:
- this is the main evidence-discovery stage
- it can be run more than once
- it is valid both early in idea formation and late in draft review

### 2. Challenger

File:
- [skills/challenger.md](/skills/challenger.md)

Use it when:
- the idea may be weak, broad, derivative, or under-specified
- the user asks whether something is novel, impactful, or publishable
- a lit review exists and the idea needs to be stress-tested

Triggers:
- `challenge my idea`, `stress-test`
- `is this novel`, `what's the impact`
- `refine my idea`, `is this publishable`

Primary job:
- destroy weak framing
- keep only defensible leverage
- forge a stronger refined contribution

Reads:
- latest literature review
- user idea statement
- venue targets

Writes:
- `results/challenger/Challenge-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `domain-generator`
- another `literature-review`
- `experiment-generator`

Notes:
- this is the adversarial filter
- it should not make assumptions silently
- any agent can use this skill if it needs a stronger research position

### 3. Domain Generator

File:
- [skills/domain-generator.md](/skills/domain-generator.md)

Use it when:
- the idea may violate domain theory or assumptions
- mechanism claims are vague
- you need theory-backed reformulation before designing experiments

Triggers:
- `ground this idea in theory`, `verify the assumptions`
- `map this idea to domain knowledge`
- `is this physically/statistically/theoretically valid`
- `correct wrong assumptions`
- `turn this vague idea into a theory-backed one`

Primary job:
- map claims to theory
- identify assumptions and failure regimes
- correct wrong domain framing

Reads:
- current idea
- latest challenger artifact when available
- latest literature review when available
- primary sources and standards

Writes:
- `results/domain/Domain-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `experiment-generator`
- `challenger` again if the core premise breaks

Notes:
- this is the validity stage, not the novelty stage
- use it when “should this work?” matters more than “has this been done?”

### 4. Experiment Generator

File:
- [skills/experiment-generator.md](/skills/experiment-generator.md)

Use it when:
- the claim is stable enough to test
- you need a reviewer-proof evaluation plan
- you want the smallest high-leverage experiment set

Triggers:
- `design experiments`, `generate experiment plan`
- `what experiments should I run`
- `high impact experiments`
- `reviewer-proof experiment plan`
- `turn this idea into experiments`

Primary job:
- turn theory-backed claims and reviewer risks into a prioritized experiment plan

Reads:
- latest domain report
- latest challenger report
- latest literature review

Writes:
- `results/experiments/Experiment-Plan-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `code-generator`
- `high-level-paper-writer` later once results exist

Notes:
- this is the evaluation-design stage
- if there is no claim-to-experiment map, this skill has not finished its job

### 5. Code Generator

File:
- [skills/code-generator.md](/skills/code-generator.md)

Use it when:
- the experiment plan is approved
- the team needs reproducible code, configs, outputs, and run layout

Triggers:
- `generate code`, `implement the experiments`
- `write runnable scripts`
- `create experiment code`
- `build the pipeline from the plan`

Primary job:
- implement the approved experiments with minimum clean structure

Reads:
- latest experiment plan
- optionally domain, challenger, and literature artifacts

Writes:
- `results/code/<topic-slug>/`
- `results/code/Codegen-<topic>-YYYY-MM-DD.md`

Next likely skills:
- external experiment execution
- `high-level-paper-writer` once outputs exist

Notes:
- this skill should not redesign the research question
- it should implement the agreed plan cleanly

### 6. High-Level Paper Writer

File:
- [skills/high-level-paper-writer.md](/skills/high-level-paper-writer.md)

Use it when:
- experiment outputs exist
- you need a paper-shaped story before writing LaTeX
- reviewer handoff should be grounded in actual current evidence

Triggers:
- `make a paper outline from results`
- `high level paper flow`
- `draft paper structure from experiments`
- `summarize the experiments and results into a paper outline`
- `prepare reviewer input from result csvs`

Primary job:
- map results and planned experiments into a paper structure
- separate supported claims from missing evidence

Reads:
- latest experiment plan
- current CSV / JSON / plot artifacts
- optional upstream reasoning artifacts

Writes:
- `results/paper/high-level-paper-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `overleaf-paper-writer`
- `paper-reviewer`

Notes:
- this is the paper-flow stage, not the final draft
- it should expose missing evidence explicitly

### 7. Overleaf Paper Writer

File:
- [skills/overleaf-paper-writer.md](/skills/overleaf-paper-writer.md)

Use it when:
- venue constraints are known
- the paper flow exists
- there is enough evidence to draft actual sections

Triggers:
- `write the paper in latex`
- `prepare overleaf draft`
- `write conference paper`
- `turn this into a latex paper`
- `draft overleaf manuscript`
- `write the submission draft`

Primary job:
- turn the paper flow and artifacts into real LaTeX
- mark missing citations, missing numbers, and unresolved placeholders honestly

Reads:
- venue guidelines
- latest high-level paper flow
- result artifacts
- optional reviewer notes

Writes:
- `results/paper/manuscript-<topic>-YYYY-MM-DD.tex`
- `results/paper/paper-handoff-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `paper-reviewer`
- another pass of itself after fixes

Notes:
- this is the drafting stage
- it should not invent evidence or citations

### 8. Paper Reviewer

File:
- [skills/paper-reviewer.md](/skills/paper-reviewer.md)
Use it when:
- a draft exists
- novelty overlap, experiment weakness, or impact bar is unclear
- you need a reviewer-grade rejection-risk analysis

Triggers:
- `review my paper`
- `check novelty of this draft`
- `find overlap with prior work`
- `what experiments are missing`
- `review this paper draft`
- `strengthen the results`
- `find related papers for the current draft`
- `is this high impact for the target conference`
- `how do we improve impact`

Primary job:
- audit the actual draft, not the intended idea
- connect claims to evidence
- identify missing literature, missing experiments, and weaker positioning

Reads:
- latest manuscript or high-level paper flow
- latest literature review
- challenger, domain, experiment, and code artifacts when available
- result tables and plots

Writes:
- `results/review/Review-<topic>-YYYY-MM-DD.md`

Next likely skills:
- `literature-review` for targeted overlap refresh
- `experiment-generator` for new must-run experiments
- `overleaf-paper-writer` for redrafting

Notes:
- this stage often triggers iteration backward
- that is expected, not a failure

## Support Skills

### Scripts

File:
- [skills/scripts.md](/skills/scripts.md)
Use it when:
- a helper script already exists
- you want to avoid spending model tokens on boilerplate
- you need scaffolds, artifact resolution, summary updates, search, ranking, BibTeX export, or PDF extraction

Triggers:
- `find the script for this`
- `use the existing scaffold`
- `what helper script already exists`
- `run the repo utility instead of writing new code`

Rule:
- all Python CLI details belong here, not duplicated across other markdown files

### Skill Conventions

File:
- [skills/general_guidelines.md](/skills/general_guidelines.md)
Use it when:
- adding a new skill
- updating skill structure
- checking output contracts, artifact naming, and summary behavior

Triggers:
- `create a new skill`
- `update this skill`
- `what is the skill file format`
- `check the output contract`
- `what naming convention should this skill use`

### Karpathy Guidelines

File:
- [skills/karpathy-guidelines.md](/skills/karpathy-guidelines.md)
Use it when:
- you need the repo-wide execution discipline
- you are about to overcomplicate, overreach, or assume missing context

Triggers:
- `what are the repo execution rules`
- `keep this scoped`
- `am I overcomplicating this`
- `what should I read before acting`

## Dependency Graph

Canonical dependency view:

```text
literature-review
  -> challenger
  -> domain-generator
  -> paper-reviewer

challenger
  -> domain-generator
  -> experiment-generator
  -> literature-review (second pass if needed)

domain-generator
  -> experiment-generator
  -> challenger (if core premise breaks)

experiment-generator
  -> code-generator
  -> high-level-paper-writer later after results exist

code-generator
  -> external execution/results
  -> high-level-paper-writer

high-level-paper-writer
  -> overleaf-paper-writer
  -> paper-reviewer

overleaf-paper-writer
  -> paper-reviewer
  -> overleaf-paper-writer again after fixes

paper-reviewer
  -> literature-review
  -> experiment-generator
  -> overleaf-paper-writer
```

## Interdependency Rules

Important rule:

- dependencies are artifact-based, not role-based

That means:

- any agent can invoke `challenger` if it needs adversarial critique
- any agent can invoke `literature-review` if overlap is unclear
- `paper-reviewer` can trigger another `literature-review`
- `domain-generator` can send the workflow back to `challenger`
- `designer`, `writer`, `reviewer`, or `implementation` agents can all use the same skills if the artifact dependency is satisfied

Examples:

- a drafting agent can use `challenger` before writing an introduction
- a reviewer agent can use `literature-review` to refresh recent overlap
- a coding agent can read `domain-generator` to understand which assumptions must be tested
- an outlining agent can use `paper-reviewer` output to restructure the paper flow

## When To Invoke Which Skill

Quick routing by user intent:

- “Find recent papers / prior work / related work” -> `literature-review`
- “Is this idea novel / impactful / publishable?” -> `challenger`
- “Does this mechanism make sense in the domain?” -> `domain-generator`
- “What experiments should we run?” -> `experiment-generator`
- “Implement the experiments” -> `code-generator`
- “Turn current results into a paper story” -> `high-level-paper-writer`
- “Write the actual LaTeX draft” -> `overleaf-paper-writer`
- “Review the current draft like a reviewer” -> `paper-reviewer`
- “Find an existing script or scaffold” -> `scripts`
- “Change the skill architecture itself” -> `skills/general_guidelines.md`

## Re-entry and Iteration

This repo is not strictly linear.

Common loops:

- `literature-review -> challenger -> literature-review`
- `challenger -> domain-generator -> challenger`
- `experiment-generator -> code-generator -> high-level-paper-writer -> paper-reviewer -> experiment-generator`
- `overleaf-paper-writer -> paper-reviewer -> overleaf-paper-writer`

Use loops when:

- novelty is still unclear
- theory breaks the current framing
- results do not support the draft claims
- reviewer objections expose missing baselines or missing comparisons

## New Agent Bootstrap Checklist

When invoked fresh in this repo:

1. read `SKILLS.md`
2. read `results/summary.md` if present
3. identify the current stage from existing artifacts
4. identify the user’s real request
5. choose the minimum skill or skill sequence that fits
6. read the chosen skill file
7. use `skills/scripts.md` for helper scripts instead of retyping boilerplate
8. write outputs under `results/`
9. append `results/summary.md`

## Current Planned Gaps

These stages are still architectural gaps in the repo:

- experiment execution / results aggregation workflow
- final LaTeX-draft preparation workflow

Until they exist as first-class skills:

- treat experiment execution as external to the current skill set
- feed resulting CSVs, JSON summaries, and plots into `high-level-paper-writer`

## Final Rule

If an agent only reads one file in this repository before choosing a skill, it should read `SKILLS.md`.
If it needs runnable commands or helper modules, it should then read `skills/scripts.md`.
