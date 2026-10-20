---
name: literature-review
description: >
  Produce a brutally honest literature review for a user-provided research topic.
  Trigger on: "literature review", "related work", "prior work", "state of the art", "survey",
  "find papers", "recent papers", "last 3 years", "review this paper", "assess paper".
  Reports 10 relevant works from the last 3 years with relevance + shortcomings,
  written to results/literature-review/. Updates results/summary.md on completion.
  See skills/karpathy-guidelines.md and skills/general_guidelines.md for execution principles.
---

Hard requirements:

1. **Time window**: last 3 years from today. Older foundational work: label **Foundational (Older than 3 years)**, use sparingly.
2. **Minimum 10 works** for the initial pass; expand on request. Prefer peer-reviewed top venues and highly cited preprints.
3. **Be honest and brutal**: Actively look for reasons the user's idea is weak, derivative, or ill-posed.
4. **Five-paragraph summary per paper** (mandatory): Every included paper gets a prose summary with exactly five labeled paragraphs — Idea, Novelty, Results, Limitations, Relevance — integrated into the main report (see §Per-Paper Five-Paragraph Summary below).
5. **Output**: write the report and intermediate JSON under `results/literature-review/`. Update `results/summary.md`.

---

## Tiered Reading Protocol

Apply to every candidate paper. Stop at the tier that answers the question.

### Tier 1 — Abstract Scan (relevance gate, ~1 min)

Answer from title + abstract only:
- **Problem**: What specific problem does this paper claim to solve?
- **Method class**: What family of methods? (e.g., contrastive learning, RLHF, GNN, diffusion)
- **Benchmark/data**: What datasets or benchmarks are mentioned?
- **Claimed gain**: What improvement is asserted?
- **Relevance verdict**: ACCEPT / REJECT / MAYBE — one clause reason

### Tier 2 — Theory Scan (novelty assessment, ~5 min)

Read intro + methodology section. Use `litreview.pdf_text` if needed.

- **Contribution type**: `empirical` | `theoretical` | `architectural` | `dataset/benchmark` | `system` | `analysis`
- **Novelty decomposition** — new / adapted / unchanged for each:
  - Problem formulation
  - Objective / loss
  - Architecture / algorithm
  - Evaluation protocol
- **Key assumptions**: What must be true for the method to work?
- **Theoretical depth**: Proofs or bounds? Tight or vacuous?

### Tier 3 — Results Deep Read (impact assessment, ~10 min)

Read experiments + results tables + ablations + limitations. Use `litreview.pdf_images` for figures.

- **Baselines compared**: Which prior methods? Notable omissions?
- **Datasets**: name, size, domain, known biases
- **Metric**: Primary metric. Is it the standard for this task?
- **Magnitude**: Exact numbers. Marginal (<1%) or substantial (>3%)?
- **Statistical significance**: p-value / confidence interval / multiple seeds?
- **Ablation quality**: thorough / partial / missing
- **Failure modes**: Conditions where the method underperforms?
- **Reproducibility**: code (yes/no/partial) | data (yes/no/partial) | hyperparams (yes/no)
- **Citation velocity**: citations per year (Semantic Scholar `citation_count` ÷ age)

---

## Workflow

1. State your topic interpretation (1-2 sentences). If ambiguous, present 2-3 readings and ask — do not assume.
2. Form 6-10 targeted search queries (synonyms, methods, tasks, venues, include year tags).
3. Collect candidates with the search and merge scripts listed in `skills/scripts.md`. Write outputs into `results/literature-review/`.
4. Tier 1 screen: `litreview.abstract_screen ... --out results/literature-review/candidates-screened.json`. Manually resolve MAYBE cases. Target ~20 survivors.
5. Tier 2 rank: `litreview.rank_candidates ... --out results/literature-review/candidates-ranked.json`. Apply theory scan to top ~15. Re-rank if novelty is low.
6. Select best 10 (direct relevance, recency, methodological novelty, diversity of approaches).
7. Tier 3 deep read the 10. Use `litreview.pdf_text`, `litreview.pdf_images`, and AlphaXiv where useful.
8. Synthesize using comparison matrix and synthesis pointers below.
9. Scaffold: `litreview.scaffold_review_md "<topic>" --works-json results/literature-review/candidates-ranked.json --out results/literature-review/literature-review-<topic-slug>-YYYY-MM-DD.md`. Fill each section.
10. **Write the five-paragraph summary for every included paper** (Idea → Novelty → Results → Limitations → Relevance). This is the primary per-paper deliverable; tier notes may be used internally but must be converted into the five paragraphs in the final report.
11. Write the brutal refinement section.
12. **Update `results/summary.md`** — append a new entry (see §Results Summary Format below).
13. **Targeted supplementary search / novelty re-check** whenever the idea is being refined, before/after a `/challenger` or `/domain-generator` pass, or on request ("find more papers on X"). See §Targeted Supplementary Search below. Fold any new works into the SAME report using the five-paragraph format; do not spin up a separate lower-quality file.

---

## Targeted Supplementary Search / Novelty Re-Check (Mandatory on Refinement)

An initial 10–30 paper sweep goes stale fast in fast-moving areas. Before finalizing an idea (and every time the idea is refined), run a **targeted supplementary search** to catch near-duplicates and new SOTA that could preempt the contribution.

**When to run**
- The user asks to "find more papers on <method/competitor>", or names specific works (e.g., "CAST", "THESEUS").
- Immediately before or after a `/challenger` or `/domain-generator` pass.
- Whenever the idea's method or claims change (new mechanism, new baseline, new claim of a gain).

**How to run**
1. **Seed queries from the idea's own mechanism and named competitors**, not just the original topic. Query each core component and each claimed novelty in isolation (e.g., "activation-space LoRA transfer", "orthogonal Procrustes task-vector transport", "cross-model circuit transfer", "intruder dimensions LoRA subspace", "feature universality across models"). Include the specific competitor names the user mentions and search for *their* follow-ups and citers.
2. **Run a novelty-kill check**: for each core component of the idea, find the closest existing work and classify `already done` / `incremental` / `genuinely unaddressed`. If a component is `already done`, say so bluntly and reposition the contribution (this is the point of the re-check).
3. **Locate the current SOTA/backends**: identify methods the idea should *use as backends and baselines* rather than reinvent. Add them to the standard-baselines list.
4. **Add every relevant new work to the same literature-review file** using the mandatory five-paragraph format (Idea → Novelty → Results → Limitations → Relevance), under a clearly dated section header, e.g. `## Round N Additions (YYYY-MM-DD) — <focus>`. Do **not** downgrade supplementary papers to bullets.
5. **Update the header counts** (papers reviewed, search scope, "Last updated"), the **anchor papers**, and add a **short round-N synthesis** noting how the novelty landscape shifted and what the defensible core now is.
6. **Update `results/summary.md`** with a supplementary-search entry (new anchor papers, repositioned contribution, revised baseline list).

**Quality bar**: the re-check must end with an explicit, one-paragraph verdict: *what is now preempted, what remains genuinely open, and which new works are baselines/backends vs. boundaries.* Never quietly keep an outdated novelty claim.

---

## Per-Paper Five-Paragraph Summary (Mandatory)

Every paper in the final report must include a **five-paragraph prose summary** in the main literature-review markdown file. This replaces bullet-only or tier-only per-paper notes in the delivered artifact. Each parapragh must be detailed, so that next time agents accesses these papers, they can understand the paper without having to read the entire paper again.

Write `Unknown — needs verification` only when a specific fact is genuinely unavailable after Tier 3 reading.

### Required structure (per paper)

```
## [Year] Title (Venue)

**Citation**: Authors. "Title." Venue, Year. DOI/arXiv: XXXX
**Links**: [Paper](...) | [AlphaXiv](https://www.alphaxiv.org/abs/<arxiv-id>) | [Code](...) | [Data](...)

### Idea
One paragraph: what problem the paper addresses, the core hypothesis or mechanism, and how the method works at a high level. Name the setting (models, data, task) concretely.

### Novelty
One paragraph: what is genuinely new versus prior work. Decompose into problem formulation, method/algorithm, and evaluation. State what was adapted unchanged. Be skeptical — flag incremental moves.

### Results
One paragraph: main empirical or theoretical findings with numbers where available (benchmarks, metrics, effect sizes, significance). Name baselines and what was beaten or not beaten.

### Limitations
One paragraph: honest weaknesses — missing baselines, narrow evaluation, unsupported assumptions, failure modes, reproducibility gaps, and conditions where the method likely breaks.

### Relevance
One paragraph: explicit connection to the user's research idea. State whether it is a baseline, enabling tool, theoretical boundary, or direct competitor. End with the threat level to the user's contribution (none / low / moderate / high) and why.
```

### Quality bar

- Each paragraph is **4–8 sentences** of connected prose, not bullets.
- **Results** must include at least one concrete number or explicit `Unknown — needs verification`.
- **Relevance** must name the user's idea or pipeline stage (e.g., Stage 2 circuit discovery, blind-transfer baseline).
- When expanding beyond 10 papers, use the **same five-paragraph format** for every additional paper — do not downgrade extended papers to bullet summaries.

### Optional internal notes (not a substitute)

Agents may use the Tier 1–3 checklist below while reading, but the delivered report must contain the five paragraphs above for each paper.

---

## Per-Paper Review Template (Internal Reading Checklist)

Use while reading; convert findings into the five mandatory paragraphs above.

```
## [Year] Title (Venue)

**Citation**: Authors. "Title." Venue, Year. DOI/arXiv: XXXX
**Links**: [Paper](...) | [AlphaXiv](https://www.alphaxiv.org/abs/<arxiv-id>) | [Code](...) | [Data](...)

### Tier 1 — Abstract
- Problem:
- Method class:
- Claimed gain:
- Verdict: ACCEPT — (reason)

### Tier 2 — Theory
- Contribution type:
- Novelty: problem formulation (new/adapted/unchanged) | loss (new/adapted/unchanged) | architecture (new/adapted/unchanged) | eval protocol (new/adapted/unchanged)
- Key assumptions:
- Theoretical depth:

### Tier 3 — Results
- Baselines:
- Datasets: (name, size, domain)
- Metric + value vs. best baseline:
- Statistical significance:
- Ablation quality: (thorough / partial / missing)
- Failure modes:
- Reproducibility: code __ | data __ | hyperparams __
- Citation velocity: ~N/year

### Critical Assessment
- **Relevance**: (why this matters to the user's idea)
- **Genuine advance**: (what it actually contributes)
- **Shortcomings**: (dataset bias, missing baselines, eval leakage, scaling confounds, weak ablations, statistical issues, limited generalization)
- **Threat to user's idea**: (how it challenges, narrows, or invalidates it)
```

---

## Synthesis Pointers

### Comparison Matrix
Table: rows = papers, columns = `Method class | Benchmark | Primary metric | SOTA gain | Reproducible | Ablations`

### Consensus vs. Contested
- ≥7 papers agree → treat as established fact
- Papers contradict on results or methodology → flag as live debate

### Solved vs. Unsolved
- Solved: multiple independent papers with high performance using different methods → marginal gain unlikely publishable
- Unsolved: high variance across papers, or consistently noted as limitation → real gap

### Research Lifecycle Position
- `Nascent` — <5 papers, no consensus benchmark
- `Active` — competing methods, benchmark in flux
- `Maturing` — 1-2 dominant methods, standard benchmark established
- `Saturated` — incremental gains only, benchmark near ceiling

Saturated areas require a new benchmark, new task formulation, or new application setting to be publishable.

### Field Orientation: Problem-First vs. Solution-First
Before synthesizing gaps, classify how the field is organized:
- **Problem-first**: The community converges on a shared problem (benchmark, task, phenomenon). Methods compete. Gaps are performance ceilings, evaluation blind spots, or underexplored conditions.
- **Solution-first**: A new method or architecture has been proposed and the field is finding applications. Gaps are application domains not yet explored, failure modes not yet documented, or theoretical understanding not yet established.

This determines where new work should anchor. Problem-first fields reward closing a performance gap. Solution-first fields reward expanding the application frontier or explaining why the solution works.

### Condition Shift Analysis ("What Changed")
For gaps that have persisted >2 years without a solution, ask: what has changed recently that might make this gap now solvable?
- **Compute**: Models or methods that were too expensive in 2021 may now run on a single GPU.
- **Data**: New large-scale datasets or annotation tools that didn't exist when prior papers were written.
- **Benchmarks**: New evaluation protocols that expose failures the old benchmarks couldn't measure.
- **Regulation / deployment context**: New constraints (privacy, latency, edge deployment) that open sub-problems.
- **Adjacent breakthroughs**: A technique from another subfield (e.g., diffusion, RLHF, SSMs) that could be transferred.

Flag each persistent gap with: "Condition that may have changed: [X]."

**Adjacent Possible (Kauffman)**: For each changed condition, map what becomes newly feasible — experiments, architectures, tasks, or benchmarks that were impossible or premature *before* the change but are now within reach. List 2-3 adjacent possibilities per changed condition. These are the strongest positions for new work because they are defensible ("this couldn't have been done before") and first-mover.

### Evaluation Protocol Norms
Extract the community's standard experiment design before leaving the literature review. This directly constrains what experiments must be run.

- **Standard splits**: what train/val/test splits appear in ≥5 papers? Deviating requires explicit justification and a comparison on the old split.
- **Standard primary metric**: what does every paper in this area report as their headline number? Using a non-standard metric without justification is a fast rejection path.
- **Standard baselines**: which methods does every competitive paper compare against? A submission missing any of these will be rejected with "incomplete evaluation" before reviewers reach the method section.
- **Benchmark saturation**: for each benchmark, what is the current best score? If the leaderboard is at ≥90% on a 100% scale, the benchmark is likely saturated — remaining gains are within noise and uninterpretable. Flag this explicitly; it changes which benchmarks should be used.
- **Known confounds in this area**: what alternative explanations did prior papers raise for their own results (or ignore)? Collect these — they are the controls the user's experiments must include to preempt reviewer objections.

### Anchor Papers
Identify 1-2 papers with the highest citation velocity. The refined idea must differentiate from them.

---

## Results Summary Format

After every run, append an entry to `results/summary.md`. Create the file if it doesn't exist.

```markdown
---

## [YYYY-MM-DD] Literature Review — <Topic>

**Stage**: literature-review
**Lifecycle position**: Nascent / Active / Maturing / Saturated
**Field orientation**: Problem-first / Solution-first

**Files**:
- Report: [literature-review/<filename>.md](/results/literature-review/<filename>.md)
- Ranked candidates: [literature-review/candidates-ranked.json](/results/literature-review/candidates-ranked.json)
- BibTeX: [literature-review/references.bib](/results/literature-review/references.bib) _(if generated)_

**Key gaps** (from synthesis — what remains unsolved):
- Gap 1: ...
- Gap 2: ...
- Gap 3: ...

**Condition shifts** (what has changed that could reopen old gaps):
- ...

**Adjacent possibilities** (what is newly feasible per changed condition):
- ...

**Anchor papers** (highest citation velocity — any new work must differentiate):
- Paper A (N citations/year)
- Paper B (N citations/year)

**Next action**: run /challenger | deepen search on [sub-topic] | proceed to /high-level-paper-writer
```
