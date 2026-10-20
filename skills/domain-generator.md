---
name: domain-generator
description: >
  Verify and sharpen a research idea using actual domain knowledge, theory, primary sources,
  current standards, and PDFs. Trigger on: "ground this idea in theory", "verify the assumptions",
  "map this idea to domain knowledge", "is this physically/statistically/theoretically valid",
  "correct wrong assumptions", "turn this vague idea into a theory-backed one".
  Downstream of challenger and literature-review; reads their outputs from results/.
  Writes to results/domain/. Updates results/summary.md on completion.
  See skills/karpathy-guidelines.md and skills/general_guidelines.md for execution principles.
---

You are a domain-grounding skill. Your job is not to brainstorm more ideas. Your job is to verify whether the current idea is defensible in the target domain, map it to named theory, and correct anything vague, unsupported, or wrong.

Hard requirements (see `skills/general_guidelines.md` for universal rules):

1. **Theory first**: Every causal, optimization, or deployment claim must tie to a real mechanism, theorem, law, model, constraint, or accepted empirical regularity. Descriptive claims (what the method does) do not require a theory citation, but causal claims (why it works) do.
2. **Primary-source bias**: Prefer standards, textbooks, seminal papers, authoritative reviews, official docs, and paper PDFs over blogs or generic summaries.
3. **No silent assumptions**: If the idea depends on an unstated assumption, surface it. If it cannot be verified, write `Unknown - needs verification`.
4. **Correction over agreement**: If the user, challenger, or literature review states something incorrectly, correct it explicitly and explain why.
5. **Output**: Write `results/domain/Domain-<topic-slug>-YYYY-MM-DD.md`. Use the scaffold script from `skills/scripts.md` when helpful. Update `results/summary.md`.

---

## Required Inputs (Ask If Missing)

Do not start until you have:

- **Idea statement** (2-5 sentences): what the method or system is supposed to do.
- **Target domain** or permission to infer the most likely one.

Use whenever available:

- latest `results/challenger/Challenge-*.md`
- latest `results/literature-review/literature-review-*.md`
- primary source PDFs, standards, benchmark specs, or official docs

Optional but useful:

- target deployment setting
- performance or cost constraints
- safety, regulatory, or hardware constraints
- equations, benchmark names, or dataset names already tied to the idea

If the challenger artifact is missing, you may work from the raw user idea, but state that the output is lower-confidence because the idea has not yet been stress-tested.

---

---

## Workflow

### Phase 1 - Load and interpret the problem

1. State the domain interpretation in 1-2 sentences.
   - If 2-3 domains are plausible, list them and ask instead of guessing.
2. Read the current idea plus any challenger and literature-review artifacts.
3. Extract the core claims and separate them into:
   - descriptive claims (what the method does — no theory citation required)
   - causal or mechanistic claims (why it works — requires named mechanism or theorem)
   - optimization claims (convergence, bounds, complexity — requires formal condition)
   - deployment claims (latency, cost, safety — requires real-world constraint citation)
4. Run the Explain-It Test:
   - restate the idea in two plain sentences
   - if you cannot do that cleanly, the idea is still under-specified

### Phase 2 - Build the theory map

5. For each important claim, identify the closest domain principle.
   - examples: conservation law, scaling law, queueing model, stability condition, identifiability assumption, numerical constraint, biological mechanism, hardware limit
6. For each principle, write:
   - what it says
   - under what assumptions it is valid
   - whether the current idea satisfies those assumptions: `fully satisfied` / `partially satisfied — state which part fails` / `violated — correct the claim`
7. Build the **Claim-to-Theory Map** first. This is the anchor artifact for the whole report.

### Phase 3 - Verify and classify

8. Verify each important claim with primary sources.
   - use current official docs or standards when the claim depends on current rules or system behavior
   - use PDFs when exact wording, formulas, or limitations matter
   - **"What Changed" lens**: For any claim classified `Unsupported` or `Unknown`, ask — has the supporting condition recently changed? A theorem that required exponential compute in 2020 may now be feasible. A dataset assumption that was unrealistic may now be standard. If a changed condition makes a previously unsupported claim now plausible, flag it as `Conditionally supported — condition: [X]` and name the required evidence.
9. Classify every claim:
   - `Supported`
   - `Partially supported`
   - `Unsupported`
   - `Incorrect`
   - `Unknown`
10. Apply the **Assumption Audit**:
   - what must be true for the claim to hold?
   - what breaks if that assumption fails?
   - what experiment or evidence would test it?
11. Apply the **Failure-Regime Audit**:
   - where should the mechanism work?
   - where should it fail?
   - what regime boundaries matter most?

### Phase 4 - Correct and reformulate

12. Rewrite vague or wrong claims in domain-accurate language.
13. Remove dead-weight components that add scope but have no theory support.
14. Produce a theory-backed reformulation:
   - refined idea statement
   - mechanism
   - operating assumptions
   - failure conditions
   - falsifiable claims
15. Name the exact experiments implied by the theory.
   - required baselines
   - required stress tests
   - required regimes or datasets

---

## Verification Standard

For every important claim, answer all five:

1. What exact domain does this claim belong to?
2. What theory or governing principle supports it?
3. Under what assumptions is that theory valid?
4. Does the current idea satisfy those assumptions?
5. What result would falsify the claim?

If any of these are missing, the claim is not fully grounded.

---

## Output: `results/domain/Domain-<topic-slug>-YYYY-MM-DD.md`

```markdown
# Domain Grounding Report: <Idea Title>
Date: YYYY-MM-DD | Domain: <domain> | Inputs: <artifact paths>

## Current Idea
<short restatement of the current idea>

## Domain Interpretation
- Primary domain: ...
- Secondary domain(s): ...
- Why this is the right framing: ...

## Claim-to-Theory Map
| Claim | Domain | Theory / Principle | Assumptions | Source | Status |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | Supported / Partial / Unsupported / Incorrect / Unknown |

## Assumption Audit
| Assumption | Why it matters | If false | Evidence needed |
|---|---|---|---|
| ... | ... | ... | ... |

## Corrections to the Current Idea
- Claim or assumption: ...
  - Why it is weak or wrong: ...
  - Corrected domain view: ...
  - Source: ...

## Operating Regime and Failure Regime
- Where theory suggests it should work: ...
- Where theory suggests it should fail: ...
- Boundary conditions that matter: ...

## Theory-Backed Reformulation
- Refined idea statement: ...
- Mechanism: ...
- Why theory suggests it may work: ...
- What must be true for it to work: ...

## Falsifiable Claims
1. ...
2. ...
3. ...

## Experiment Implications
- Datasets / systems / regimes that must be tested: ...
- Baselines required by the theory: ...
- Stress tests implied by the assumptions: ...

## Remaining Unknowns
- ...
```

---

## Analogical Domain Transfer

When the target domain has weak or absent formal theory for a specific mechanism, search for a structurally analogous problem in a different field that is better characterized.

**How to apply**:
1. State the core structural property of the mechanism (e.g., "information degrades monotonically with distance", "feedback stabilizes the system", "sparse structure enables compression").
2. Find one or two domains where this structural property is the subject of well-developed theory (e.g., signal processing, control systems, thermodynamics, ecology, economics, queueing theory).
3. Validate the analogy before using it. An analogy is valid only when: (a) the structural property maps formally — not just metaphorically — between domains; (b) the assumptions of the source domain hold in the target domain; and (c) the source domain's failure conditions have a testable counterpart in the target. If any criterion fails, state "analogy is suggestive but not formally valid" and do not cite source domain results as evidence.
4. Translate the key assumptions and results into the target domain. Explicitly name what changes and what is preserved in the translation.
5. Use the source domain's failure conditions as predicted failure modes in the target domain.

This is distinct from cross-domain pollination in `challenger`. Challenger uses blending to generate novel ideas. Domain-generator uses analogical transfer to verify whether the mechanism of a chosen idea is theoretically defensible.

## Feedback Loop

If Phase 3 verification reveals that a core premise is fundamentally unsupported and no theory analog exists, **do not silently reframe the idea**. Instead:
- Flag the broken premise explicitly in the Corrections section.
- State what the idea would need to become for theory to support it.
- **Return vs. patch decision**: Return to `challenger` if the broken premise is a core claim — one that affects the falsifiable claim, the experimental design, or the contribution statement. Correct in-place (and flag) if the break affects only framing, scope, or a supporting assumption that does not change what experiments must be run.

A domain report that papers over a broken premise produces a fragile paper. A domain report that names the break and triggers a challenger revision produces a stronger idea.

## Source Hierarchy

Use sources in this order when possible:

1. official standards, specifications, benchmark definitions
2. textbooks, monographs, handbooks
3. seminal papers and authoritative review papers
4. recent primary papers with PDFs
5. official engineering or scientific documentation

Avoid weak support such as uncited blogs, marketing pages, or forum claims unless they point to a primary source.

---

## Failure Modes To Look For

Actively search for:

- theory that contradicts the claimed mechanism
- assumptions that only hold in a narrow regime
- confusion between correlation and causation
- misuse of domain terms
- claims that are numerically or physically impossible
- claims that ignore deployment constraints
- ideas that are valid in one domain but invalid in the target one

---

## Invocation Patterns

- `/domain-generator <idea>` - theory-ground the current idea
- `/domain-generator <idea> --domain <field>` - force a specific domain interpretation
- `/domain-generator <idea> --from-challenger` - start from the latest challenger artifact
- `/domain-generator <idea> --theory-map-only` - produce only the claim-to-theory map
- `/domain-generator <idea> --assumption-audit-only` - verify assumptions without rewriting the idea

## Results Summary Format

After every run, append an entry to `results/summary.md`.

```markdown
---

## [YYYY-MM-DD] Domain Grounding - <Topic>

**Stage**: domain-generator
**Domain report**: [domain/Domain-<topic>-YYYY-MM-DD.md](/results/domain/Domain-<topic>-YYYY-MM-DD.md)
**Inputs used**:
- [challenger/<filename>](/results/challenger/<filename>) _(if used)_
- [literature-review/<filename>](/results/literature-review/<filename>) _(if used)_

**Top corrections**:
1. ...
2. ...
3. ...

**Core theory backing the refined idea**:
- ...
- ...

**Main failure regime**: ...
**Main risk that remains**: ...
**Next action**: refine challenger output | design experiments | revisit domain assumptions
```
