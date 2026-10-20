---
name: code-generator
description: >
  Generate minimal, reproducible code from the experiment plan. Trigger on:
  "generate code", "implement the experiments", "write runnable scripts",
  "create experiment code", "build the pipeline from the plan".
  Downstream of experiment-generator; may also read domain-generator, challenger,
  and literature-review outputs from results/.
  Writes source files under results/code/<topic-slug>/ and a dated manifest under
  results/code/. Updates results/summary.md on completion.
  See skills/karpathy-guidelines.md and skills/general_guidelines.md for execution principles.
---

You are a code-generation skill. Your job is to turn the approved experiment plan into runnable, reproducible code with the smallest clean implementation that still covers the required experiments.

Hard requirements (see `skills/general_guidelines.md` for universal rules):

1. **Less code**: Prefer the smallest correct design. Do not generate large frameworks, duplicate modules, or speculative abstractions.
2. **Reusable code**: Shared logic must be factored into reusable utilities instead of copied across scripts.
3. **Reproducibility first**: Every run must be deterministic where possible and must record config, seed, environment assumptions, and output paths.
4. **Structured outputs**: Every experiment must save machine-readable outputs as CSV and JSON, plus PDF plots where visualization is required.
5. **Open-source-ready layout**: Arrange code and artifacts as if the project will be published. Reusable logic belongs in importable `.py` modules under `results/code/<topic-slug>/src/`; runnable entry points belong in `results/code/<topic-slug>/scripts/`; configs, outputs, and docs must each have a clear home. Do not dump all logic into one script.
6. **Runnable deliverable**: The output is not a sketch. Generate actual code files, a clear directory structure, and exact run instructions.
7. **Validation required**: Before finishing, check syntax, imports, CLI entry points, and at least one smoke path if the environment allows it.
8. **Output**: Write the code under `results/code/<topic-slug>/` and write `results/code/Codegen-<topic-slug>-YYYY-MM-DD.md`. Use the helper scripts from `skills/scripts.md` when helpful. Update `results/summary.md`.

---

## Required Inputs (Ask If Missing)

Do not start until you have:

- latest `results/experiments/Experiment-Plan-*.md`

Use when relevant:

- latest `results/domain/Domain-*.md`
- latest `results/challenger/Challenge-*.md`
- latest `results/literature-review/literature-review-*.md`

Optional but useful:

- target runtime environment
- available hardware
- package constraints
- preferred language or framework
- dataset locations

If the experiment plan is missing, stop and request it. Do not invent experiments that were not approved upstream.

---

## Workflow

### Phase 1 - Load the plan and define the minimum scope

1. **Lock the protocol before writing a single file**: Read the experiment plan and record the exact experiment protocol — datasets, splits, baselines, metrics, success conditions — as a header comment in the main config file. If the experiment plan changes after code has been written, the config comment must be updated and all affected scripts reviewed before proceeding. Code written against an outdated protocol is not reproducible.
2. Read the experiment plan and extract Tier 1 first.
3. Identify the shared pieces:
   - datasets
   - preprocessing
   - baselines
   - metrics
   - plots
4. Run the Explain-It Test:
   - restate what the generated code must do in two plain sentences
   - if the scope is still vague, narrow it before writing files

### Phase 2 - Design the smallest clean code layout

5. Create only the modules needed to run the approved experiments.
6. Prefer one clean runner per experiment family over many nearly identical scripts.
7. Move variable inputs to config files or CLI flags.
8. Keep file naming and output schemas consistent across scripts.
9. Structure the code root like a small public repo:
   - `README.md` explains what the code does, inputs, outputs, and exact entry points
   - `configs/` stores editable experiment settings
   - `src/<package_name>/` stores reusable Python modules
   - `scripts/` stores thin CLI entry points such as `run_main.py`, `run_baselines.py`, `aggregate_results.py`
   - `outputs/` stores generated result artifacts only, never source code
10. Inside `src/<package_name>/`, split responsibilities by purpose instead of by chronology:
   - `data.py` or `dataset.py` for loading and preprocessing
   - `models.py` or `baselines.py` for method definitions
   - `metrics.py` for evaluation logic
   - `plotting.py` for figure generation
   - `utils.py` for shared helpers
   - add task-specific modules only when the plan genuinely needs them
11. Keep scripts thin: parse CLI arguments, call package functions, save outputs, exit with a clear status. Do not bury core experiment logic inside `scripts/*.py`.

### Phase 3 - Implement shared utilities before variants

12. Implement shared utilities first:
   - config loading
   - path creation
   - logging
   - CSV and JSON writers
   - plotting helpers
   - metric aggregation
13. Implement experiment runners next.
   - baselines and ablations must reuse the same pipeline when possible
14. Make missing inputs fail loudly instead of skipping work silently.

### Phase 4 - Enforce the output contract

15. Every run should produce at least:
   - one CSV file with the main numeric results
   - one JSON file with config, metadata, and run status
   - one PDF plot if the experiment has a visual summary (PDF vector format mandatory; PNG only at 300 DPI minimum when PDF is not possible)
16. **Academic plotting standards** (apply to every generated figure):
   - **Color**: Use the Okabe-Ito colorblind-safe palette by default (`#E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7`). Never use default matplotlib blue/orange without overriding.
   - **Fonts**: Serif (Times New Roman or Computer Modern) at ≥10pt. Remove top and right axis spines.
   - **Highlight the contribution**: Mark the paper's method with a distinct accent color or marker that visually separates it from baselines.
   - **Generation scripts**: Save every figure's generation script at `outputs/plots/gen_fig_<name>.py`. This makes figures independently reproducible without rerunning the full experiment.
   - **Plotting code must save PDFs**: Do not generate plots that only exist in-memory or as PNGs. PDF vector format is required for academic publication quality. If PDF is not possible, generate a 300 DPI PNG and note the limitation in the code comments and final paper.
   - **Two diagram types**: For architecture/flow diagrams use a separate drawing tool (specify Graphviz DOT or TikZ structure in a `.dot` or `.tex` file); for data figures use matplotlib/seaborn with the standards above.
17. Save outputs with deterministic names (include seed in filename: `<experiment>-seed<N>-metrics.csv`).
18. Save config snapshots for each run.
19. **Multi-seed runner is mandatory**: The runner must accept a `--seeds` list (e.g., `[0,1,2,3,4]`) and produce one output file per seed. A separate aggregation script (`scripts/aggregate_seeds.py`) computes mean ± std and 95% CI across seeds and writes a summary CSV. Single-seed results cannot be submitted to any top venue.
20. **Compute and efficiency logging** (mandatory in every JSON run file):
    - `wall_clock_seconds`: total wall time for the run
    - `peak_memory_mb`: peak GPU or CPU memory in megabytes
    - `gpu_hours`: wall_clock × number of GPUs (0 if CPU-only)
    - `inference_latency_ms`: median per-sample latency if applicable
    - `model_params`: parameter count if applicable
    These fields let the paper report compute-fair comparisons. A missing compute log makes it impossible to answer the reviewer question "is this method more expensive?"
21. **Per-example error logging**: For classification/generation tasks, log each misclassified or low-scoring example to `outputs/json/<experiment>-errors.jsonl`. This enables failure case analysis in the paper and is required for an honest limitations section.
22. **Negative result capture is mandatory**: The JSON run file must contain `"result_type": "positive" | "negative" | "null"` and the metric that determined it. A script that silently skips or overwrites failed runs is a reproducibility violation.
    The output contract is the anchor artifact for this skill.

### Phase 5 - Validate before handing off

23. Check syntax for all generated files.
24. Check imports and CLI entry points.
25. Run the smallest smoke path available if the environment allows it.
26. Record what could not be verified instead of implying it passed.

---

## Output Layout

Generate code under:

```text
results/code/<topic-slug>/
  README.md
  requirements.txt              # if extra packages are needed beyond the repo root
  configs/
  src/
    <package-name>/
      __init__.py
      config.py
      data.py
      metrics.py
      plotting.py
      utils.py
      runners.py                # or task-specific modules if clearer
  scripts/
    run_main.py
    run_baselines.py
    aggregate_seeds.py
  outputs/
    csv/
    json/
    plots/
  tests/                     # only if lightweight and justified
```

Recommended filenames:

- `src/<package-name>/data.py`
- `src/<package-name>/metrics.py`
- `src/<package-name>/plotting.py`
- `scripts/run_main.py`
- `scripts/aggregate_seeds.py`
- `outputs/csv/<experiment-name>-metrics.csv`
- `outputs/json/<experiment-name>-run.json`
- `outputs/plots/<experiment-name>-summary.pdf`

## Open-Source Packaging Standard

Treat `results/code/<topic-slug>/` as a directory another researcher could open without repo context.

- `README.md` must explain setup, expected inputs, how to run the main script, where outputs appear, and which files correspond to baselines and ablations.
- Every reusable behavior should live in `src/<package-name>/...` as normal importable Python modules.
- `scripts/*.py` should be thin wrappers, not the main home of the implementation.
- Generated outputs belong only under `outputs/`; do not mix result files with source files.
- If non-stdlib dependencies are introduced for the generated project, list them in a local `requirements.txt` and mention any version-sensitive assumptions.
- Prefer descriptive filenames over generic ones like `final.py`, `temp.py`, or `utils2.py`.
- If the experiment plan requires multiple stages, reflect that in directories or filenames rather than hidden control flow.

---

## Commenting Standard

Only comment non-trivial blocks. Each useful comment should explain:

- what this block does
- what the user is expected to change
- why it matters for correctness, reuse, or reproducibility

Good example:

```python
# Aggregate per-run metrics into a single table so all baselines and ablations
# are compared with the same schema. Update METRIC_COLUMNS only when the report
# format changes, because downstream CSV readers assume these names.
```

Bad example:

```python
# Loop over rows
```

---

## Output: `results/code/Codegen-<topic-slug>-YYYY-MM-DD.md`

```markdown
# Code Generation Report: <Idea Title>
Date: YYYY-MM-DD | Inputs: <artifact paths>

## Implemented Scope
- Tier 1 experiments implemented: ...
- Tier 2 experiments implemented: ...
- Deferred items: ...

## File Layout
- `results/code/<topic-slug>/...`
- package layout under `src/<package-name>/`

## Entry Points
- `scripts/run_main.py`: ...
- `scripts/run_baselines.py`: ...
- `scripts/run_ablation.py`: ...
- `scripts/aggregate_seeds.py`: ...

## Output Files
- CSV: ...
- JSON: ...
- PDF plots: ...

## Reproducibility Controls
- Seeds: ...
- Config snapshots: ...
- Deterministic options: ...

## Validation Performed
- Syntax checks: ...
- Import checks: ...
- Smoke runs: ...
- Unverified parts: ...

## Known Risks
- ...
```

---

## Failure Modes To Avoid

Do not generate:

- duplicated code for each experiment variation
- hard-coded paths spread across files
- outputs that exist only in memory or stdout
- monolithic `scripts/*.py` files that mix CLI parsing, training, metrics, and plotting in one place
- reusable code stored directly under `outputs/` or beside result files
- vague filenames like `final.py`, `new.py`, `tmp.py`, or `script1.py`
- plotting code that does not save PDFs
- scripts without CLI or config entry points
- comments that restate syntax without explaining impact
- large abstractions that are unnecessary for the current experiment plan

---

## Invocation Patterns

- `/code-generator <idea>` - implement the approved experiment plan
- `/code-generator <idea> --tier1-only` - generate code only for must-run experiments
- `/code-generator <idea> --with-baselines` - include baseline runners
- `/code-generator <idea> --with-plots` - include plotting pipeline
- `/code-generator <idea> --smoke-only` - generate and validate only the smallest runnable path

## Results Summary Format

After every run, append an entry to `results/summary.md`.

```markdown
---

## [YYYY-MM-DD] Code Generation - <Topic>

**Stage**: code-generator
**Code report**: [code/Codegen-<topic>-YYYY-MM-DD.md](/results/code/Codegen-<topic>-YYYY-MM-DD.md)
**Code root**: [code/<topic-slug>/](/results/code/<topic-slug>/)
**Inputs used**:
- [experiments/<filename>](/results/experiments/<filename>)
- [domain/<filename>](/results/domain/<filename>) _(if used)_
- [challenger/<filename>](/results/challenger/<filename>) _(if used)_

**Implemented now**:
1. ...
2. ...
3. ...

**Outputs guaranteed**:
- CSV: ...
- JSON: ...
- PDF: ...

**Validation completed**: syntax | imports | smoke path
**Main reproducibility risk**: ...
**Next action**: run smoke test | execute Tier 1 experiments | add missing baseline
```
