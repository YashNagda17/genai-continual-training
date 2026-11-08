## PaperWriting Skills

This repository stores reusable skills, shared instructions, and helper scripts for an AI-assisted industrial research workflow.

The research workflow is in [ARCHITECTURE.md](/ARCHITECTURE.md); the current implementation blueprint is [design/README.md](/design/README.md).

The active experiment is [Cost-Efficient Incremental Fine-Tuning for Multilingual Summarization](/Idea_Refined_2.md). It teaches `google/mt5-small` to summarize English news, then updates the same frozen base with LoRA so it can summarize Hindi without a second model. XL-Sum supplies the articles. Decisions use validation. Test scores are saved for the report. Only a scientific run is a result. Smoke and pilot runs are development checks.

## What the project does

One shared LoRA adapter, rank 16 and alpha 32, trains on a frozen mT5-small (300,176,768 parameters). English is seed 0. Hindi arms use seeds 1, 2, and 3. Each language uses 2,000 training rows, 200 validation rows, and 500 test rows. The English replay pool is 500 rows taken from the English training split and written once. Replay is 2 English rows per 16 Hindi rows. Decoding is greedy, up to 64 new tokens.

| Step | What happens |
|---|---|
| 0 | One English update, up to 20 summaries, truncation coverage, and tokens per character. A person still has to mark whether those summaries are readable. |
| 1 | Train the English checkpoint and write the pool. Train Hindi with no replay (X). Record English forgetting, Hindi gain, the script-guard share (G5), and the gate: continue, stop the replay story, or fix the setup. Time base, English, and Hindi-only summaries, merged and unmerged. |
| 2 | Replay English while learning Hindi. Compare random (A), loss (B), drift (G1), and past summaries (G2). Repair-and-stop (G3) continues from the parent with less English forgetting. |
| 3 | Map where English and Hindi gradients clash (C1) and write that prediction before any block run. Train late blocks and early-plus-late blocks (C2). Combine the best revision with the best blocks (D), and train a full-English reference. |
| 4 | Add new rank only where the maps clash (E1–E4), if the maps are stable and the new-parameter counts match within 3%. Otherwise record skipped or rejected and keep the combined arm. |
| 5 | Shrink the random arm and the overall winner by an energy rule and a generation test (F). Time the winner, the shrunk adapter, and the block arms. |
| 6 | Train a separate English checkpoint whose inputs start with `summarize in {language}:`, then run Hindi-only from it (G4). |
| 7 | Run drift together with past summaries. Run the frozen-room arm (E5) when Phase 4 completed. Record reverse order, a third language, and past summaries without references as not run. |

Each phase writes `report.json`, `report.csv`, and `report.md` under its directory, plus `final/<arm>/` so a later comparison can load that adapter. A finished phase report is reused. A finished arm is not retrained. An interrupted arm resumes from its checkpoint. Details are in [src/README.md](/src/README.md) and [the experiment contract](/design/summarization-experiments.md).

## Commands

Run these from the repository root after `source ~/venv/bin/activate`. Smoke needs no downloads. Pilot and scientific need XL-Sum and the mT5-small weights:

```sh
./download_datasets.sh xlsum
./download_model_weights.sh mt5
```

Smoke checks every phase on a tiny local model. It is not a result:

```sh
python -m src.summarization_experiment --protocol smoke --output results/code/summarization-smoke
```

Pilot runs Phase 0 on real XL-Sum and is the place to fix the number of English and Hindi passes. Scientific runs Phase 0 through Phase 7 on the GPU and refuses to start unless `preregistration.json` is already in the output directory. Use a separate directory for each protocol.

```sh
python -m src.summarization_experiment --protocol pilot --output results/code/summarization-pilot --device cuda:0
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0
```

`--phase N` runs one phase and reads the earlier reports from the same output directory. The same flag works for `smoke`, `pilot`, and `scientific`.

```sh
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 0
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 1
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 2
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 3
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 4
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 5
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 6
python -m src.summarization_experiment --protocol scientific --output results/code/summarization --device cuda:0 --phase 7
```

Phase 2 and later need the reports from the phases before them. A scientific run stops after Phase 1 when the gate is not `continue`.

## Tracking: what is left

| Item | State |
|---|---|
| Phases 0–7, resume, per-arm scores, timings, and saved adapters | Implemented. Checked on the smoke model only. |
| Pilot on real XL-Sum, including a human read of the 20 summaries | Not run. Phase 0 leaves `readability_review` as `pending`. The pilot must fix `english_passes`, `hindi_passes`, and `stability_threshold`. |
| `preregistration.json` in the scientific output directory | Required before the scientific command. The runner checks it and does not write it. It needs the date, locked settings, predictions P1–P10, winner rules, and gate thresholds from [the experiment contract](/design/summarization-experiments.md). |
| Scientific run on the GPU, seeds 1–3 | Not started. This is the only run that can be reported. |
| G1 compute-matched control | Not a phase. Tier 3 in the plan. |
| Blind rating sheets, two raters, and Cohen's kappa | Not produced. The kappa function exists. The sheets and the rater loop do not. |
| One report that marks P1–P10 and principles M, N, K, L, and P | Phase reports store the comparisons they compute. The single prediction and principle table is not assembled. |
| Reverse order, a third language, and past summaries without references | Recorded as skipped in Phase 7. |
| Qwen3-0.6B fallback | Not implemented. The plan uses it only if mT5 cannot carry the experiment. |

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
- `results/paper/`
- `results/review/`
- `results/summary.md`

There are still legacy sample artifacts under `resources/lit-review/`. Those are older examples, not the target architecture.

## Planned Additions

The summarization phases above are the active experiment. The remaining experiment work is in [Tracking: what is left](#tracking-what-is-left). The writing workflow still has no final LaTeX-draft preparation step.
