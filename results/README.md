# Results

Experiment results are in [report.md](/results/report.md), [report.csv](/results/report.csv), and [report.json](/results/report.json).

The five phases each have CSV, JSON, and Markdown reports under `results/phases/phase1/` through `phase5/`: forgetting, replay, block selection, compression, and distillation. Imported measurements are preserved under this numbering.

Experiment settings are in [report.json](report.json). Import provenance and source-file records are in [manifest.json](manifest.json).

Each local run saves predictions under `results/code/<run>/phase<N>/<arm>/predictions/`.
Open the English or Hindi JSON files to compare each example's `prediction` with its
reference `target`. Prediction JSON files are allowed by `.gitignore` for every run
and appear in Git changes when generated, ready to add and commit. Other local run
outputs remain ignored.

Each local experiment run saves its generated text in `results/code/<run-name>/phase<1-5>/<arm>/predictions/`.
Open `english-test-plain.json` or `hindi-test-plain.json` to inspect each article's ID, target summary, and model prediction. The corresponding `validation` and `guarded` files are in the same folder. Prediction JSON files are included by Git; model checkpoints and other run artifacts stay local.
To put a new run's predictions in a commit, use `git add results/code/<run-name>` from the repository root.
