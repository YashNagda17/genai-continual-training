# Experiment entry point

Owns [summarization_experiment.py](/src/summarization_experiment.py) and [phases/](/src/phases/). The active experiment is [Idea_Refined_2.md](/Idea_Refined_2.md); [summarization-experiments.md](/design/summarization-experiments.md) owns its protocol.

The entry point resolves the protocol, validates scientific pre-registration before loading assets, prepares only selected XL-Sum rows, and executes phases in order under one writer lock. Pilot defaults to Phase 0. `--phase N` reads matching completed earlier reports and runs only N. Scientific execution stops after an unsuccessful X gate.

Phase modules own experiment decisions. Shared execution owns per-seed attempts, bounded run settings, model release, completed-arm reuse, and incomplete numerical resume. Reporting owns progress/status events, JSON/CSV/Markdown publication and atomic final adapter exports. Data preparation owns ID selection and prefix variants. Serving owns one timing pass without warm-up.

Changing settings or prepared-data identity refuses cached reuse. Completed phase reports publish after their artifacts and tables. Failure/interruption preserves prior recovery state and logs the error; it does not publish a completed report.

The former example, CSReL runner and combined-pipeline entry points have been removed. Historical paper mechanisms remain documented under [Papers/](/Papers/) and their original design references.
