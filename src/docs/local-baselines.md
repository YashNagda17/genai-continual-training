# Fresh local baseline pilot

`src.baseline_experiment` can prepare its own English source with `--from-scratch`.
It reads the locally cached XL-Sum data, selects rows with a seeded streaming
reservoir, trains an English adapter, and writes a replay pool. It then runs
CSReL, its random-memory control, ordinary random replay, and TSR. It does not
invoke the main experiment phase handlers or require a previous Drive run.

For the 4 GB laptop GPU, run this from PowerShell in the repository root:

```powershell
python -u -m src.baseline_experiment --from-scratch --source results/code/mbart-local-pilot-source-01 --output results/code/mbart-local-pilot-01 --epochs 2 --seed 1 --device cuda:0 --teacher-device cpu --batch-size 1 --analysis-batch-size 1 --replay-per-batch 1 --train-examples 100 --validation-examples 10 --test-examples 25 --pool-examples 25 --max-source-length 128 --max-target-length 32 --max-new-tokens 64 --gradient-checkpointing
```

Counts are per language; the replay pool is drawn from the English training
rows. English source training and each Hindi arm use two epochs. TSR adds replay
updates and CPU teacher inference. This is a smaller exploratory pilot: its
data, token lengths, and replay ratio differ from the recorded full experiment.
Runtime must be measured on the local host; completion within two hours is not
guaranteed.

Use fresh source and output directories. Once source preparation completes,
`source.json` records portable checkpoint and pool paths. To reuse that source,
omit `--from-scratch` and choose a new output directory. Data-size and token-limit
flags apply only when preparing a fresh source; reused sources retain their
prepared views and lengths. Batch, replay, and generation settings still apply.
Do not change the running checkout while training.

Reports and adapters are saved under the output directory. The source directory
contains `prepared/`, `english/`, `preparation.json`, and `source.json`; it does not
contain experiment phase reports. Preparation status is saved separately from
baseline results so a failed preparation is not mistaken for a complete source.
