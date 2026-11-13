# Multilingual summarization experiments

mBART learns English summarization, then Hindi, with LoRA and English replay on XL-Sum.

## Main command to run

After completing the setup and downloads below, run this **from the repository root** (the folder containing `requirements.txt`):

```sh
python -u -m src.summarization_experiment --device cuda:0 --gradient-checkpointing --output results/code/summarization-run-01
```

This starts the scientific experiment and runs **Phase 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7** automatically. You do not need to run each phase separately. The scientific run continues beyond Phase 1 only if its gate is `continue`; some later comparisons can be skipped when their conditions are not met.

Use a **fresh output folder for every run**, for example `summarization-run-02` next time. The runner refuses folders containing previous run artifacts and does not resume interrupted runs. `-u` shows Python output without buffering. Gradient checkpointing reduces activation memory use at the cost of extra computation.

## Step-by-step setup

### 1. Download the code

Install Git and Python 3.11 or newer, then open a terminal and run:

```sh
git clone https://github.com/YashNagda17/genai-continual-training.git
cd genai-continual-training
```

Alternatively, open the [GitHub repository](https://github.com/YashNagda17/genai-continual-training), select **Code -> Download ZIP**, extract it, and open a terminal inside the extracted folder. If you already have the code, open its folder and continue with Step 2.

### 2. Create and activate a Python environment

Choose the commands for your terminal.

**Windows Command Prompt (CMD):**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed but you have a newer supported Python, use `python -m venv .venv` instead. If PowerShell blocks activation, use the CMD commands in a Command Prompt terminal.

**Linux / WSL / cloud terminal:**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Reactivate this environment whenever you open a new terminal. Run all remaining commands from the repository root with this environment active.

### 3. Install dependencies and check the GPU

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the NVIDIA GPU commands in this guide, verify that PyTorch can use CUDA:

```sh
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable; check your PyTorch build and NVIDIA driver'; print('GPU:', torch.cuda.get_device_name(0))"
```

If CUDA is unavailable, install the appropriate CUDA-enabled PyTorch build using the [official PyTorch installer](https://pytorch.org/get-started/locally/) and repeat the check. `--device xpu` is available for supported Intel GPUs; omitting `--device` enables automatic device detection. Use the smoke check below for a small CPU run.

### 4. Download the dataset and model weights

These Python commands work in **Windows CMD, PowerShell, Linux, and WSL**. They download the English/Hindi XL-Sum splits and mBART summarizer into the locations used by the runner:

```sh
python -m src.datasets.dataset_sources xlsum src/datasets/data
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('ctu-aic/mbart25-multilingual-summarization-multilarge-cs', revision='main', cache_dir='src/models/model_weights', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'pytorch_model.bin']))"
```

Allow both downloads to finish before training. Keep enough free disk space for the model, dataset, and run artifacts.

On Linux / WSL, you can use the helper scripts **instead of** those two Python download commands:

```sh
bash download_datasets.sh xlsum
bash download_model_weights.sh mbart
```

### 5. Check that the project runs

**Smoke check without model or dataset downloads** (requires the dependencies from Step 3):

```sh
python -u -m src.summarization_experiment --protocol smoke --output results/code/summarization-smoke-01
```

This uses a tiny random model on CPU by default and saves artifacts in the specified folder. It checks execution, not scientific quality. You can run it before Step 4.

To run the smoke check on NVIDIA CUDA instead, choose another fresh output folder:

```sh
python -u -m src.summarization_experiment --protocol smoke --device cuda:0 --output results/code/summarization-cuda-smoke-01
```

Smoke runs save their reports in the output folder. For a real-model development run after downloading the model and dataset, use the pilot in Step 6.

### 6. Run the experiment

**Full scientific run:**

```sh
python -u -m src.summarization_experiment --device cuda:0 --gradient-checkpointing --output results/code/summarization-run-01
```

The scientific defaults are four training passes per language, 2,000 training rows, 200 validation rows, and 500 test rows per language. The English replay pool contains 500 training rows. Replay uses two English rows per 16 Hindi rows. Decoding is greedy, with at most 256 new tokens. Every arm runs once with fixed shared randomness; the runner does not perform seed sweeps.

**Pilot with saved outputs**, using the standard dataset sizes and one epoch:

```sh
python -u -m src.summarization_experiment --pilot --epochs 1 --device cuda:0 --gradient-checkpointing --output results/code/summarization-pilot-01
```

The pilot runs all seven phase jobs without the scientific Phase 1 stop gate; later conditional comparisons still use their eligibility rules. `--pilot` alone keeps the default dataset sizes and four passes. `--epochs 1` reduces training to one pass per language; it does not reduce the dataset sizes.

For five passes on the default dataset sizes, use another fresh folder:

```sh
python -u -m src.summarization_experiment --pilot --epochs 5 --device cuda:0 --gradient-checkpointing --output results/code/summarization-pilot-5epochs
```

Nondefault epoch counts require a pilot or smoke run. The same pass count applies to English and Hindi training stages throughout that invocation.

## Phases in execution order

One shared LoRA adapter, rank 16 and alpha 32, trains on the frozen mBART summarizer. Each comparison below is an experiment arm. Validation decides gates, winners, and shrink ranks; test scores are report-only.

| Order | Phase | What happens |
|---|---|---|
| 1 | English training and Hindi-only baseline | Train the English checkpoint and save the replay pool. Learn Hindi without replay (X), measure English forgetting, Hindi gain, and script-guard share (G5). Record the gate: continue, stop the replay story, or fix the setup. Time base, English, and Hindi-only summaries, merged and unmerged. |
| 2 | English replay comparisons | Learn Hindi while replaying English. Compare random (A), loss (B), drift (G1), and past-summary replay (G2). Run repair-and-stop (G3) with the selector from the loss/drift parent that has less English forgetting. |
| 3 | Gradient maps, block placement, and combination | Map where English/Hindi gradients clash (C1) and save the prediction before block runs. Compare late and early-plus-late blocks (C2), combine the best revision with the best blocks (D), and train a full-English replay reference. Run the compute-matched random control when drift wins or conditional comparisons are forced. |
| 4 | Extra adapter capacity | Test new adapter room (E1-E4) if gradient maps are stable and new-parameter counts match within 3%. Otherwise record skipped or rejected and keep the combined arm. |
| 5 | Adapter shrinking and serving | Shrink the random arm and overall winner using an energy rule and a generation test (F). Time the winner, shrunk adapter, and block arms. |
| 6 | Language-prefix comparison | When validation script-recovery share meets the threshold, train a separate English checkpoint with `summarize in {language}:` inputs, then learn Hindi without replay (G4). Otherwise record skipped. |
| 7 | Cached teacher knowledge distillation | Compare matched reference replay against cached English-teacher KD on mBART. Report English/Hindi validation chrF and ROUGE-L differences. |

The scientific runner stops after Phase 1 when its gate is not `continue`. Read `run.json` and `phase1/report.md` in your output folder for the outcome. There is no separate phase-selection command; each invocation starts at Phase 1 and follows the order above.

## Where to find your outputs

For `--output results/code/summarization-run-01`, outputs are organized as follows. Later phase folders appear only if reached:

```text
results/code/summarization-run-01/
  prepared/                 Prepared real-data views
  phase1/                   English checkpoint, pool, Hindi-only arm, reports
  phase2/                   Replay arms and reports
  phase3/                   Maps, block/combined arms, reports
  phase4/                   Extra-room results or skip/rejection reports
  phase5/                   Shrinking and serving reports
  phase6/                   Language-prefix results or skip report
  phase7/                   KD comparison and reports
    kd/report.md            Detailed paired KD report
  run.json                  Final status and completed phase numbers
```

Each phase writes `report.md`, `report.csv`, and `report.json`. Each trained arm saves one final adapter, predictions, metrics, and measured costs; reports point directly to that adapter. Saved progress and completed reports never skip work on a new invocation.

## Separate CSReL / TSR baseline comparisons

After a real mBART run, run these additional comparisons independently of Phases 1-7:

```sh
python -u -m src.baseline_experiment --source results/code/summarization-run-01 --output results/code/mbart-baselines-01 --epochs 1 --device cuda:0 --gradient-checkpointing
```

`--source` must contain prepared data, its English adapter, and replay pool. Imported reports included in this repository do not contain these training inputs. Keep baseline output separate from the source run.

This trains each baseline for one epoch and reports CSReL, its matched random-memory control, TSR, and ordinary random replay. Reports and final adapters are saved in the baseline output folder. These are summarization adaptations, not exact paper reproductions.

## Recorded results

[Experiment results](results/report.md) are available as [CSV](results/report.csv) and [JSON](results/report.json), with reports for Phases 1-7 under `results/phases/`. These recorded results are separate from new runs you launch with the commands above.

The recorded scientific run passed the Phase 1 gate. Hindi-only, random replay, and the combined winner have results for seeds 1-3. New adapter room and language-prefix training were skipped; KD did not meet the selection criterion. The current runner runs each arm once without automatic seed sweeps.

## Tracking: what is left

| Item | State |
|---|---|
| Phases 1-7, fresh runs, per-arm scores, timings, and saved adapters | Implemented; recorded CUDA results are linked above. |
| Scientific run on the GPU, seeds 1-3 | CUDA results recorded for Hindi-only, random, and combined; see [results and phase outcomes](results/report.md). |
| CSReL and TSR baseline experiments | Implemented in the separate baseline runner; measurements pending. |
| G1 compute-matched control | Conditional comparison implemented in Phase 3. |
| Blind rating sheets, two raters, and Cohen's kappa | Sheets and the rater loop are not produced; the kappa function exists. |
| One report that marks P1-P10 and principles M, N, K, L, and P | Phase reports store their comparisons; the single prediction and principle table is not assembled. |
| Phase 7 | Paired cached mBART teacher KD versus matched replay, with validation chrF/ROUGE-L reports. |

## Project layout and research workflow

```text
results/                    Recorded results and local experiment outputs
src/                        Models, datasets, pipelines, and phase runner
src/docs/                   Archived research notes and paper references
requirements.txt            Python dependencies
download_model_weights.sh   Bash model download helper
download_datasets.sh        Bash dataset download helper
README.md                   Setup, commands, and phase order
```

See [src/README.md](src/README.md) for implementation details, [SKILLS.md](src/docs/SKILLS.md) for the writing/research workflow, and [ARCHITECTURE.md](src/docs/ARCHITECTURE.md) for stages and artifacts. The writing workflow still has no final LaTeX-draft preparation step.
