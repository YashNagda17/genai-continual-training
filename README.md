# Multilingual summarization experiments with continual learning

This project trains mBART on English summarization, then Hindi, and measures how much English performance is retained. It compares replay, block selection, adapter compression, and knowledge distillation using English and Hindi XL-Sum data.

## Main command

After setup and downloads, run this from the repository root (the folder containing `requirements.txt`):

```sh
python -u -m src.summarization_experiment --device cuda:0 --output results/code/summarization-run-01
```

The scientific run attempts the report's five phases in order: 1. Forgetting, 2. Replay, 3. Block selection, 4. Compression, 5. Distillation. It stops after Phase 1 if the forgetting gate does not say `continue`. Use a new `--output` folder for each run. Add `--gradient-checkpointing` if GPU activation memory is tight; it trades speed for lower memory use.

## Step-by-step setup

### 1. Download the code

Install Git and Python 3.11 or newer, then run:

```sh
git clone https://github.com/YashNagda17/genai-continual-training.git
cd genai-continual-training
```

Alternatively, download the repository ZIP from GitHub, extract it, and open a terminal in the extracted folder.

### 2. Create a Python environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Windows CMD, use `.venv\Scripts\activate.bat` instead of the PowerShell activation command. On Linux, WSL, or a cloud terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Reactivate this environment in each new terminal, and run the remaining commands from the repository root.

### 3. Install dependencies and check the GPU

```sh
python -m pip install -r requirements.txt
```

For the NVIDIA GPU commands below, verify CUDA:

```sh
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
```

If CUDA is unavailable, install a CUDA-enabled PyTorch build using the [official PyTorch installer](https://pytorch.org/get-started/locally/) and repeat the check. The tiny CPU check in Step 5 does not require a GPU.

### 4. Download the dataset and model weights

```sh
python -m src.datasets.dataset_sources xlsum src/datasets/data
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('ctu-aic/mbart25-multilingual-summarization-multilarge-cs', revision='main', cache_dir='src/models/model_weights', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'pytorch_model.bin']))"
```

Allow both downloads to finish before a real-model run. The smoke check below can run before these downloads.

### 5. Run the tiny CPU check

```sh
python -u -m src.summarization_experiment --protocol smoke --device cpu --output results/code/summarization-smoke-01
```

This uses a tiny model and a tiny synthetic dataset. It exercises all five phases, but its scores are not scientific results. Choose a fresh output folder each time. A successful run writes `run.json` with `status: completed` and `completed_phases: [1, 2, 3, 4, 5]`.

### 6. Run the real model

The main command at the top runs the full scientific protocol. For a smaller real-model check using 50 training examples per language, one epoch, and saved outputs, run:

```sh
python -u -m src.summarization_experiment --check --examples 50 --device cuda:0 --output results/code/all-phases-50-01
```

`--check` uses the real mBART model and XL-Sum data; it is much heavier than the smoke check. You may add `--gradient-checkpointing` to either real-model command. The scientific defaults use four training passes per language, 2,000 training, 200 validation, and 500 test articles per language. Current model input and training-target limits are 512 and 256 tokens; saved historical results were generated with earlier limits.

## Reports and predictions

Each run saves its reports in its `results/code/<run-name>/phase<1-5>/` folders. The retained imported measurements are summarized in [results/report.md](results/report.md).

Training arms save prediction JSON files under `results/code/<run-name>/phase<1-3>/<arm>/predictions/`. For example, `phase2/drift_past/predictions/hindi-test-plain.json` contains article IDs, target summaries, and generated summaries. English/Hindi, validation/test, and plain/guarded outputs are separate files. Phase 4 serving and Phase 5 distillation store their evaluations in their own reports.

Prediction JSON files are visible to Git; checkpoints and other run artifacts remain ignored. After a new run, use `git add results/code/<run-name>` to include its predictions in your next commit. A run interrupted before evaluation may have fewer prediction files.
