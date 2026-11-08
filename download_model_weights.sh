#!/usr/bin/env bash
# Download one Hugging Face model snapshot into src/models/model_weights/ (same layout as model_causal_lm.py).
# Weights on disk stay in hub format. Causal models load in float16, then cast to float32 before the optimizer.
# google/mt5-small is the exception: load it in bf16 where the device supports it, otherwise fp32, never fp16.
#
# Usage:
#   ./download_model_weights.sh <model_id> [hf_token]
#   ./download_model_weights.sh all [hf_token]
#
# Examples:
#   ./download_model_weights.sh Qwen/Qwen3-0.6B
#   ./download_model_weights.sh google/mt5-small
#   ./download_model_weights.sh HuggingFaceTB/SmolLM2-135M-Instruct hf_xxxxxxxx
#   ./download_model_weights.sh all hf_xxxxxxxx

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <model_id|all> [hf_token]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$ROOT/src/models/model_weights"
MODEL="$1"
TOKEN="${2:-}"

if [[ -n "$TOKEN" ]]; then
  export HF_TOKEN="$TOKEN"
fi

mkdir -p "$CACHE_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

python3 - "$CACHE_DIR" "$MODEL" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

cache_dir = Path(sys.argv[1])
selector = sys.argv[2]

if selector == "all":
    models = [
        "Qwen/Qwen3-0.6B",
        "LiquidAI/LFM2.5-350M",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "google/mt5-small",
    ]
else:
    models = [selector]

# *.model keeps mT5's SentencePiece file. pytorch_model.bin is its weight file;
# the causal models publish safetensors instead.
patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "pytorch_model.bin"]
cache_dir.mkdir(parents=True, exist_ok=True)
load_note = {
    "google/mt5-small": "load dtype=bf16 or fp32, never fp16",
}

for model_id in models:
    path = snapshot_download(
        model_id,
        revision="main",
        cache_dir=str(cache_dir),
        allow_patterns=patterns,
    )
    note = load_note.get(model_id, "load dtype=float16, optimizer dtype=float32")
    print(f"ok {model_id} -> {path} ({note})")
PY
