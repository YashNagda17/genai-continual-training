#!/usr/bin/env bash
# Fetch only the current plan's mT5-small snapshot. This does not load model tensors.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [mt5|google/mt5-small|all] [hf_token]"
  exit 0
fi
if [[ $# -gt 2 || ( "${1:-mt5}" != "mt5" && "${1:-mt5}" != "google/mt5-small" && "${1:-mt5}" != "all" ) ]]; then
  echo "Usage: $0 [mt5|google/mt5-small|all] [hf_token]" >&2
  exit 1
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${2:-}" ]]; then export HF_TOKEN="$2"; fi
python3 - "$PROJECT_ROOT/src/models/model_weights" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
cache = Path(sys.argv[1])
cache.mkdir(parents=True, exist_ok=True)
print(snapshot_download('google/mt5-small', revision='main', cache_dir=str(cache),
      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'pytorch_model.bin']))
PY
