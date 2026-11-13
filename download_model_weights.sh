#!/usr/bin/env bash
# Download the mBART summarizer into the cache used by the experiment runner.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [mbart|ctu-aic/mbart25-multilingual-summarization-multilarge-cs|all] [hf_token]"
  exit 0
fi
if [[ $# -gt 2 || ( "${1:-mbart}" != "mbart" && "${1:-mbart}" != "ctu-aic/mbart25-multilingual-summarization-multilarge-cs" && "${1:-mbart}" != "all" ) ]]; then
  echo "Usage: $0 [mbart|ctu-aic/mbart25-multilingual-summarization-multilarge-cs|all] [hf_token]" >&2
  exit 1
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
if [[ -n "${2:-}" ]]; then export HF_TOKEN="$2"; fi
"$PYTHON_BIN" - "$PROJECT_ROOT/src/models/model_weights" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
cache = Path(sys.argv[1])
cache.mkdir(parents=True, exist_ok=True)
print(snapshot_download('ctu-aic/mbart25-multilingual-summarization-multilarge-cs', revision='main', cache_dir=str(cache),
      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'pytorch_model.bin']))
PY
