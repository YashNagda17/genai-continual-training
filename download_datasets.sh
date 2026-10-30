#!/usr/bin/env bash
# Fetch each instruction source into canonical JSONL. Hub files stay in the source cache.
#
# Usage:
#   ./download_datasets.sh [clinc_oos|schuster|bitext_customer_support|trace|all] [hf_token]
#
# Output, for each selected source:
#   src/datasets/data/<dataset_id>/<commit>/records.jsonl
#   src/datasets/data/<dataset_id>/<commit>/manifest.json

set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [clinc_oos|schuster|bitext_customer_support|trace|all] [hf_token]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-all}"
TOKEN="${2:-}"

if [[ -n "$TOKEN" ]]; then
  export HF_TOKEN="$TOKEN"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

cd "$ROOT"
python3 - "$ROOT/src/datasets/data" "$NAME" "$TOKEN" <<'PY'
import sys
from pathlib import Path

from src.datasets.dataset_sources import BitextSource, ClincOOSSource, SchusterSource, TraceSource

root = Path(sys.argv[1])
name = sys.argv[2]
token = sys.argv[3] or None

# Same conversion path for every source: download the pinned snapshot, write JSONL.
sources = {
    "clinc_oos": ClincOOSSource,                  # plus split, one task, intent name is the response
    "schuster": SchusterSource,                  # English alarm / reminder / weather intents
    "bitext_customer_support": BitextSource,     # category tasks, intent class, assistant reply
    "trace": TraceSource,                        # eight TRACE-5000 tasks, Lima excluded
}
selected = list(sources) if name == "all" else [name]
if name not in sources and name != "all":
    raise SystemExit(f"Unknown dataset {name}")
for key in selected:
    print(sources[key]().fetch(root, token=token))
PY
