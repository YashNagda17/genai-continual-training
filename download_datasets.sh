#!/usr/bin/env bash
# Fetch the English/Hindi XL-Sum archives and convert them to canonical JSONL.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [xlsum|all] [hf_token]"
  exit 0
fi
if [[ $# -gt 2 || ( "${1:-xlsum}" != "xlsum" && "${1:-xlsum}" != "all" ) ]]; then
  echo "Usage: $0 [xlsum|all] [hf_token]" >&2
  exit 1
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${2:-}" ]]; then export HF_TOKEN="$2"; fi
cd "$PROJECT_ROOT"
python3 -m src.datasets.dataset_sources xlsum "$PROJECT_ROOT/src/datasets/data"
