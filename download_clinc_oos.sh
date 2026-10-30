#!/usr/bin/env bash
# Download clinc/clinc_oos (plus: in-scope plus out-of-scope) and write InstructionDataset JSONL.
# Hub files stay in the local dataset cache. The converted records are the load path.
#
# Usage:
#   ./download_clinc_oos.sh [hf_token]
#
# Output:
#   src/datasets/data/clinc_oos/<commit>/records.jsonl
#   src/datasets/data/clinc_oos/<commit>/manifest.json

set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [hf_token]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$ROOT/src/datasets/data/clinc_oos/hub"
TOKEN="${1:-}"

if [[ -n "$TOKEN" ]]; then
  export HF_TOKEN="$TOKEN"
fi

mkdir -p "$CACHE_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

python3 - "$ROOT" "$CACHE_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
dataset_id = "clinc/clinc_oos"
config = "plus"
splits = ("train", "validation", "test")

snapshot = Path(snapshot_download(
    dataset_id,
    repo_type="dataset",
    revision="main",
    cache_dir=str(cache_dir),
    allow_patterns=[f"{config}/*.parquet", "README.md"],
))
revision = snapshot.name
names = None
rows = []
counts = {}
for split in splits:
    table = pq.read_table(snapshot / config / f"{split}-00000-of-00001.parquet")
    if names is None:
        meta = json.loads(table.schema.metadata[b"huggingface"])
        names = meta["info"]["features"]["intent"]["names"]
    data = table.to_pydict()
    counts[split] = len(data["text"])
    for index, (text, intent) in enumerate(zip(data["text"], data["intent"])):
        label = names[int(intent)]
        rows.append({
            "id": f"{split}-{index:05d}",
            "task": "clinc",
            "split": split,
            "prompt": text,
            "response": label,
            "class": label,
        })

out = root / "src/datasets/data/clinc_oos" / revision
out.mkdir(parents=True, exist_ok=True)
payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
encoded = payload.encode("utf-8")
(out / "records.jsonl").write_bytes(encoded)
manifest = {
    "dataset_id": "clinc_oos",
    "source": dataset_id,
    "config": config,
    "revision": revision,
    "task": "clinc",
    "counts": counts,
    "records": len(rows),
    "sha256": hashlib.sha256(encoded).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"ok {dataset_id}@{revision} config={config} records={len(rows)} -> {out / 'records.jsonl'}")
print("counts", counts)
PY
