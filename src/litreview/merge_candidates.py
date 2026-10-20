from __future__ import annotations

import argparse
from typing import Any

from .util import dump_json, ensure_parent_dir, load_json, uniq_by


def load_items(path: str) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "items" in data:
        return data["items"] or []
    if isinstance(data, list):
        return data
    raise SystemExit(f"Unsupported JSON shape in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge and de-duplicate candidate paper JSON files.")
    ap.add_argument("inputs", nargs="+", help="Input JSON files (arXiv/S2/OpenAlex/etc.)")
    ap.add_argument("--out", default="results/literature-review/candidates-merged.json")
    args = ap.parse_args()

    merged: list[dict[str, Any]] = []
    for p in args.inputs:
        merged.extend(load_items(p))

    merged = uniq_by(merged, ["doi", "arxiv_id", "title"])
    ensure_parent_dir(args.out)
    dump_json(args.out, {"count": len(merged), "items": merged})
    print(f"Wrote {len(merged)} merged candidates to {args.out}")


if __name__ == "__main__":
    main()
