from __future__ import annotations

import argparse
from pathlib import Path

from .util import append_summary_entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a prepared markdown entry to results/summary.md.")
    ap.add_argument("--entry-file", required=True, help="Markdown file containing one summary entry.")
    ap.add_argument("--summary", default="results/summary.md", help="Summary markdown path.")
    args = ap.parse_args()

    entry_text = Path(args.entry_file).read_text(encoding="utf-8")
    append_summary_entry(args.summary, entry_text)
    print(f"Appended summary entry from {args.entry_file} to {args.summary}")


if __name__ == "__main__":
    main()

