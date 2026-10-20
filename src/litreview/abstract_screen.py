"""abstract_screen.py

Keyword-relevance filter for candidate papers.

Given a merged JSON file and a topic string, computes a fast inclusion/exclusion
decision for each paper based on:
  - Required terms (ALL must appear in title or abstract)
  - Excluded terms (ANY causes rejection)
  - Topic keyword overlap (configurable minimum threshold)

This is intentionally lightweight and deterministic — no LLM, no API calls.
Use it to narrow 50-100 candidates to a smaller set before the agent reads them.

Usage:
    PYTHONPATH=src python -m litreview.abstract_screen merged.json "RAG evaluation long-context QA" \\
        --require "retrieval" --require "evaluat" \\
        --exclude "survey" --exclude "tutorial" \\
        --min-score 0.2 --out screened.json
"""
from __future__ import annotations

import argparse
import re
from typing import Any

from .util import dump_json, ensure_parent_dir, load_json


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _full_text(item: dict[str, Any]) -> str:
    return " ".join(filter(None, [item.get("title") or "", item.get("abstract") or ""]))


def screen(
    items: list[dict[str, Any]],
    query: str,
    *,
    required: list[str] | None = None,
    excluded: list[str] | None = None,
    min_score: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (accepted, rejected) lists with screening metadata attached."""
    query_tokens = _tokens(query)
    accepted, rejected = [], []

    for it in items:
        text = _full_text(it).lower()
        text_tokens = _tokens(text)

        # Required term check (substring, not token — allows stemming like "evaluat")
        fail_required: list[str] = []
        for term in (required or []):
            if term.lower() not in text:
                fail_required.append(term)

        # Excluded term check
        fail_excluded: list[str] = []
        for term in (excluded or []):
            if term.lower() in text:
                fail_excluded.append(term)

        # Keyword overlap score
        overlap = query_tokens & text_tokens
        score = len(overlap) / len(query_tokens) if query_tokens else 1.0

        meta = {
            "_screen_score": round(score, 4),
            "_screen_overlap": sorted(overlap),
            "_screen_missing_required": fail_required,
            "_screen_hit_excluded": fail_excluded,
        }

        if fail_required or fail_excluded or score < min_score:
            rejected.append({**it, **meta, "_screen_pass": False})
        else:
            accepted.append({**it, **meta, "_screen_pass": True})

    # Sort accepted by score descending
    accepted.sort(key=lambda x: x["_screen_score"], reverse=True)
    return accepted, rejected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Screen candidate papers by keyword relevance. Writes accepted/rejected JSON."
    )
    ap.add_argument("input", help="Merged candidates JSON (object with 'items' or bare list).")
    ap.add_argument("query", help="Topic string for keyword overlap scoring.")
    ap.add_argument(
        "--require", action="append", default=[], metavar="TERM",
        help="Substring that MUST appear in title/abstract (repeatable). Supports partial stems."
    )
    ap.add_argument(
        "--exclude", action="append", default=[], metavar="TERM",
        help="Substring that causes rejection if found (repeatable)."
    )
    ap.add_argument(
        "--min-score", type=float, default=0.1,
        help="Minimum keyword overlap fraction to accept (default 0.1)."
    )
    ap.add_argument("--out", default="results/literature-review/candidates-screened.json",
                    help="Output path for accepted papers.")
    ap.add_argument("--rejected-out", default=None,
                    help="Optional path to write rejected papers for inspection.")
    args = ap.parse_args()

    data = load_json(args.input)
    items: list[dict[str, Any]] = data["items"] if isinstance(data, dict) else data

    accepted, rejected = screen(
        items,
        args.query,
        required=args.require,
        excluded=args.exclude,
        min_score=args.min_score,
    )

    ensure_parent_dir(args.out)
    dump_json(args.out, {"query": args.query, "count": len(accepted), "items": accepted})
    print(f"Accepted {len(accepted)} / {len(items)} papers → {args.out}")

    if args.rejected_out:
        ensure_parent_dir(args.rejected_out)
        dump_json(args.rejected_out, {"count": len(rejected), "items": rejected})
        print(f"Rejected {len(rejected)} papers → {args.rejected_out}")


if __name__ == "__main__":
    main()
