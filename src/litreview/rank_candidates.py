"""rank_candidates.py

Score and rank merged candidate papers by a composite of:
  - Recency (higher = newer)
  - Citation count (log-scaled, from Semantic Scholar source)
  - Abstract keyword overlap with a user-provided topic query

Usage (module):
    PYTHONPATH=src python -m litreview.rank_candidates merged.json "retrieval augmented generation" --top 15 --out ranked.json
"""
from __future__ import annotations

import argparse
import math
import re
from datetime import date
from typing import Any

from .util import coerce_date, dump_json, ensure_parent_dir, in_last_3_years, load_json


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, length >= 3."""
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())}


def _recency_score(item: dict[str, Any], now: date) -> float:
    """1.0 for today, decaying toward 0.0 over 3 years."""
    pub = coerce_date(item.get("published")) or (
        date(int(item["year"]), 6, 1) if item.get("year") else None
    )
    if pub is None:
        return 0.0
    delta_days = (now - pub).days
    window_days = 365 * 3
    return max(0.0, 1.0 - delta_days / window_days)


def _citation_score(item: dict[str, Any]) -> float:
    """Log-scaled citation count, normalised to [0, 1] using log(1000) as ceiling."""
    c = item.get("citation_count") or item.get("citationCount") or 0
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = 0.0
    if c <= 0:
        return 0.0
    ceiling = math.log1p(1000)
    return min(1.0, math.log1p(c) / ceiling)


def _keyword_score(item: dict[str, Any], query_tokens: set[str]) -> float:
    """Fraction of query tokens found in title + abstract."""
    if not query_tokens:
        return 0.0
    text = " ".join(filter(None, [item.get("title") or "", item.get("abstract") or ""]))
    found = query_tokens & _tokenize(text)
    return len(found) / len(query_tokens)


def score_items(
    items: list[dict[str, Any]],
    query: str,
    *,
    w_recency: float = 0.35,
    w_citation: float = 0.30,
    w_keyword: float = 0.35,
) -> list[dict[str, Any]]:
    now = date.today()
    query_tokens = _tokenize(query)
    scored = []
    for it in items:
        r = _recency_score(it, now)
        c = _citation_score(it)
        k = _keyword_score(it, query_tokens)
        score = w_recency * r + w_citation * c + w_keyword * k
        scored.append({**it, "_score": round(score, 4), "_recency": round(r, 4),
                        "_citation": round(c, 4), "_keyword": round(k, 4)})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score and rank candidate papers. Writes a ranked JSON file."
    )
    ap.add_argument("input", help="Merged candidates JSON (object with 'items' or bare list).")
    ap.add_argument("query", help="Topic query string used for keyword scoring.")
    ap.add_argument("--top", type=int, default=20, help="Keep top-N results (default 20).")
    ap.add_argument(
        "--w-recency", type=float, default=0.35, metavar="W",
        help="Weight for recency score (default 0.35)."
    )
    ap.add_argument(
        "--w-citation", type=float, default=0.30, metavar="W",
        help="Weight for citation score (default 0.30)."
    )
    ap.add_argument(
        "--w-keyword", type=float, default=0.35, metavar="W",
        help="Weight for keyword overlap score (default 0.35)."
    )
    ap.add_argument("--out", default="results/literature-review/candidates-ranked.json")
    args = ap.parse_args()

    data = load_json(args.input)
    items: list[dict[str, Any]] = data["items"] if isinstance(data, dict) else data

    ranked = score_items(
        items,
        args.query,
        w_recency=args.w_recency,
        w_citation=args.w_citation,
        w_keyword=args.w_keyword,
    )
    top = ranked[: args.top]

    ensure_parent_dir(args.out)
    dump_json(args.out, {"query": args.query, "total_scored": len(ranked), "count": len(top), "items": top})
    print(f"Scored {len(ranked)} candidates, wrote top {len(top)} to {args.out}")


if __name__ == "__main__":
    main()
