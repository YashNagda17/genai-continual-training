"""openalex_search.py

Query the OpenAlex API for candidate papers.

OpenAlex is free, requires no API key (though setting OPENALEX_EMAIL in the
environment enables the "polite pool" with higher rate limits).

API docs: https://docs.openalex.org/

Usage:
    PYTHONPATH=src python -m litreview.openalex_search "contrastive learning time series" --limit 40 --out oa.json
    OPENALEX_EMAIL=you@example.com PYTHONPATH=src python -m litreview.openalex_search "RAG evaluation" --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Any
from urllib.parse import urlencode

from .util import coerce_date, dump_json, ensure_parent_dir, http_get, in_last_3_years, uniq_by

OA_SEARCH_ENDPOINT = "https://api.openalex.org/works"


def _extract_item(w: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex Work object to the shared candidate schema."""
    title = (w.get("title") or "").strip()

    # Authors
    authors = [
        a.get("author", {}).get("display_name")
        for a in (w.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    ]

    # Venue / source
    primary_loc = w.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue = source.get("display_name") or ""

    # Dates / year
    pub_date_str = w.get("publication_date") or ""
    pub_date = coerce_date(pub_date_str)
    year = w.get("publication_year") or (pub_date.year if pub_date else None)

    # IDs
    ids = w.get("ids") or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "").strip() or None
    arxiv_raw = ids.get("arxiv") or ""
    arxiv_id = arxiv_raw.replace("https://arxiv.org/abs/", "").strip() or None

    # URLs
    oa_url = w.get("id") or None  # OpenAlex canonical URL
    landing_url = primary_loc.get("landing_page_url") or None
    pdf_url = primary_loc.get("pdf_url") or None

    abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))

    return {
        "source": "openalex",
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue or None,
        "published": pub_date.isoformat() if pub_date else (str(year) if year else None),
        "url": landing_url or oa_url,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pdf_url": pdf_url,
        "citation_count": w.get("cited_by_count"),
        "abstract": abstract,
    }


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as inverted index {word: [positions]}."""
    if not inverted:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join(w for _, w in words)


def openalex_search(
    query: str,
    *,
    limit: int = 50,
    last_3_years_only: bool = True,
    email: str | None = None,
) -> list[dict[str, Any]]:
    now = date.today()
    items: list[dict[str, Any]] = []
    page = 1
    per_page = min(200, max(1, limit))

    _email = email or os.environ.get("OPENALEX_EMAIL")
    mailto = f"mailto:{_email}" if _email else None

    while len(items) < limit:
        params: dict[str, Any] = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "select": ",".join([
                "id", "title", "authorships", "publication_year", "publication_date",
                "primary_location", "ids", "cited_by_count", "abstract_inverted_index",
            ]),
            "sort": "publication_date:desc",
        }
        if mailto:
            params["mailto"] = _email  # polite-pool header variant

        url = f"{OA_SEARCH_ENDPOINT}?{urlencode(params)}"
        headers: dict[str, str] = {}
        if mailto:
            headers["User-Agent"] = f"litreview/0.1 ({mailto})"

        raw = http_get(url, headers=headers)
        data = json.loads(raw.decode("utf-8"))
        results = data.get("results") or []
        if not results:
            break

        for w in results:
            it = _extract_item(w)
            pub = coerce_date(it.get("published"))
            if last_3_years_only and not in_last_3_years(pub, now=now):
                continue
            items.append(it)
            if len(items) >= limit:
                break

        # Check if there are more pages
        meta = data.get("meta") or {}
        total = meta.get("count") or 0
        if page * per_page >= total:
            break
        page += 1

    items = uniq_by(items, ["doi", "arxiv_id", "title"])
    return items[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Search OpenAlex and output JSON candidates."
    )
    ap.add_argument("query", help="Free-text search query.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--all-years", action="store_true",
                    help="Do not restrict to last ~3 years.")
    ap.add_argument("--email", default=None,
                    help="Email for OpenAlex polite pool (also reads OPENALEX_EMAIL env var).")
    ap.add_argument("--out", default="results/literature-review/openalex-results.json")
    args = ap.parse_args()

    items = openalex_search(
        args.query,
        limit=args.limit,
        last_3_years_only=not args.all_years,
        email=args.email,
    )
    ensure_parent_dir(args.out)
    dump_json(args.out, {"query": args.query, "count": len(items), "items": items})
    print(f"Wrote {len(items)} results to {args.out}")


if __name__ == "__main__":
    main()
