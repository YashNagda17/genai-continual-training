from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Any
from urllib.parse import urlencode

from .util import coerce_date, dump_json, http_get, in_last_3_years, uniq_by


S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"


def s2_search(
    query: str,
    *,
    limit: int = 50,
    last_3_years_only: bool = True,
    fields: str | None = None,
) -> list[dict[str, Any]]:
    # Public endpoint is rate limited; for heavier usage, set S2_API_KEY.
    # Docs: https://api.semanticscholar.org/api-docs/graph
    fields = fields or ",".join(
        [
            "title",
            "authors",
            "year",
            "venue",
            "publicationDate",
            "publicationTypes",
            "externalIds",
            "url",
            "openAccessPdf",
            "citationCount",
            "influentialCitationCount",
            "abstract",
        ]
    )

    items: list[dict[str, Any]] = []
    offset = 0
    page_size = min(100, max(1, limit))
    now = date.today()

    headers = {}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    while len(items) < limit:
        params = {"query": query, "limit": page_size, "offset": offset, "fields": fields}
        url = f"{S2_ENDPOINT}?{urlencode(params)}"
        raw = http_get(url, headers=headers)

        import json

        data = json.loads(raw.decode("utf-8"))
        page = data.get("data") or []
        if not page:
            break

        for p in page:
            title = (p.get("title") or "").strip()
            year = p.get("year")
            pub_date = coerce_date(p.get("publicationDate")) or (date(int(year), 1, 1) if year else None)
            if last_3_years_only and not in_last_3_years(pub_date, now=now):
                continue

            ext = p.get("externalIds") or {}
            doi = ext.get("DOI") or ext.get("doi")
            arxiv_id = ext.get("ArXiv") or ext.get("arXiv") or ext.get("ARXIV")
            oa = p.get("openAccessPdf") or {}

            items.append(
                {
                    "source": "semantic_scholar",
                    "title": title,
                    "authors": [a.get("name") for a in (p.get("authors") or []) if a.get("name")],
                    "year": year,
                    "venue": p.get("venue"),
                    "published": pub_date.isoformat() if pub_date else None,
                    "url": p.get("url"),
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "pdf_url": oa.get("url"),
                    "citation_count": p.get("citationCount"),
                    "influential_citation_count": p.get("influentialCitationCount"),
                    "abstract": p.get("abstract"),
                    "publication_types": p.get("publicationTypes"),
                }
            )

        offset += page_size

    items = uniq_by(items, ["doi", "arxiv_id", "title"])
    return items[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description="Search Semantic Scholar and output JSON candidates.")
    ap.add_argument("query", help='Search query, e.g. "retrieval augmented generation evaluation"')
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--all-years", action="store_true", help="Do not restrict to last ~3 years.")
    ap.add_argument("--out", default="results/literature-review/s2-results.json")
    args = ap.parse_args()

    items = s2_search(args.query, limit=args.limit, last_3_years_only=not args.all_years)
    from .util import ensure_parent_dir

    ensure_parent_dir(args.out)
    dump_json(args.out, {"query": args.query, "count": len(items), "items": items})
    print(f"Wrote {len(items)} results to {args.out}")


if __name__ == "__main__":
    main()
