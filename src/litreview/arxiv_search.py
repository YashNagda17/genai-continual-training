from __future__ import annotations

import argparse
from datetime import date
from typing import Any

from .util import coerce_date, dump_json, http_get, in_last_3_years, uniq_by


def require_feedparser():
    try:
        import feedparser  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: feedparser. Install it (e.g. add `feedparser` to requirements.txt)."
        ) from e


def arxiv_api_url(query: str, *, start: int, max_results: int, sort_by: str, sort_order: str) -> str:
    # API reference: https://arxiv.org/help/api/
    # Example query: all:"graph neural network" AND cat:cs.LG
    from urllib.parse import quote_plus

    q = quote_plus(query)
    return (
        "https://export.arxiv.org/api/query?"
        f"search_query={q}&start={start}&max_results={max_results}"
        f"&sortBy={sort_by}&sortOrder={sort_order}"
    )


def parse_feed_bytes(feed_bytes: bytes) -> list[dict[str, Any]]:
    require_feedparser()
    import feedparser

    feed = feedparser.parse(feed_bytes)
    out: list[dict[str, Any]] = []
    for e in feed.entries or []:
        title = (e.get("title") or "").replace("\n", " ").strip()
        summary = (e.get("summary") or "").replace("\n", " ").strip()
        published = e.get("published") or e.get("updated")
        published_date = coerce_date(published)
        year = published_date.year if published_date else None

        # arXiv ids often appear in the entry id like "http://arxiv.org/abs/xxxx.xxxxxvN"
        entry_id = e.get("id") or ""
        arxiv_id = entry_id.rsplit("/", 1)[-1].replace("abs/", "")
        arxiv_id = arxiv_id.split("v", 1)[0] if arxiv_id else None

        authors = [a.get("name") for a in (e.get("authors") or []) if a.get("name")]

        links = e.get("links") or []
        pdf_url = None
        html_url = None
        for ln in links:
            href = ln.get("href")
            if not href:
                continue
            if ln.get("type") == "application/pdf" or href.endswith(".pdf") or "/pdf/" in href:
                pdf_url = href
            if "/abs/" in href:
                html_url = href

        out.append(
            {
                "source": "arxiv",
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published_date.isoformat() if published_date else None,
                "year": year,
                "url": html_url or entry_id or None,
                "pdf_url": pdf_url,
                "arxiv_id": arxiv_id,
                "categories": [t.get("term") for t in (e.get("tags") or []) if t.get("term")],
            }
        )
    return out


def search_arxiv(
    query: str,
    *,
    limit: int = 50,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    last_3_years_only: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    start = 0
    page_size = min(100, max(1, limit))
    now = date.today()

    while len(results) < limit:
        url = arxiv_api_url(query, start=start, max_results=page_size, sort_by=sort_by, sort_order=sort_order)
        feed_bytes = http_get(url)
        page = parse_feed_bytes(feed_bytes)
        if not page:
            break
        for it in page:
            if last_3_years_only:
                pub = coerce_date(it.get("published"))
                if not in_last_3_years(pub, now=now):
                    continue
            results.append(it)
            if len(results) >= limit:
                break
        start += page_size

    # de-dupe on arxiv_id, then title
    results = uniq_by(results, ["arxiv_id", "title"])
    return results[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description="Search arXiv and output JSON candidates.")
    ap.add_argument("query", help='arXiv API query string, e.g. all:"graph neural networks" AND cat:cs.LG')
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--all-years", action="store_true", help="Do not restrict to last ~3 years.")
    ap.add_argument("--out", default="results/literature-review/arxiv-results.json")
    args = ap.parse_args()

    items = search_arxiv(args.query, limit=args.limit, last_3_years_only=not args.all_years)
    from .util import ensure_parent_dir

    ensure_parent_dir(args.out)
    dump_json(args.out, {"query": args.query, "count": len(items), "items": items})
    print(f"Wrote {len(items)} results to {args.out}")


if __name__ == "__main__":
    main()
