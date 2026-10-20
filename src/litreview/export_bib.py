"""export_bib.py

Convert a candidate/ranked JSON file to a BibTeX .bib file.

Each item becomes a @article or @misc entry (preprint if no venue).
Citation keys are auto-generated as FirstAuthorYYYY_<slug>.

Usage:
    PYTHONPATH=src python -m litreview.export_bib ranked.json --out references.bib
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .util import ensure_parent_dir, load_json, slugify_topic


def _escape_bib(s: str) -> str:
    """Minimal BibTeX special-char escaping."""
    return (
        s.replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def _cite_key(item: dict[str, Any]) -> str:
    """Generate a unique-ish BibTeX citation key: FirstAuthorYYYY_titleword."""
    authors = item.get("authors") or []
    first = (authors[0] if authors else "Unknown").split()[-1]  # last name of first author
    first = re.sub(r"[^a-zA-Z]", "", first)[:12] or "Anon"
    year = str(item.get("year") or "XXXX")
    title = item.get("title") or ""
    # First significant word of title
    words = [w for w in re.findall(r"[A-Za-z]{4,}", title) if w.lower() not in
             {"with", "from", "that", "this", "using", "based", "towards", "toward",
              "large", "deep", "neural", "learning", "model", "models"}]
    slug = words[0][:10] if words else "paper"
    return f"{first}{year}_{slug}"


def _format_authors(authors: list[str] | None) -> str:
    if not authors:
        return "Unknown"
    return " and ".join(authors)


def item_to_bib(item: dict[str, Any], key: str) -> str:
    """Render a single BibTeX entry string."""
    venue = (item.get("venue") or "").strip()
    arxiv_id = item.get("arxiv_id") or ""
    doi = item.get("doi") or ""
    url = item.get("url") or item.get("pdf_url") or ""
    year = item.get("year") or ""
    title = _escape_bib((item.get("title") or "Unknown title").strip())
    authors_str = _format_authors(item.get("authors"))

    # Choose entry type
    if venue and venue.lower() not in {"", "arxiv", "biorxiv", "medrxiv", "ssrn", "preprint"}:
        entry_type = "article"
    else:
        entry_type = "misc"

    fields: list[tuple[str, str]] = [
        ("author", authors_str),
        ("title", f"{{{title}}}"),
        ("year", str(year)),
    ]
    if venue and entry_type == "article":
        fields.append(("journal", _escape_bib(venue)))
    if doi:
        fields.append(("doi", doi))
    if arxiv_id:
        fields.append(("eprint", arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if url and not doi:
        fields.append(("url", url))
    if item.get("abstract"):
        # Trim abstract to 400 chars to keep .bib readable
        abstract = _escape_bib((item["abstract"] or "")[:400].replace("\n", " ").strip())
        if abstract:
            fields.append(("abstract", f"{{{abstract}...}}"))

    field_lines = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}"
                              for k, v in fields)
    return f"@{entry_type}{{{key},\n{field_lines}\n}}"


def export_bib(items: list[dict[str, Any]], out_path: str) -> int:
    keys_used: dict[str, int] = {}
    entries: list[str] = []
    for it in items:
        base_key = _cite_key(it)
        count = keys_used.get(base_key, 0)
        key = base_key if count == 0 else f"{base_key}_{chr(ord('a') + count - 1)}"
        keys_used[base_key] = count + 1
        entries.append(item_to_bib(it, key))
    ensure_parent_dir(out_path)
    Path(out_path).write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export candidate/ranked JSON to a BibTeX .bib file."
    )
    ap.add_argument("input", help="JSON file with 'items' array or bare list.")
    ap.add_argument("--out", default="results/literature-review/references.bib",
                    help="Output .bib path (default: results/literature-review/references.bib).")
    ap.add_argument("--top", type=int, default=None,
                    help="Export only top-N items (default: all).")
    args = ap.parse_args()

    data = load_json(args.input)
    items: list[dict[str, Any]] = data["items"] if isinstance(data, dict) else data
    if args.top is not None:
        items = items[: args.top]

    n = export_bib(items, args.out)
    print(f"Wrote {n} BibTeX entries to {args.out}")


if __name__ == "__main__":
    main()
