from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from .util import ensure_parent_dir, load_json, slugify_topic


def _fmt_authors(authors: list[str] | None) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def scaffold(topic: str, works: list[dict[str, Any]], out_path: str) -> None:
    lines: list[str] = []
    lines.append(f"# Literature Review: {topic}")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append("- Scope: last ~3 years (rolling window)")
    lines.append("")
    lines.append("## Topic Interpretation")
    lines.append("")
    lines.append("_Write 1-2 sentences clarifying the topic and assumptions._")
    lines.append("")
    lines.append("## Search Strategy")
    lines.append("")
    lines.append("- Queries used: _fill in_")
    lines.append("- Inclusion criteria: last 3 years, direct relevance, strong methods/evidence")
    lines.append("- Exclusion criteria: irrelevant/derivative, missing evidence, non-primary sources")
    lines.append("- Coverage limitations: _fill in_")
    lines.append("")
    lines.append("## Related Works (10)")
    lines.append("")

    for i, w in enumerate(works, start=1):
        title = (w.get("title") or "Unknown title").strip()
        year = w.get("year") or (w.get("published") or "")[:4] or "Unknown year"
        venue = (w.get("venue") or "Unknown venue").strip()
        url = w.get("url") or w.get("pdf_url") or "Unknown"
        doi = w.get("doi") or "Unknown"
        arxiv_id = w.get("arxiv_id") or "Unknown"
        authors = _fmt_authors(w.get("authors"))

        lines.append(f"## [{year}] {title} ({venue})")
        lines.append("")
        lines.append("**Citation**")
        lines.append(f"- Authors: {authors}")
        lines.append(f"- Venue: {venue}")
        lines.append(f"- Year: {year}")
        lines.append(f"- DOI: {doi}")
        lines.append(f"- arXiv: {arxiv_id}")
        lines.append(f"- Link: {url}")
        lines.append("")
        lines.append("**Why it's relevant**")
        lines.append("")
        lines.append("_Explain the connection to the user's idea._")
        lines.append("")
        lines.append("**What it contributes**")
        lines.append("")
        lines.append("_Summarize method + evidence + key results._")
        lines.append("")
        lines.append("**Shortcomings / gaps / failure modes**")
        lines.append("")
        lines.append("_Be specific: missing baselines, leakage, weak eval, scaling confounds, etc._")
        lines.append("")
        lines.append("**Implications for the user's idea**")
        lines.append("")
        lines.append("_How this forces the idea to change (or why it might be non-novel)._")
        lines.append("")

    lines.append("## Cross-Paper Synthesis")
    lines.append("")
    lines.append("- Patterns/consensus: _fill in_")
    lines.append("- Disagreements: _fill in_")
    lines.append("- Gaps still worth pursuing: _fill in_")
    lines.append("")
    lines.append("## Brutal Critique Of The Current Idea")
    lines.append("")
    lines.append("- Likely reviewer objections: _fill in_")
    lines.append("- What looks incremental/derivative: _fill in_")
    lines.append("- Unrealistic assumptions: _fill in_")
    lines.append("")
    lines.append("## Refined Proposal")
    lines.append("")
    lines.append("- Contribution statement: _fill in_")
    lines.append("- 2-3 testable claims: _fill in_")
    lines.append("- Experimental plan (datasets/baselines/metrics): _fill in_")
    lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold a literature review markdown file under results/literature-review/.")
    ap.add_argument("topic", help="Topic string (used for title and filename slug).")
    ap.add_argument("--works-json", default=None, help="Optional JSON file containing {'items': [...]} or a list.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output .md path (default: results/literature-review/literature-review-<slug>-YYYY-MM-DD.md)",
    )
    args = ap.parse_args()

    works: list[dict[str, Any]] = []
    if args.works_json:
        data = load_json(args.works_json)
        if isinstance(data, dict) and "items" in data:
            works = data["items"] or []
        elif isinstance(data, list):
            works = data
        else:
            raise SystemExit("Unsupported JSON format: expected list or object with 'items'.")

    slug = slugify_topic(args.topic)
    out_path = args.out or f"results/literature-review/literature-review-{slug}-{date.today().isoformat()}.md"
    ensure_parent_dir(out_path)
    scaffold(args.topic, works[:10], out_path)
    print(f"Wrote scaffold to {out_path}")


if __name__ == "__main__":
    main()
