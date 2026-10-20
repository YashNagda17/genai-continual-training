from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .util import ensure_parent_dir, slugify_topic


def scaffold(*, idea: str, topic: str | None, lit_review_paths: list[str], out_path: str) -> None:
    lines: list[str] = []
    lines.append("# Challenge")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    if topic:
        lines.append(f"- Topic: {topic}")
        lines.append(f"- Topic slug: {slugify_topic(topic)}")
    lines.append("")
    lines.append("## Idea Statement (Verbatim)")
    lines.append("")
    lines.append(idea.strip() or "Unknown (user did not provide).")
    lines.append("")
    lines.append("## Declared Constraints (Verbatim)")
    lines.append("")
    lines.append("- Deployment setting: Unknown")
    lines.append("- Data availability/licensing: Unknown")
    lines.append("- Privacy/security constraints: Unknown")
    lines.append("- Compute budget/latency: Unknown")
    lines.append("- Target venues: Unknown")
    lines.append("")
    lines.append("## Unknowns Blocking Conclusions")
    lines.append("")
    lines.append("- _Fill in unknowns that prevent a grounded critique._")
    lines.append("")
    lines.append("## Evidence Base Used")
    lines.append("")
    lines.append("### Local Literature Review Artifacts")
    lines.append("")
    if lit_review_paths:
        for p in lit_review_paths:
            lines.append(f"- {p}")
    else:
        lines.append("- Unknown (no literature-review file paths provided).")
    lines.append("")
    lines.append("### Additional Web Sources")
    lines.append("")
    lines.append("- _Add inline links as you cite them. Do not rely on uncited claims._")
    lines.append("")
    lines.append("## Novelty And Prior-Art Mismatch")
    lines.append("")
    lines.append("- _List closest prior works and explain overlap (with citations)._")
    lines.append("")
    lines.append("## Assumption Audit")
    lines.append("")
    lines.append("| Assumption | Why risky | Evidence needed | Consequence if false |")
    lines.append("|---|---|---|---|")
    lines.append("| _Fill_ | _Fill_ | _Fill_ | _Fill_ |")
    lines.append("")
    lines.append("## Production / Industry Challenge")
    lines.append("")
    lines.append("- Cost/latency: _Fill with citations_")
    lines.append("- Privacy/security: _Fill with citations_")
    lines.append("- Monitoring and failure modes: _Fill with citations_")
    lines.append("- Governance/compliance: _Fill with citations_")
    lines.append("")
    lines.append("## Evaluation Attack Plan")
    lines.append("")
    lines.append("- How to kill this idea: _Fill_")
    lines.append("- Required baselines: _Fill_")
    lines.append("- Required stress tests: _Fill_")
    lines.append("")
    lines.append("## Venue Acceptance Check")
    lines.append("")
    lines.append("- _For each target venue, cite official pages and explain fit/mismatch._")
    lines.append("")
    lines.append("## Brutal Verdict")
    lines.append("")
    lines.append("- _3-5 strongest rejection reasons as currently framed (each grounded)._")
    lines.append("")
    lines.append("## Salvage Options (Evidence-Backed Pivots)")
    lines.append("")
    lines.append("- _2-4 pivots with tighter claims, clearer scope, and measurable contributions._")
    lines.append("")

    ensure_parent_dir(out_path)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold results/challenger/Challenge-<slug>-YYYY-MM-DD.md for the challenger skill.")
    ap.add_argument("idea", help="Idea statement (prefer verbatim user text).")
    ap.add_argument("--topic", default=None, help="Optional topic label for slugging/metadata.")
    ap.add_argument(
        "--lit-review",
        action="append",
        default=[],
        help="Path(s) to literature-review markdown files under results/literature-review/ (repeatable).",
    )
    ap.add_argument("--out", default=None, help="Output markdown path (default: results/challenger/Challenge-<slug>-YYYY-MM-DD.md).")
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        slug_source = args.topic or args.idea
        out_path = f"results/challenger/Challenge-{slugify_topic(slug_source)}-{date.today().isoformat()}.md"

    scaffold(idea=args.idea, topic=args.topic, lit_review_paths=list(args.lit_review), out_path=out_path)
    print(f"Wrote scaffold to {out_path}")


if __name__ == "__main__":
    main()
