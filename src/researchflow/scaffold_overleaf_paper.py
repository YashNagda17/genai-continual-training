from __future__ import annotations

import argparse

from .util import artifact_path, render_path_list, slugify_topic, today_ymd, write_text


def manuscript_text(topic: str, venue: str | None, inputs: list[str]) -> str:
    venue_text = venue or "Unknown venue"
    return f"""% Auto-generated scaffold for {topic}
% Date: {today_ymd()}
% Venue: {venue_text}
% Inputs:
% {chr(10).join('% ' + p for p in inputs) if inputs else '% None provided'}

\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

\\title{{{topic}}}
\\author{{[AUTHOR NAMES]}}
\\date{{{today_ymd()}}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
% TODO: Fill from the evidence-backed two-sentence contribution statement.
\\end{{abstract}}

\\section{{Introduction}}
% TODO: Problem framing, why now, and contribution ladder positioning.

\\section{{Related Work}}
% TODO: Closest prior work, overlap risks, and differentiation.

\\section{{Method}}
% TODO: Evidence-backed method description only.

\\section{{Experimental Setup}}
% TODO: Datasets, baselines, metrics, and evaluation protocol.

\\section{{Results}}
% TODO: Add only numbers backed by current CSV/JSON artifacts.

\\section{{Discussion and Limitations}}
% TODO: Main risks, unsupported claims, and known failure modes.

\\section{{Conclusion}}
% TODO: Keep claims at the rung supported by evidence.

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""


def handoff_text(topic: str, venue: str | None, manuscript_out: str, inputs: list[str]) -> str:
    venue_text = venue or "Unknown venue"
    return f"""# Paper Handoff: {topic}
Date: {today_ymd()} | Venue: {venue_text}

## Files Written
- Main draft: {manuscript_out}

## Inputs Used
{render_path_list(inputs)}

## Missing Citations
- ...

## Missing Experiments
- ...

## Unresolved Placeholders
- ...

## Venue Assumptions Used
- ...
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold an Overleaf-ready manuscript .tex and handoff markdown file.")
    ap.add_argument("topic", help="Topic string for title and filename.")
    ap.add_argument("--venue", default=None, help="Target venue or conference.")
    ap.add_argument("--input", action="append", default=[], help="Relevant upstream artifact path (repeatable).")
    ap.add_argument("--manuscript-out", default=None, help="Output .tex path.")
    ap.add_argument("--handoff-out", default=None, help="Output handoff markdown path.")
    args = ap.parse_args()

    slug = slugify_topic(args.topic)
    manuscript_out = args.manuscript_out or f"results/paper/manuscript-{slug}-{today_ymd()}.tex"
    handoff_out = args.handoff_out or f"results/paper/paper-handoff-{slug}-{today_ymd()}.md"

    write_text(manuscript_out, manuscript_text(args.topic, args.venue, list(args.input)))
    write_text(handoff_out, handoff_text(args.topic, args.venue, manuscript_out, list(args.input)))
    print(f"Wrote manuscript scaffold to {manuscript_out}")
    print(f"Wrote handoff scaffold to {handoff_out}")


if __name__ == "__main__":
    main()

