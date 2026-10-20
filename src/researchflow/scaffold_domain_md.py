from __future__ import annotations

import argparse

from .util import artifact_path, render_path_list, write_text


def scaffold(topic: str, domain: str | None, idea: str | None, inputs: list[str], out_path: str) -> None:
    idea_text = (idea or "Unknown - fill in the current idea statement.").strip()
    domain_text = (domain or "Unknown").strip()
    text = f"""# Domain Grounding Report: {topic}
Date: {{date}} | Domain: {domain_text} | Inputs:
{render_path_list(inputs)}

## Current Idea
{idea_text}

## Domain Interpretation
- Primary domain: {domain_text}
- Secondary domain(s): ...
- Why this is the right framing: ...

## Claim-to-Theory Map
| Claim | Domain | Theory / Principle | Assumptions | Source | Status |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | Supported / Partial / Unsupported / Incorrect / Unknown |

## Assumption Audit
| Assumption | Why it matters | If false | Evidence needed |
|---|---|---|---|
| ... | ... | ... | ... |

## Corrections to the Current Idea
- Claim or assumption: ...
  - Why it is weak or wrong: ...
  - Corrected domain view: ...
  - Source: ...

## Operating Regime and Failure Regime
- Where theory suggests it should work: ...
- Where theory suggests it should fail: ...
- Boundary conditions that matter: ...

## Theory-Backed Reformulation
- Refined idea statement: ...
- Mechanism: ...
- Why theory suggests it may work: ...
- What must be true for it to work: ...

## Falsifiable Claims
1. ...
2. ...
3. ...

## Experiment Implications
- Datasets / systems / regimes that must be tested: ...
- Baselines required by the theory: ...
- Stress tests implied by the assumptions: ...

## Remaining Unknowns
- ...
"""
    write_text(out_path, text.replace("{date}", __import__("datetime").date.today().isoformat()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold a domain-grounding report markdown file.")
    ap.add_argument("topic", help="Topic string for title and filename.")
    ap.add_argument("--domain", default=None, help="Primary target domain.")
    ap.add_argument("--idea", default=None, help="Current idea statement.")
    ap.add_argument("--input", action="append", default=[], help="Relevant upstream artifact path (repeatable).")
    ap.add_argument("--out", default=None, help="Output markdown path.")
    args = ap.parse_args()

    out_path = args.out or artifact_path("domain", "Domain", args.topic)
    scaffold(args.topic, args.domain, args.idea, list(args.input), out_path)
    print(f"Wrote scaffold to {out_path}")


if __name__ == "__main__":
    main()

