from __future__ import annotations

import argparse
import json

from .util import latest_matching


ALIASES = {
    "lit-review": "results/literature-review/literature-review-*.md",
    "challenger": "results/challenger/Challenge-*.md",
    "domain": "results/domain/Domain-*.md",
    "experiments": "results/experiments/Experiment-Plan-*.md",
    "codegen": "results/code/Codegen-*.md",
    "paper-flow": "results/paper/high-level-paper-*.md",
    "manuscript": "results/paper/manuscript-*.tex",
    "review": "results/review/Review-*.md",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve the latest artifact path for one or more workflow stages.")
    ap.add_argument("stages", nargs="*", choices=sorted(ALIASES), help="Artifact aliases to resolve.")
    ap.add_argument("--glob", dest="globs", action="append", default=[], help="Additional raw glob patterns.")
    ap.add_argument("--plain", action="store_true", help="Print only resolved paths, one per line.")
    args = ap.parse_args()

    resolved: dict[str, str | None] = {}
    for stage in args.stages:
        resolved[stage] = latest_matching(ALIASES[stage])
    for pattern in args.globs:
        resolved[pattern] = latest_matching(pattern)

    if args.plain:
        for value in resolved.values():
            if value:
                print(value)
        return

    print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

