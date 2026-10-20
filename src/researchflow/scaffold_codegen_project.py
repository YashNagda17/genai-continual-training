from __future__ import annotations

import argparse
from pathlib import Path

from .util import artifact_path, render_path_list, slugify_topic, write_text


def maybe_write(path: str, text: str, overwrite: bool) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        return
    write_text(path, text)


def scaffold(topic: str, inputs: list[str], code_root: str, manifest_out: str, overwrite: bool, with_tests: bool) -> None:
    root = Path(code_root)
    dirs = [
        root / "configs",
        root / "src",
        root / "scripts",
        root / "outputs" / "csv",
        root / "outputs" / "json",
        root / "outputs" / "plots",
    ]
    if with_tests:
        dirs.append(root / "tests")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    readme = f"""# {topic}

This directory stores generated code for the approved experiment plan.

## Inputs
{render_path_list(inputs)}

## Layout
- `configs/`: experiment configs and parameter surfaces
- `src/`: reusable implementation modules
- `scripts/`: runnable entry points
- `outputs/csv/`: numeric result tables
- `outputs/json/`: run metadata and config snapshots
- `outputs/plots/`: saved PDF plots
"""
    maybe_write((root / "README.md").as_posix(), readme, overwrite)

    manifest = f"""# Code Generation Report: {topic}
Date: {{date}} | Inputs:
{render_path_list(inputs)}

## Implemented Scope
- Tier 1 experiments implemented: ...
- Tier 2 experiments implemented: ...
- Deferred items: ...

## File Layout
- `{code_root}/...`

## Entry Points
- `scripts/run_main.py`: ...
- `scripts/run_baselines.py`: ...
- `scripts/run_ablation.py`: ...

## Output Files
- CSV: ...
- JSON: ...
- PDF plots: ...

## Reproducibility Controls
- Seeds: ...
- Config snapshots: ...
- Deterministic options: ...

## Validation Performed
- Syntax checks: ...
- Import checks: ...
- Smoke runs: ...
- Unverified parts: ...

## Known Risks
- ...
"""
    maybe_write(manifest_out, manifest.replace("{date}", __import__("datetime").date.today().isoformat()), overwrite)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the standard results/code project layout and manifest scaffold.")
    ap.add_argument("topic", help="Topic string for slugging and report title.")
    ap.add_argument("--input", action="append", default=[], help="Relevant upstream artifact path (repeatable).")
    ap.add_argument("--code-root", default=None, help="Code root directory.")
    ap.add_argument("--manifest-out", default=None, help="Codegen manifest output markdown path.")
    ap.add_argument("--with-tests", action="store_true", help="Create an empty tests/ directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite scaffold files if they already exist.")
    args = ap.parse_args()

    slug = slugify_topic(args.topic)
    code_root = args.code_root or f"results/code/{slug}"
    manifest_out = args.manifest_out or artifact_path("code", "Codegen", args.topic)
    scaffold(args.topic, list(args.input), code_root, manifest_out, args.overwrite, args.with_tests)
    print(f"Wrote code scaffold under {code_root} and manifest {manifest_out}")


if __name__ == "__main__":
    main()

