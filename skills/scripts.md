---
name: scripts
description: >
  Script registry skill. Use when you need to run existing repo Python utilities instead of rewriting one-off code.
  Contains a catalog of available scripts (inputs/outputs/scope/dependencies) and the canonical CLI invocations.
---

Use this skill whenever you are about to write ad-hoc Python to do something that an existing repo script already supports.

## How To Run Repo Scripts

These scripts live under `src/` and are written as importable modules. Run them with:

```bash
PYTHONPATH=src python -m <module> [args...]
```

Example:

```bash
PYTHONPATH=src python -m litreview.arxiv_search 'all:"graph neural network" AND cat:cs.LG' --limit 30 --out arxiv.json
```

If a script errors with "Missing dependency", add the listed package(s) to `requirements.txt` and install.

Default output location:
- Unless you override `--out` / `--out-dir`, scripts in this registry write into `results/`.
- Scripts create the needed `results/<stage>/` directory automatically if missing.

## Script Catalog

### `litreview.arxiv_search`

Scope:
- Query arXiv via the official arXiv API and return candidate papers.
- Defaults to filtering to a rolling "last ~3 years" window.

Inputs:
- Positional `query` (string): arXiv API query, e.g. `all:"retrieval augmented generation" AND cat:cs.CL`.
- `--limit` (int): max results (default 50).
- `--all-years` (flag): disables last-3-years filtering.
- `--out` (path): JSON output path (default `results/literature-review/arxiv-results.json`).

Outputs:
- JSON file: `{"query": "...", "count": N, "items": [...]}`.
- Each item includes: `title`, `authors`, `published` (ISO), `year`, `url`, `pdf_url`, `arxiv_id`, `categories`.

Dependencies:
- Required: `requests`, `feedparser`

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.arxiv_search 'all:"diffusion model" AND cat:cs.CV' --limit 80 --out arxiv.json
```

### `litreview.semantic_scholar_search`

Scope:
- Query Semantic Scholar Graph API for candidate papers.
- Defaults to filtering to a rolling "last ~3 years" window.

Inputs:
- Positional `query` (string): free-text query.
- `--limit` (int): max results (default 50).
- `--all-years` (flag): disables last-3-years filtering.
- `--out` (path): JSON output path (default `results/literature-review/s2-results.json`).
- Optional env var: `S2_API_KEY` for higher rate limits.

Outputs:
- JSON file: `{"query": "...", "count": N, "items": [...]}`.
- Each item includes: `title`, `authors`, `year`, `venue`, `published`, `url`, `doi`, `arxiv_id`, `pdf_url`,
  `citation_count`, `influential_citation_count`, `abstract`.

Dependencies:
- Required: `requests`

Canonical usage:
```bash
S2_API_KEY=... PYTHONPATH=src python -m litreview.semantic_scholar_search "retrieval augmented generation evaluation" --limit 50 --out s2.json
```

### `litreview.merge_candidates`

Scope:
- Merge multiple candidate JSON files (from arXiv / Semantic Scholar / etc.) and de-duplicate.
- De-duplication keys: `(doi, arxiv_id, title)` lowercased; best-effort.

Inputs:
- Positional `inputs...` (paths): JSON files. Accepts either:
  - object with `items`, or
  - a top-level list of items.
- `--out` (path): merged JSON path (default `results/literature-review/candidates-merged.json`).

Outputs:
- JSON file: `{"count": N, "items": [...]}`.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.merge_candidates arxiv.json s2.json --out merged.json
```

### `litreview.scaffold_review_md`

Scope:
- Create a literature review markdown scaffold under `results/literature-review/` named
  `results/literature-review/literature-review-<topic-slug>-YYYY-MM-DD.md`.
- Optionally pre-populate the 10 paper headings from a JSON file.

Inputs:
- Positional `topic` (string): used for the report title and filename slug.
- `--works-json` (path, optional): JSON file that is either:
  - object with `items`, or
  - a top-level list of works.
- `--out` (path, optional): override output filename.

Outputs:
- Markdown file in `results/literature-review/` (default naming described above).

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.scaffold_review_md "RAG evaluation for long-context QA" --works-json merged.json
```

### `litreview.scaffold_challenge_md`

Scope:
- Create a challenger report scaffold at `results/challenger/Challenge-<topic-slug>-YYYY-MM-DD.md` with a consistent structure.
- This is a template; the `challenger` skill fills it with cite-heavy critique.

Inputs:
- Positional `idea` (string): the user's idea statement (prefer verbatim).
- `--topic` (string, optional): topic label used for metadata/slugging.
- `--lit-review <path>` (repeatable): path(s) to relevant literature review markdown files.
- `--out` (path): output markdown path (default `results/challenger/Challenge-<topic-slug>-YYYY-MM-DD.md`).

Outputs:
- Markdown file: `results/challenger/Challenge-<topic-slug>-YYYY-MM-DD.md`.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.scaffold_challenge_md "My idea is ..." --topic "RAG evaluation" --lit-review results/literature-review/literature-review-rag-evaluation-2026-04-17.md
```

### `litreview.pdf_text`

Scope:
- Extract text from a local PDF, or download a PDF from a URL and extract text.
- Uses `pymupdf` if installed; otherwise falls back to `pypdf`.

Inputs:
- Positional `input` (path or URL).
- `--engine` (auto|pymupdf|pypdf): extraction backend (default `auto`).
- `--out` (path, optional): output `.txt` path.

Outputs:
- Text file (`.txt`) containing extracted PDF text.

Dependencies:
- Required for URL input: `requests`
- One of: `pymupdf` (preferred) or `pypdf`

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.pdf_text https://arxiv.org/pdf/XXXX.XXXXX.pdf --out paper.txt
```

### `litreview.pdf_images`

Scope:
- Extract embedded images from a local PDF into a directory (useful for quickly inspecting figures).

Inputs:
- Positional `pdf` (path).
- `--out-dir` (path): output directory (default `results/literature-review/pdf-images`).

Outputs:
- A directory of `.png` files, one per extracted image.

Dependencies:
- Required: `pymupdf` (import name `fitz`)

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.pdf_images paper.pdf --out-dir figures
```

### `litreview.openalex_search`

Scope:
- Query the OpenAlex API for candidate papers. Free, no API key required.
- Broader coverage than arXiv/S2 for interdisciplinary and non-CS venues.
- Defaults to filtering to a rolling "last ~3 years" window.

Inputs:
- Positional `query` (string): free-text query.
- `--limit` (int): max results (default 50).
- `--all-years` (flag): disables last-3-years filtering.
- `--email` (string): email for polite-pool rate limit (also reads `OPENALEX_EMAIL` env var).
- `--out` (path): JSON output path (default `results/literature-review/openalex-results.json`).

Outputs:
- JSON file: `{"query": "...", "count": N, "items": [...]}`.
- Each item includes: `title`, `authors`, `year`, `venue`, `published`, `url`, `doi`, `arxiv_id`, `pdf_url`, `citation_count`, `abstract`.

Dependencies:
- Required: `requests`

Canonical usage:
```bash
OPENALEX_EMAIL=you@example.com PYTHONPATH=src python -m litreview.openalex_search "contrastive learning time series" --limit 50 --out oa.json
```

---

### `litreview.abstract_screen`

Scope:
- Fast keyword-based relevance filter for candidate papers. No API calls, no LLM.
- Use between `merge_candidates` and `rank_candidates` to remove clearly off-topic papers.
- Supports required terms (must appear) and excluded terms (causes rejection).

Inputs:
- Positional `input` (path): merged candidates JSON.
- Positional `query` (string): topic string for keyword overlap scoring.
- `--require TERM` (repeatable): substring that MUST appear in title/abstract (supports partial stems like `evaluat`).
- `--exclude TERM` (repeatable): substring that causes rejection if found.
- `--min-score` (float): minimum keyword overlap fraction to accept (default 0.1).
- `--out` (path): accepted papers output (default `results/literature-review/candidates-screened.json`).
- `--rejected-out` (path, optional): write rejected papers for inspection.

Outputs:
- JSON file with accepted papers: `{"query": "...", "count": N, "items": [...]}`.
- Each item gains `_screen_score`, `_screen_overlap`, `_screen_missing_required`, `_screen_hit_excluded`, `_screen_pass` fields.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.abstract_screen merged.json "RAG evaluation QA" \
  --require "retrieval" --require "evaluat" \
  --exclude "survey" --exclude "tutorial" \
  --min-score 0.15 --out screened.json --rejected-out rejected.json
```

---

### `litreview.rank_candidates`

Scope:
- Score and rank screened/merged candidates using a composite of: recency, citation count (log-scaled), and abstract keyword overlap.
- Use after `abstract_screen` to determine the top-N papers to pass to the agent for detailed reading.

Inputs:
- Positional `input` (path): JSON file (object with `items` or bare list).
- Positional `query` (string): topic query for keyword scoring.
- `--top` (int): keep top-N results (default 20).
- `--w-recency` (float): weight for recency score (default 0.35).
- `--w-citation` (float): weight for citation count score (default 0.30).
- `--w-keyword` (float): weight for keyword overlap score (default 0.35).
- `--out` (path): output JSON (default `results/literature-review/candidates-ranked.json`).

Outputs:
- JSON file: `{"query": "...", "total_scored": N, "count": M, "items": [...]}`.
- Each item gains `_score`, `_recency`, `_citation`, `_keyword` score fields.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.rank_candidates screened.json "RAG evaluation long-context" --top 15 --out ranked.json
```

---

### `litreview.export_bib`

Scope:
- Convert a candidate/ranked JSON file to a BibTeX `.bib` file.
- Auto-generates citation keys as `FirstAuthorYYYY_TitleWord`.
- Uses `@article` for papers with a named venue; `@misc` for preprints (arXiv, etc.).
- Use in the `overleaf-paper-writer` workflow to generate a `references.bib` file.

Inputs:
- Positional `input` (path): JSON file (object with `items` or bare list).
- `--out` (path): output `.bib` path (default `results/literature-review/references.bib`).
- `--top` (int, optional): export only first N items.

Outputs:
- A `.bib` file with one `@article` or `@misc` entry per paper.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m litreview.export_bib results/literature-review/candidates-ranked.json \
  --top 15 --out results/paper/references.bib
```

---

### `researchflow.resolve_latest`

Scope:
- Resolve the latest artifact path for one or more workflow stages.
- Use this to locate the newest review, challenger, domain, experiment, codegen, paper, or review artifact without repeated shell globbing.

Inputs:
- Positional `stages...`: aliases from `lit-review | challenger | domain | experiments | codegen | paper-flow | manuscript | review`.
- `--glob` (repeatable): additional raw glob patterns to resolve.
- `--plain` (flag): print only the resolved paths, one per line.

Outputs:
- JSON object mapping alias or glob to the latest matching path by default.
- Plain path list when `--plain` is used.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.resolve_latest lit-review challenger domain experiments paper-flow
```

### `researchflow.append_summary`

Scope:
- Append a prepared markdown entry to `results/summary.md`.
- Keeps the summary-update boilerplate out of the model context.

Inputs:
- `--entry-file` (path, required): markdown file containing one summary entry.
- `--summary` (path): summary markdown path (default `results/summary.md`).

Outputs:
- Updated `results/summary.md`.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.append_summary --entry-file results/tmp/summary-entry.md
```

### `researchflow.scaffold_domain_md`

Scope:
- Create a domain-grounding report scaffold under `results/domain/`.
- Use when the `domain-generator` skill needs the standard output shape without spending tokens on the repeated template.

Inputs:
- Positional `topic` (string): used for title and filename slug.
- `--domain` (string, optional): primary domain label.
- `--idea` (string, optional): current idea statement.
- `--input` (repeatable): upstream artifact path(s).
- `--out` (path, optional): override output filename.

Outputs:
- Markdown file: `results/domain/Domain-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_domain_md "RAG evaluation" \
  --domain "information retrieval" \
  --input results/challenger/Challenge-rag-evaluation-2026-04-25.md
```

### `researchflow.scaffold_experiment_plan_md`

Scope:
- Create an experiment-plan scaffold under `results/experiments/`.
- Use when the `experiment-generator` skill needs the standard claim-to-experiment map and tier structure.

Inputs:
- Positional `topic` (string): used for title and filename slug.
- `--venue` (string, optional): target venue.
- `--input` (repeatable): upstream artifact path(s).
- `--out` (path, optional): override output filename.

Outputs:
- Markdown file: `results/experiments/Experiment-Plan-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_experiment_plan_md "RAG evaluation" \
  --venue "ACL" \
  --input results/domain/Domain-rag-evaluation-2026-04-25.md
```

### `researchflow.scaffold_codegen_project`

Scope:
- Create the standard `results/code/<topic-slug>/` directory layout plus a codegen manifest scaffold.
- Use when the `code-generator` skill needs reusable directory setup instead of regenerating it in-context.

Inputs:
- Positional `topic` (string): used for slugging and the manifest title.
- `--input` (repeatable): upstream artifact path(s).
- `--code-root` (path, optional): override code root directory.
- `--manifest-out` (path, optional): override manifest markdown path.
- `--with-tests` (flag): also create `tests/`.
- `--overwrite` (flag): overwrite scaffold files if they already exist.

Outputs:
- Directory tree under `results/code/<topic-slug>/`.
- Markdown file: `results/code/Codegen-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_codegen_project "RAG evaluation" \
  --input results/experiments/Experiment-Plan-rag-evaluation-2026-04-25.md \
  --with-tests
```

### `researchflow.scaffold_high_level_paper_md`

Scope:
- Create a high-level paper-flow scaffold under `results/paper/`.
- Use when the `high-level-paper-writer` skill needs the outline, status map, and reviewer handoff structure.

Inputs:
- Positional `topic` (string): used for title and filename slug.
- `--input` (repeatable): upstream artifact path(s).
- `--out` (path, optional): override output filename.

Outputs:
- Markdown file: `results/paper/high-level-paper-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_high_level_paper_md "RAG evaluation" \
  --input results/experiments/Experiment-Plan-rag-evaluation-2026-04-25.md
```

### `researchflow.scaffold_overleaf_paper`

Scope:
- Create a manuscript `.tex` scaffold plus a paper handoff markdown file under `results/paper/`.
- Use when the `overleaf-paper-writer` skill needs the standard paper skeleton without retyping LaTeX boilerplate.

Inputs:
- Positional `topic` (string): used for the title and filename slug.
- `--venue` (string, optional): target venue.
- `--input` (repeatable): upstream artifact path(s).
- `--manuscript-out` (path, optional): override manuscript `.tex` path.
- `--handoff-out` (path, optional): override handoff markdown path.

Outputs:
- LaTeX file: `results/paper/manuscript-<topic-slug>-YYYY-MM-DD.tex` by default.
- Markdown file: `results/paper/paper-handoff-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_overleaf_paper "RAG evaluation" \
  --venue "ACL" \
  --input results/paper/high-level-paper-rag-evaluation-2026-04-25.md
```

### `researchflow.scaffold_review_md`

Scope:
- Create a reviewer-style critique scaffold under `results/review/`.
- Use when the `paper-reviewer` skill needs the standard claim-to-result audit and fix-plan structure.

Inputs:
- Positional `topic` (string): used for title and filename slug.
- `--draft` (path, optional): current manuscript or paper-flow path.
- `--input` (repeatable): upstream artifact path(s).
- `--out` (path, optional): override output filename.

Outputs:
- Markdown file: `results/review/Review-<topic-slug>-YYYY-MM-DD.md` by default.

Dependencies:
- None beyond stdlib.

Canonical usage:
```bash
PYTHONPATH=src python -m researchflow.scaffold_review_md "RAG evaluation" \
  --draft results/paper/manuscript-rag-evaluation-2026-04-25.tex \
  --input results/paper/high-level-paper-rag-evaluation-2026-04-25.md
```

---

## When To Extend This Skill

If you add a new reusable script, update this file with:
- Scope (what it does / does not do)
- Inputs (CLI args + env vars)
- Outputs (file formats + schema)
- Dependencies (required vs optional)
- One canonical invocation example
