from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from .util import ensure_parent_dir, http_get


def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), flags=re.I))


def download_pdf(url: str, out_path: str) -> str:
    data = http_get(url, timeout_s=60)
    Path(out_path).write_bytes(data)
    return out_path


def extract_text_pymupdf(pdf_path: str) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pymupdf (import name: fitz).") from e

    doc = fitz.open(pdf_path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


def extract_text_pypdf(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pypdf.") from e

    reader = PdfReader(pdf_path)
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_text(pdf_path: str, engine: str = "auto") -> str:
    if engine == "pymupdf":
        return extract_text_pymupdf(pdf_path)
    if engine == "pypdf":
        return extract_text_pypdf(pdf_path)
    if engine != "auto":
        raise SystemExit(f"Unknown engine: {engine} (expected auto|pymupdf|pypdf)")

    try:
        return extract_text_pymupdf(pdf_path)
    except Exception:
        return extract_text_pypdf(pdf_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download (optional) and extract text from a PDF.")
    ap.add_argument("input", help="PDF path or URL")
    ap.add_argument("--engine", default="auto", help="auto|pymupdf|pypdf")
    ap.add_argument("--out", default=None, help="Output .txt path (default: <pdf_basename>.txt)")
    args = ap.parse_args()

    inp = args.input
    tmp_pdf = None
    if _is_url(inp):
        fd, tmp_pdf = tempfile.mkstemp(prefix="litreview-", suffix=".pdf")
        os.close(fd)
        download_pdf(inp, tmp_pdf)
        pdf_path = tmp_pdf
    else:
        pdf_path = inp

    try:
        text = extract_text(pdf_path, engine=args.engine)
    finally:
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)

    out_path = args.out
    if out_path is None:
        base = Path(inp).name
        if base.lower().endswith(".pdf"):
            base = base[:-4]
        out_path = f"results/literature-review/{base}.txt"

    ensure_parent_dir(out_path)
    Path(out_path).write_text(text, encoding="utf-8", errors="ignore")
    print(f"Wrote text to {out_path}")


if __name__ == "__main__":
    main()
