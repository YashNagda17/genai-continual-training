from __future__ import annotations

import argparse
from pathlib import Path


def extract_images(pdf_path: str, out_dir: str) -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: pymupdf (import name: fitz).") from e

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    n = 0
    for page_i in range(len(doc)):
        page = doc[page_i]
        for img_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out_file = outp / f"page{page_i+1:03d}-img{img_i+1:03d}.png"
            pix.save(out_file.as_posix())
            pix = None
            n += 1
    doc.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract embedded images from a PDF (requires PyMuPDF).")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--out-dir", default="results/literature-review/pdf-images")
    args = ap.parse_args()

    n = extract_images(args.pdf, args.out_dir)
    print(f"Extracted {n} images to {args.out_dir}/")


if __name__ == "__main__":
    main()
