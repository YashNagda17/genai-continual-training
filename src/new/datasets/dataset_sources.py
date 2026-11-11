"""Fetch English and Hindi XL-Sum into one canonical JSONL file.

Run: python -m src.new.datasets.dataset_sources"""

from pathlib import Path
import json
import tarfile

from huggingface_hub import snapshot_download

LANGUAGES = ("english", "hindi")
DATA = Path(__file__).resolve().parents[2] / "datasets" / "data" / "xlsum"


def xlsum_directory():
    found = [path for path in DATA.iterdir() if (path / "records.jsonl").is_file()] if DATA.is_dir() else []
    if len(found) != 1:
        raise SystemExit("XL-Sum is not fetched. Run python -m src.new.datasets.dataset_sources")
    return found[0]


def fetch(root=DATA):
    """Official splits; the article is the input and the summary is the target."""
    snapshot = Path(snapshot_download("csebuetnlp/xlsum", repo_type="dataset", cache_dir=str(root / "hub"),
                                      allow_patterns=[f"data/{l}_XLSum_v2.0.tar.bz2" for l in LANGUAGES]))
    out = root / snapshot.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for language in LANGUAGES:
            with tarfile.open(snapshot / "data" / f"{language}_XLSum_v2.0.tar.bz2", "r:bz2") as archive:
                members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
                for split, suffix in (("train", "train"), ("validation", "val"), ("test", "test")):
                    for line in archive.extractfile(members[f"{language}_{suffix}.jsonl"]):
                        row = json.loads(line)
                        text, summary = str(row.get("text", "")).strip(), str(row.get("summary", "")).strip()
                        if str(row.get("id", "")).strip() and text and summary:
                            handle.write(json.dumps({"id": f"{language}/{row['id']}", "split": split,
                                                     "language": language, "text": text, "summary": summary},
                                                    ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    print(fetch())
