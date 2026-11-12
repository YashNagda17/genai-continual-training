"""Fetch English and Hindi XL-Sum into canonical JSONL."""

from pathlib import Path
import json
import sys
import tarfile

from huggingface_hub import snapshot_download

from .dataset_instruction import SPLITS

XLSUM_LANGUAGES = ("english", "hindi")
XLSUM_SPLIT_FILES = (("train", "train"), ("validation", "val"), ("test", "test"))


def xlsum_archive_rows(archive_path, language):
    """Read one language tar. Official `val` is stored as `validation`."""
    rows, dropped, seen = [], 0, set()
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        for split, suffix in XLSUM_SPLIT_FILES:
            name = f"{language}_{suffix}.jsonl"
            member = members.get(name)
            if member is None:
                raise ValueError(f"{archive_path} is missing {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"{archive_path} could not read {name}")
            for line in handle:
                data = json.loads(line)
                article_id = str(data.get("id", "")).strip()
                article = str(data.get("text", "")).strip()
                summary = str(data.get("summary", "")).strip()
                if not article_id or not article or not summary:
                    dropped += 1
                    continue
                row_id = f"{language}/{article_id}"
                if row_id in seen:
                    raise ValueError(f"Duplicate XL-Sum id {row_id}")
                seen.add(row_id)
                rows.append({"id": row_id, "task": language, "split": split, "language": language,
                             "text": article, "summary": summary})
    return rows, dropped


def publish(root, dataset_id, source, revision, rows, extra):
    """Write canonical JSONL and its manifest. An existing directory is kept."""
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if counts["train"] == 0:
        raise ValueError(f"{dataset_id} produced no training records")
    task_counts = {}
    for row in rows:
        bucket = task_counts.setdefault(row["task"], {split: 0 for split in SPLITS})
        bucket[row["split"]] += 1
    out = Path(root) / dataset_id / revision
    records = out / "records.jsonl"
    manifest_path = out / "manifest.json"
    if records.exists() and manifest_path.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with records.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset_id": dataset_id, "source": source, "revision": revision,
                "counts": counts, "task_counts": task_counts,
                "task_order": list(dict.fromkeys(row["task"] for row in rows)),
                "records": len(rows), **extra}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_source(directory, dataset_cls):
    """Load a fetched directory into the given dataset class. This does not download."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dataset = dataset_cls()
    raw, _ = dataset.load({"path": directory / "records.jsonl", "dataset_id": manifest["dataset_id"],
                           "revision": manifest.get("revision", manifest["dataset_id"])})
    dataset.manifest["source"] = dict(manifest)
    return dataset, raw


class DatasetSource:
    """Download one Hub snapshot and convert it. Subclasses own the file layout."""

    dataset_id = ""
    source = ""
    patterns = ()
    extra = {}

    def download(self, root, token=None, revision="main"):
        cache = Path(root) / self.dataset_id / "hub"
        cache.mkdir(parents=True, exist_ok=True)
        kwargs = {"repo_id": self.source, "repo_type": "dataset", "revision": revision,
                  "cache_dir": str(cache), "allow_patterns": list(self.patterns)}
        if token:
            kwargs["token"] = token
        return Path(snapshot_download(**kwargs))

    def convert(self, snapshot):
        raise NotImplementedError

    def fetch(self, root, token=None, revision="main"):
        snapshot = self.download(root, token, revision)
        rows = self.convert(snapshot)
        return publish(root, self.dataset_id, self.source, snapshot.name,
                       rows, {**self.extra, **getattr(self, "report", {})})


class XLSumSource(DatasetSource):
    """English and Hindi XL-Sum with official split membership."""

    dataset_id = "xlsum"
    source = "csebuetnlp/xlsum"
    patterns = tuple(f"data/{language}_XLSum_v2.0.tar.bz2" for language in XLSUM_LANGUAGES)
    extra = {"protocol": "official", "license": "cc-by-nc-sa-4.0", "schema": "summarization",
             "languages": list(XLSUM_LANGUAGES),
             "usage": "Official splits. The article is the input and the summary is the target."}

    def convert(self, snapshot):
        rows, dropped = [], 0
        for language in XLSUM_LANGUAGES:
            archive = snapshot / "data" / f"{language}_XLSum_v2.0.tar.bz2"
            if not archive.is_file():
                raise ValueError(f"XL-Sum snapshot is missing {archive.name}")
            language_rows, language_dropped = xlsum_archive_rows(archive, language)
            rows.extend(language_rows)
            dropped += language_dropped
        self.report = {"empty_dropped": dropped}
        return rows


SOURCES = {"xlsum": XLSumSource}


def fetch_source(name, root, token=None, revision="main"):
    """Fetch the current plan's source."""
    try:
        source = SOURCES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown dataset source {name}") from error
    return source.fetch(root, token=token, revision=revision)


def main(argv):
    name = argv[1]
    root = Path(argv[2]) if len(argv) > 2 else Path("src/datasets/data")
    token = argv[3] if len(argv) > 3 and argv[3] else None
    names = list(SOURCES) if name == "all" else [name]
    for item in names:
        print(fetch_source(item, root, token=token))


if __name__ == "__main__":
    main(sys.argv)
