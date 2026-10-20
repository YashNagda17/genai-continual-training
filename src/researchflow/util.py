from __future__ import annotations

import glob
import re
from datetime import date
from pathlib import Path


def today_ymd() -> str:
    return date.today().isoformat()


def slugify_topic(topic: str, max_len: int = 60) -> str:
    text = topic.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "topic"
    return text[:max_len].strip("-") or "topic"


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    Path(path).write_text(text, encoding="utf-8")


def artifact_path(subdir: str, prefix: str, topic: str, ext: str = ".md") -> str:
    slug = slugify_topic(topic)
    return f"results/{subdir}/{prefix}-{slug}-{today_ymd()}{ext}"


def latest_matching(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def render_path_list(paths: list[str]) -> str:
    if not paths:
        return "- None provided"
    return "\n".join(f"- {p}" for p in paths)


def append_summary_entry(summary_path: str, entry_text: str) -> None:
    entry = entry_text.strip()
    if not entry:
        raise SystemExit("Summary entry is empty.")

    block = entry if entry.startswith("---") else f"---\n\n{entry}"
    target = Path(summary_path)
    if not target.exists():
        write_text(summary_path, block.rstrip() + "\n")
        return

    existing = target.read_text(encoding="utf-8").rstrip()
    target.write_text(existing + "\n\n" + block.rstrip() + "\n", encoding="utf-8")

