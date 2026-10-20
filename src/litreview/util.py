from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


def today_ymd() -> str:
    return date.today().isoformat()


def three_year_window_start(d: date | None = None) -> date:
    # "Last 3 years" is ambiguous (calendar years vs rolling window).
    # Use a rolling window of ~3 years.
    d = d or date.today()
    return d - timedelta(days=365 * 3)


def in_last_3_years(published: date | None, now: date | None = None) -> bool:
    if published is None:
        return False
    now = now or date.today()
    return published >= three_year_window_start(now) and published <= now


def slugify_topic(topic: str, max_len: int = 60) -> str:
    s = topic.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "topic"
    return s[:max_len].strip("-") or "topic"


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_date(s: str | None) -> date | None:
    if not s:
        return None
    # Common forms: "2025-01-12", "2025-01-12T...", "2025"
    s = s.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m-%d").date()
        if re.fullmatch(r"\d{4}", s):
            return date(int(s), 1, 1)
        # ISO-ish: 2025-01-12T10:11:12Z
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        return None


def require_requests():
    try:
        import requests  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: requests. Install it (e.g. add `requests` to requirements.txt)."
        ) from e


def http_get(url: str, *, timeout_s: int = 30, headers: dict[str, str] | None = None) -> bytes:
    require_requests()
    import requests

    hdrs = {"User-Agent": os.environ.get("LITREVIEW_UA", "litreview/0.1")}
    if headers:
        hdrs.update(headers)
    r = requests.get(url, timeout=timeout_s, headers=hdrs)
    r.raise_for_status()
    return r.content


def uniq_by(items: Iterable[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        k = tuple((it.get(x) or "").strip().lower() for x in keys)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


@dataclass(frozen=True)
class Work:
    title: str
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    published_date: str | None = None  # ISO date string if known
