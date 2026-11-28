#!/usr/bin/env python3
"""Classify each entity host's robots.txt for the engine's user agent (M1-02).

Operator diagnostic, unwired by design. Reads ``entity.primary_domain`` from a SQLite run DB
(opened read-only), GETs only ``/robots.txt`` per host (one request per host, sequential,
10 s timeout), and classifies the result with the engine's own :class:`RobotsPolicy` and
``USER_AGENT`` so it matches crawl behaviour:

- ``blanket``: the homepage and every priority path are disallowed;
- ``partial``: the homepage is disallowed but some priority paths are allowed;
- ``allow``: the homepage is allowed;
- ``unreachable``: no 2xx robots.txt (status class ``4xx``/``5xx``/``3xx``) or a transport
  error (``error``). The engine treats 404/410 as allow-all and other failures as disallow-all.

Per-host detail goes to ``<out>/hosts.jsonl`` (keep ``out`` gitignored, D-003);
``<out>/summary.json`` and stdout carry aggregate counts only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.crawl_policy import RobotsPolicy  # noqa: E402
from adapters.http_transport import get_robots  # noqa: E402
from app.cli import USER_AGENT  # noqa: E402

TIMEOUT_S = 10.0
PRIORITY_PATHS = ("/about", "/services", "/contact", "/company")
CATEGORIES = ("blanket", "partial", "allow", "unreachable")

Fetch = Callable[[str], tuple[int, bytes]]


@dataclass(frozen=True)
class Diagnosis:
    category: str
    status_class: str
    allowed_paths: tuple[str, ...] = ()


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status < 600 else "error"


def classify(url: str, fetch: Fetch) -> Diagnosis:
    """Classify ``url``'s host robots.txt; ``fetch`` is called at most once."""
    seen: dict[str, int] = {}

    def recording_fetch(robots_url: str) -> tuple[int, bytes]:
        status, body = fetch(robots_url)
        seen["status"] = status
        return status, body

    policy = RobotsPolicy(recording_fetch, USER_AGENT)
    home_allowed = policy.allowed(url)  # the only fetch; later calls hit the cache
    status = seen.get("status")
    if status is None or isinstance(status, bool) or not isinstance(status, int):
        return Diagnosis("unreachable", "error")
    if not 200 <= status < 300:
        return Diagnosis("unreachable", _status_class(status))
    if home_allowed:
        return Diagnosis("allow", "2xx")
    base = url.rstrip("/")
    allowed = tuple(p for p in PRIORITY_PATHS if policy.allowed(base + p))
    return Diagnosis("partial" if allowed else "blanket", "2xx", allowed)


def entity_urls(db: Path) -> list[str]:
    """Homepage URLs of all entities, id order, distinct hosts; DB opened read-only."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT primary_domain FROM entity ORDER BY id").fetchall()
    finally:
        con.close()
    hosts = dict.fromkeys(str(r[0]).strip().lower() for r in rows)
    return [f"https://{h}/" for h in hosts if h]


def _live_fetch(robots_url: str) -> tuple[int, bytes]:
    return get_robots(robots_url, user_agent=USER_AGENT, timeout_s=TIMEOUT_S)


def main(argv: Sequence[str] | None = None, *, fetch: Fetch = _live_fetch) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not args.db.is_file():
        print(f"robots_diagnose: no such DB: {args.db}", file=sys.stderr)
        return 2
    urls = entity_urls(args.db)
    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter({c: 0 for c in CATEGORIES})
    status_classes: Counter[str] = Counter()
    partial_paths: Counter[str] = Counter()
    with (args.out / "hosts.jsonl").open("w", encoding="utf-8") as fh:
        for url in urls:
            d = classify(url, fetch)
            counts[d.category] += 1
            status_classes[d.status_class] += 1
            partial_paths.update(d.allowed_paths)
            row = {
                "url": url,
                "category": d.category,
                "status_class": d.status_class,
                "allowed_paths": list(d.allowed_paths),
            }
            fh.write(json.dumps(row) + "\n")
    summary = {
        "user_agent": USER_AGENT,
        "hosts": len(urls),
        "counts": dict(counts),
        "status_classes": dict(sorted(status_classes.items())),
        "partial_allowed_paths": dict(sorted(partial_paths.items())),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
