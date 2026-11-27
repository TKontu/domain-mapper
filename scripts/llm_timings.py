"""LM-06: per-call timing summary of ``llm_call`` rows in a SQLite file.

Planner/operator tool for LM-04 lane records, unwired by design (no production
caller). Opens the database read-only and prints, per purpose, calls, errors
(``status != 'ok'``) and p50/p95/max ``latency_ms``; overall, the wall span,
sum of latencies and peak concurrency. Each call's interval is
``[created_at - latency_ms, created_at]`` because ``created_at`` is stamped when
the row is written after the call (core/llm_calls.py ``log_llm_call``).
Intervals that only touch do not overlap.

Rows are streamed ordered by ``created_at``; latencies are kept per purpose for
percentiles, so memory is O(rows).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple


class CallRow(NamedTuple):
    purpose: str
    status: str
    latency_ms: int
    created_at: datetime


class InputError(Exception):
    """The database file or its llm_call table is unusable."""


def _parse_ts(value: str) -> datetime:
    """Parse a stored timestamp as naive UTC (SQLite drops the offset on write)."""
    ts = datetime.fromisoformat(value)
    offset = ts.utcoffset() or timedelta(0)
    return ts.replace(tzinfo=None) - offset


def read_rows(path: Path, run_id: int | None = None) -> Iterator[CallRow]:
    """Stream ``llm_call`` rows ordered by ``created_at`` from a read-only SQLite file."""
    if not Path(path).is_file():
        raise InputError(f"no such database file: {path}")
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        sql = "SELECT purpose, status, latency_ms, created_at FROM llm_call"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        try:
            cursor = conn.execute(sql + " ORDER BY created_at, id", params)
        except sqlite3.DatabaseError as exc:
            raise InputError(f"cannot read llm_call from {path}: {exc}") from exc
        for purpose, status, latency, created in cursor:
            yield CallRow(purpose, status, int(latency or 0), _parse_ts(str(created)))
    finally:
        conn.close()


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of a non-empty sorted list."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def summarise(rows: Iterable[CallRow]) -> dict[str, Any]:
    """Summarise rows into per-purpose and overall timing figures (JSON-ready)."""
    latencies: dict[str, list[int]] = {}
    errors: dict[str, int] = {}
    events: list[tuple[datetime, int]] = []  # (time, +1 start / -1 end)
    total = 0
    lat_sum = 0
    for row in rows:
        total += 1
        lat_sum += row.latency_ms
        latencies.setdefault(row.purpose, []).append(row.latency_ms)
        errors[row.purpose] = errors.get(row.purpose, 0) + (row.status != "ok")
        start = row.created_at - timedelta(milliseconds=row.latency_ms)
        events.append((start, 1))
        events.append((row.created_at, -1))
    # Ends sort before starts at the same instant, so touching intervals do not overlap.
    events.sort()
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    wall = 0
    if events:
        wall = round((events[-1][0] - events[0][0]).total_seconds() * 1000)
    purposes = {}
    for purpose in sorted(latencies):
        values = sorted(latencies[purpose])
        purposes[purpose] = {
            "calls": len(values),
            "errors": errors[purpose],
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": values[-1],
        }
    return {
        "purposes": purposes,
        "overall": {
            "calls": total,
            "wall_span_ms": wall,
            "latency_sum_ms": lat_sum,
            "peak_concurrency": peak,
        },
    }


def _format_text(summary: dict[str, Any]) -> str:
    lines = [f"{'purpose':<24} calls errors p50_ms p95_ms max_ms"]
    for name, p in summary["purposes"].items():
        lines.append(
            f"{name:<24} {p['calls']:>5} {p['errors']:>6} {p['p50_ms']:>6} "
            f"{p['p95_ms']:>6} {p['max_ms']:>6}"
        )
    lines.extend(f"{key}: {value}" for key, value in summary["overall"].items())
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = summarise(read_rows(args.db_path, args.run_id))
    except InputError as exc:
        print(f"llm_timings: {exc}".replace("\n", " "), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2) if args.json else _format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
