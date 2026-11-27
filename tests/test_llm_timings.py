"""Offline tests for scripts/llm_timings.py (LM-06)."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import core.llm_calls  # noqa: F401  (registers llm_call)
from core.llm_calls import LlmCall
from core.runs import Run, create_all

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _load():
    spec = importlib.util.spec_from_file_location(
        "llm_timings", ROOT / "scripts" / "llm_timings.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _db(tmp_path: Path, calls: list[tuple[str, str, int, float]], runs: int = 1) -> Path:
    """Build a SQLite llm_call table; calls are (purpose[@run], status, latency_ms, end_s)."""
    path = tmp_path / "t.db"
    engine = create_engine(f"sqlite:///{path}")
    create_all(engine)
    with Session(engine) as s:
        ids = []
        for _ in range(runs):
            r = Run(stage="assess")
            s.add(r)
            s.flush()
            ids.append(r.id)
        for purpose, status, lat, end in calls:
            name, _, run = purpose.partition("@")
            s.add(
                LlmCall(
                    run_id=ids[int(run) - 1] if run else ids[0],
                    model="m",
                    purpose=name,
                    prompt_hash="0" * 64,
                    latency_ms=lat,
                    status=status,
                    created_at=T0 + timedelta(seconds=end),
                )
            )
        s.commit()
    engine.dispose()
    return path


def test_peak_full_overlap_sequential_and_touching(tmp_path):
    (tmp_path / "a").mkdir()
    full = _db(tmp_path / "a", [("s", "ok", 1000, 1.0)] * 3)
    assert mod.summarise(mod.read_rows(full))["overall"]["peak_concurrency"] == 3
    (tmp_path / "b").mkdir()
    seq = _db(tmp_path / "b", [("s", "ok", 1000, 1.0), ("s", "ok", 1000, 3.0)])
    assert mod.summarise(mod.read_rows(seq))["overall"]["peak_concurrency"] == 1
    (tmp_path / "c").mkdir()
    touch = _db(tmp_path / "c", [("s", "ok", 1000, 1.0), ("s", "ok", 1000, 2.0)])
    out = mod.summarise(mod.read_rows(touch))["overall"]
    assert out["peak_concurrency"] == 1
    assert out["wall_span_ms"] == 2000
    assert out["latency_sum_ms"] == 2000


def test_per_purpose_counts_errors_percentiles(tmp_path):
    calls = [("screen", "ok", lat, float(i)) for i, lat in enumerate(range(100, 1100, 100))]
    calls += [("verify", "error", 50, 20.0), ("verify", "ok", 70, 21.0)]
    summary = mod.summarise(mod.read_rows(_db(tmp_path, calls)))
    sc = summary["purposes"]["screen"]
    assert (sc["calls"], sc["errors"], sc["p50_ms"], sc["p95_ms"], sc["max_ms"]) == (
        10,
        0,
        500,
        1000,
        1000,
    )
    v = summary["purposes"]["verify"]
    assert (v["calls"], v["errors"], v["p50_ms"], v["p95_ms"], v["max_ms"]) == (2, 1, 50, 70, 70)
    assert summary["overall"]["calls"] == 12


def test_run_id_filter_and_json(tmp_path, capsys):
    path = _db(tmp_path, [("a", "ok", 100, 1.0), ("b@2", "ok", 200, 2.0)], runs=2)
    assert mod.main([str(path), "--run-id", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data["purposes"]) == ["b"]
    assert data == mod.summarise(mod.read_rows(path, run_id=2))
    assert mod.main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "a" in text and "peak_concurrency" in text


@pytest.mark.parametrize("kind", ["missing", "notable"])
def test_bad_input_exit_2(tmp_path, capsys, kind):
    path = tmp_path / "x.db"
    if kind == "notable":
        import sqlite3

        sqlite3.connect(path).execute("create table other (x int)").connection.close()
    assert mod.main([str(path)]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.strip()
    if kind == "missing":
        assert not path.exists()
