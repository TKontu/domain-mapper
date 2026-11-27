"""Tests for ``landscape cost-report`` (CO-01); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.llm_calls import CostSummaryRow, record_llm_call, summarise_llm_calls
from core.runs import Run, create_all


@pytest.fixture
def engine(tmp_path, monkeypatch) -> Engine:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    engine = create_engine(url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def add_run(engine: Engine, stage: str) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage=stage, status="finished", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def call(engine, run_id, model="m", purpose="p", tin=10, tout=5, cost=0.001, ms=100, ok=True):
    record_llm_call(
        engine,
        run_id=run_id,
        model=model,
        purpose=purpose,
        prompt="x",
        tokens_in=tin,
        tokens_out=tout,
        cost=cost,
        latency_ms=ms,
        status="ok" if ok else "error",
        error=None if ok else "LlmError: boom",
    )


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["cost-report", *args])


def test_summarise_groups_and_sorts(engine):
    screen = add_run(engine, "screen")
    assess = add_run(engine, "assess")
    call(engine, screen, model="b", purpose="screen", ms=100)
    call(engine, screen, model="b", purpose="screen", ms=300, ok=False, cost=0.0)
    call(engine, screen, model="a", purpose="screen", ms=50)
    call(engine, assess, model="z", purpose="extract", tin=7, tout=3, cost=0.5, ms=10)
    with Session(engine) as session:
        rows = summarise_llm_calls(session)
    assert [(r.stage, r.purpose, r.model) for r in rows] == [
        ("assess", "extract", "z"),
        ("screen", "screen", "a"),
        ("screen", "screen", "b"),
    ]
    b = rows[2]
    assert b == CostSummaryRow(
        stage="screen",
        purpose="screen",
        model="b",
        calls=2,
        ok=1,
        error=1,
        tokens_in=20,
        tokens_out=10,
        cost=pytest.approx(0.001),
        mean_latency_ms=pytest.approx(200.0),
        max_latency_ms=300,
    )


def test_summarise_filters_by_run(engine):
    r1 = add_run(engine, "screen")
    r2 = add_run(engine, "screen")
    call(engine, r1)
    call(engine, r2)
    call(engine, r2)
    with Session(engine) as session:
        rows = summarise_llm_calls(session, run_id=r2)
    assert [r.calls for r in rows] == [2]


def test_summarise_uses_one_group_by_query(engine):
    run = add_run(engine, "screen")
    for _ in range(5):
        call(engine, run)
    statements: list[str] = []
    event.listen(engine, "before_cursor_execute", lambda *a: statements.append(a[2]))
    with Session(engine) as session:
        summarise_llm_calls(session)
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1
    assert "GROUP BY" in selects[0].upper()


def test_cli_prints_lines_and_total(engine):
    run = add_run(engine, "screen")
    call(engine, run, model="m1", purpose="screen", tin=10, tout=5, cost=0.25, ms=100)
    call(engine, run, model="m1", purpose="screen", tin=1, tout=1, cost=0.0, ms=300, ok=False)
    result = invoke()
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines == [
        "stage screen purpose screen model m1 calls 2 ok 1 error 1 tokens_in 11"
        " tokens_out 6 cost 0.250000 mean_latency_ms 200.0 max_latency_ms 300",
        "total calls 2 ok 1 error 1 tokens_in 11 tokens_out 6 cost 0.250000"
        " mean_latency_ms 200.0 max_latency_ms 300",
    ]


def test_cli_total_weights_mean_by_calls(engine):
    run = add_run(engine, "screen")
    call(engine, run, model="a", ms=100)
    call(engine, run, model="b", ms=400)
    call(engine, run, model="b", ms=400)
    result = invoke()
    assert result.exit_code == 0
    assert (
        result.output.strip().splitlines()[-1].endswith("mean_latency_ms 300.0 max_latency_ms 400")
    )


def test_cli_zero_rows_prints_only_total(engine):
    result = invoke()
    assert result.exit_code == 0
    assert result.output.strip() == (
        "total calls 0 ok 0 error 0 tokens_in 0 tokens_out 0 cost 0.000000"
        " mean_latency_ms 0.0 max_latency_ms 0"
    )


def test_cli_run_filter(engine):
    r1 = add_run(engine, "screen")
    r2 = add_run(engine, "assess")
    call(engine, r1)
    call(engine, r2)
    result = invoke("--run", str(r2))
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("stage assess ")
    assert lines[1].startswith("total calls 1 ")


def test_cli_unknown_run_exits_2(engine):
    result = invoke("--run", "999")
    assert result.exit_code == 2
    assert "error: run 999 not found" in result.output


def test_cli_database_error_exits_2(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DM_DATABASE_URL", f"sqlite:///{tmp_path}/missing/dir/dm.sqlite")
    result = invoke()
    assert result.exit_code == 2
    assert "error: database unavailable (OperationalError)" in result.output
    assert "missing" not in result.output
