"""Tests for ``landscape reassess-plan`` (SD-02) and the SD-01 run-selection nits in both
subcommands; SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import core.snapshot_store  # noqa: F401  (registers the snapshot table)
from adapters.fetcher import FetchResult
from app import cli
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.chunk_store import ChunkRow
from core.runs import Run, create_all
from core.snapshot_store import record_snapshot
from core.verify import Route

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def add_run(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="crawl", status="finished", started_at=T0)
        session.add(run)
        session.flush()
        return run.id


def snap_chunk(engine: Engine, run_id: int, url: str, body: bytes) -> int:
    result = FetchResult(
        final_url=url,
        status=200,
        content_type="text/html",
        body_bytes=body,
        headers={},
        fetched_at=T0,
    )
    sid = record_snapshot(
        engine, entity_id=1, url=url, result=result, source_kind="crawl", run_id=run_id
    )
    with Session(engine) as session, session.begin():
        row = ChunkRow(
            snapshot_id=sid,
            seq=0,
            text="t",
            lang=None,
            page=None,
            heading_path=[],
            char_start=0,
            char_end=1,
        )
        session.add(row)
        session.flush()
        return row.id


def assess(engine: Engine, run_id: int, criterion: str, chunk_ids: list[int]) -> int:
    quotes = tuple(Quote(c, "q", True, "exact", (0, 1)) for c in chunk_ids)
    ext = Extraction(status="strong", quotes=quotes, rationale="r", checks_passed=True, failures=())
    return record_assessment(
        engine,
        entity_id=1,
        criterion_id=criterion,
        extraction=ext,
        route=Route("accept", "high", "ok"),
        rubric_version="v1",
        extractor_model="x",
        verifier_model="y",
        run_id=run_id,
    )


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


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def two_runs(engine: Engine) -> tuple[int, int]:
    base, head = add_run(engine), add_run(engine)
    snap_chunk(engine, base, "https://e/a", b"1")
    snap_chunk(engine, head, "https://e/a", b"2")
    return base, head


def test_prints_stale_lines_and_total(engine):
    base, head = add_run(engine), add_run(engine)
    ca = snap_chunk(engine, base, "https://e/a", b"old")
    cs = snap_chunk(engine, base, "https://e/s", b"same")
    cg = snap_chunk(engine, base, "https://e/gone", b"x")
    snap_chunk(engine, head, "https://e/a", b"new")
    snap_chunk(engine, head, "https://e/s", b"same")
    a2 = assess(engine, base, "c2", [ca])
    a1 = assess(engine, base, "c1", [cg, ca, cs])
    assess(engine, base, "c3", [cs])
    result = invoke("reassess-plan", "--entity", "1")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"c1\tv1\t{a1}\tchanged:https://e/a,removed:https://e/gone",
        f"c2\tv1\t{a2}\tchanged:https://e/a",
        "total stale 2 current 3",
    ]
    explicit = invoke(
        "reassess-plan", "--entity", "1", "--base-run", str(base), "--head-run", str(head)
    )
    assert explicit.stdout == result.stdout


@pytest.mark.parametrize("command", ["snapshot-diff", "reassess-plan"])
def test_equal_runs_exit_2(engine, command):
    base, _head = two_runs(engine)
    result = invoke(command, "--entity", "1", "--base-run", str(base), "--head-run", str(base))
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: base and head run are the same"
    assert result.stdout == ""


@pytest.mark.parametrize("command", ["snapshot-diff", "reassess-plan"])
def test_base_alone_at_newest_exit_2(engine, command):
    _base, head = two_runs(engine)
    result = invoke(command, "--entity", "1", "--base-run", str(head))
    assert result.exit_code == 2
    assert result.stderr.strip() == (
        f"error: no run newer than base run {head} has snapshots for entity 1"
    )
    assert result.stdout == ""


def test_too_few_runs_exit_2(engine):
    snap_chunk(engine, add_run(engine), "https://e/a", b"1")
    result = invoke("reassess-plan", "--entity", "1")
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: entity 1 has fewer than two runs with snapshots"


def test_database_unavailable_exits_2(engine, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OperationalError("select", {}, Exception("down"))

    monkeypatch.setattr(cli, "stale_assessments", boom)
    result = invoke("reassess-plan", "--entity", "1")
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: database unavailable (OperationalError)"
