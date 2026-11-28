"""Tests for ``landscape snapshot-diff`` (SD-01); SQLite under a temp dir, CLI via CliRunner."""

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
from core.runs import Run, create_all
from core.snapshot_store import record_snapshot

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def add_run(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="crawl", status="finished", started_at=T0)
        session.add(run)
        session.flush()
        return run.id


def snap(engine: Engine, run_id: int, url: str, body: bytes) -> None:
    result = FetchResult(
        final_url=url,
        status=200,
        content_type="text/html",
        body_bytes=body,
        headers={},
        fetched_at=T0,
    )
    record_snapshot(engine, entity_id=1, url=url, result=result, source_kind="crawl", run_id=run_id)


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
    return CliRunner().invoke(cli.app, ["snapshot-diff", *args])


def test_prints_lines_and_total(engine):
    base, head = add_run(engine), add_run(engine)
    snap(engine, base, "https://e/b", b"same")
    snap(engine, base, "https://e/a", b"old")
    snap(engine, base, "https://e/gone", b"x")
    snap(engine, head, "https://e/a", b"new")
    snap(engine, head, "https://e/b", b"same")
    snap(engine, head, "https://e/new", b"y")
    result = invoke("--entity", "1")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "changed\thttps://e/a",
        "unchanged\thttps://e/b",
        "removed\thttps://e/gone",
        "added\thttps://e/new",
        "total changed 1 unchanged 1 added 1 removed 1",
    ]


def test_explicit_runs(engine):
    r1, r2, r3 = add_run(engine), add_run(engine), add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://e/a", b"2")
    snap(engine, r3, "https://e/a", b"1")
    result = invoke("--entity", "1", "--base-run", str(r1), "--head-run", str(r3))
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == "unchanged\thttps://e/a"


def test_too_few_runs_exits_2(engine):
    snap(engine, add_run(engine), "https://e/a", b"1")
    result = invoke("--entity", "1")
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: entity 1 has fewer than two runs with snapshots"
    assert result.stdout == ""


def test_named_run_without_snapshots_exits_2(engine):
    r1 = add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    result = invoke("--entity", "1", "--base-run", str(r1), "--head-run", "42")
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: run 42 has no snapshots for entity 1"


def test_database_unavailable_exits_2(engine, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OperationalError("select", {}, Exception("down"))

    monkeypatch.setattr(cli, "diff_entity_runs", boom)
    result = invoke("--entity", "1")
    assert result.exit_code == 2
    assert result.stderr.strip() == "error: database unavailable (OperationalError)"
