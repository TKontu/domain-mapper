"""Tests for ``landscape search`` (IX-04); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.chunk_store import ChunkRow
from core.keyword_search import KeywordHit, search_chunks
from core.runs import Run, create_all
from core.snapshot_store import SnapshotRow


@pytest.fixture
def db_url(tmp_path, monkeypatch) -> str:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no stray .env
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    return url


@pytest.fixture
def engine(db_url) -> Engine:
    engine = create_engine(db_url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def add_snapshot(engine: Engine, entity_id: int | None) -> int:
    row = SnapshotRow(
        entity_id=entity_id,
        url="https://example.test/page",
        final_url="https://example.test/page",
        content_type="text/html",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_kind="web",
    )
    with Session(engine) as session, session.begin():
        session.add(row)
        session.flush()
        return row.id


def add_chunks(
    engine: Engine,
    snapshot_id: int,
    texts: list[str],
    headings: list[tuple[str, ...]] | None = None,
) -> None:
    with Session(engine) as session, session.begin():
        for seq, text in enumerate(texts):
            session.add(
                ChunkRow(
                    snapshot_id=snapshot_id,
                    seq=seq,
                    text=text,
                    lang=None,
                    page=None,
                    heading_path=list(headings[seq]) if headings else [],
                    char_start=0,
                    char_end=len(text),
                )
            )


def invoke(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["search", *args])


def expected_lines(hits: list[KeywordHit]) -> list[str]:
    """The spec line format: score, ids, seq, then the heading tail when nonempty."""
    lines = []
    for hit in hits:
        line = f"{hit.score:.4f} chunk {hit.chunk_id} snapshot {hit.snapshot_id} seq {hit.seq}"
        if hit.heading_path:
            line = f"{line} {' > '.join(hit.heading_path)}"
        lines.append(line)
    return lines


def run_rows(engine: Engine) -> list[tuple[str, str]]:
    with Session(engine) as session:
        return [(r.stage, r.status) for r in session.scalars(sa.select(Run).order_by(Run.id))]


def test_prints_one_line_per_hit_in_search_chunks_order(engine) -> None:
    snap = add_snapshot(engine, None)
    add_chunks(
        engine,
        snap,
        ["apple banana cherry", "apple date elderberry", "other text"],
        headings=[("Guide",), ("Guide", "Chapters"), ()],
    )

    result = invoke("apple")

    assert result.exit_code == 0, result.output
    hits = search_chunks(engine, "apple")
    assert hits
    assert result.stdout.splitlines() == expected_lines(hits)
    assert run_rows(engine) == []  # read-only: no run row, nothing written


def test_line_omits_trailing_space_without_heading_path(engine) -> None:
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["apple only here"], headings=[()])

    result = invoke("apple")

    assert result.exit_code == 0, result.output
    hits = search_chunks(engine, "apple")
    line = result.stdout.splitlines()[0]
    assert line == (
        f"{hits[0].score:.4f} chunk {hits[0].chunk_id} snapshot {hits[0].snapshot_id} seq 0"
    )


def test_entity_filter_passed_through(engine) -> None:
    snap1 = add_snapshot(engine, 1)
    snap2 = add_snapshot(engine, 2)
    add_chunks(engine, snap1, ["widget alpha", "widget beta"])
    add_chunks(engine, snap2, ["widget gamma"])

    result = invoke("widget", "--entity", "1")

    assert result.exit_code == 0, result.output
    hits = search_chunks(engine, "widget", entity_id=1)
    assert result.stdout.splitlines() == expected_lines(hits)
    # "0.5 x chunk C snapshot S seq N": the snapshot id is the field after "snapshot".
    assert {line.split(" snapshot ")[1].split(" ")[0] for line in result.stdout.splitlines()} == {
        str(snap1)
    }


def test_limit_passed_through(engine) -> None:
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["same term", "same term", "same term"])

    result = invoke("same", "--limit", "2")

    assert result.exit_code == 0, result.output
    hits = search_chunks(engine, "same", limit=2)
    assert result.stdout.splitlines() == expected_lines(hits)
    assert len(hits) == 2


def test_no_hits_prints_no_hits_and_exits_0(engine) -> None:
    add_snapshot(engine, None)

    result = invoke("zzz_no_such_term")

    assert result.exit_code == 0, result.output
    assert result.stdout == "no hits\n"


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_or_whitespace_query_exits_2(engine, query) -> None:
    result = invoke(query)

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert "error:" in result.stderr


@pytest.mark.parametrize("args", [["--limit", "0"], ["--limit=-1"]])
def test_limit_below_one_exits_2(engine, args) -> None:
    result = invoke("apple", *args)

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert "error:" in result.stderr


def test_validation_happens_before_opening_the_database(monkeypatch) -> None:
    opened: list[object] = []

    def spy(settings):
        opened.append(settings)
        raise AssertionError("database opened")

    monkeypatch.setattr(cli, "_open_engine", spy)

    result = invoke("")

    assert result.exit_code == 2
    assert opened == []


class FakeEngine:
    """Stand-in for :class:`~sqlalchemy.engine.Engine`; only ``dispose`` matters."""

    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_search_disposes_engine_when_search_succeeds(monkeypatch) -> None:
    fake = FakeEngine()
    monkeypatch.setattr(cli, "_open_engine", lambda settings: fake)
    monkeypatch.setattr(cli, "search_chunks", lambda engine, query, **kwargs: [])

    result = invoke("term")

    assert result.exit_code == 0, result.output
    assert fake.disposed


def test_search_disposes_engine_when_search_raises(monkeypatch) -> None:
    fake = FakeEngine()
    monkeypatch.setattr(cli, "_open_engine", lambda settings: fake)

    def boom(engine, query, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "search_chunks", boom)

    result = invoke("term")

    assert result.exit_code != 0
    assert fake.disposed
