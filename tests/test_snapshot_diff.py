"""Tests for ``core.snapshot_diff`` (SD-01); offline SQLite under a temp dir."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

import core.snapshot_store  # noqa: F401  (registers the snapshot table)
from adapters.fetcher import FetchResult
from core.runs import Run, create_all
from core.snapshot_diff import SnapshotDiffError, UrlDiff, diff_entity_runs
from core.snapshot_store import record_snapshot

T0 = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture
def engine(tmp_path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path}/dm.sqlite")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def add_run(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="crawl", status="finished", started_at=T0)
        session.add(run)
        session.flush()
        return run.id


def snap(engine, run_id, url, body, entity_id=1, minutes=0) -> None:
    result = FetchResult(
        final_url=url,
        status=200,
        content_type="text/html",
        body_bytes=body,
        headers={},
        fetched_at=T0 + timedelta(minutes=minutes),
    )
    record_snapshot(
        engine, entity_id=entity_id, url=url, result=result, source_kind="crawl", run_id=run_id
    )


def statuses(diffs: list[UrlDiff]) -> list[tuple[str, str]]:
    return [(d.status, d.url) for d in diffs]


def test_classifies_all_four_statuses_sorted_by_url(engine):
    base, head = add_run(engine), add_run(engine)
    snap(engine, base, "https://e/b", b"same")
    snap(engine, base, "https://e/a", b"old")
    snap(engine, base, "https://e/gone", b"x")
    snap(engine, head, "https://e/a", b"new")
    snap(engine, head, "https://e/b", b"same")
    snap(engine, head, "https://e/new", b"y")
    diff = diff_entity_runs(engine, 1)
    assert (diff.base_run_id, diff.head_run_id) == (base, head)
    assert statuses(diff.urls) == [
        ("changed", "https://e/a"),
        ("unchanged", "https://e/b"),
        ("removed", "https://e/gone"),
        ("added", "https://e/new"),
    ]
    assert diff.counts() == {"changed": 1, "unchanged": 1, "added": 1, "removed": 1}


def test_latest_snapshot_per_url_within_a_run(engine):
    base, head = add_run(engine), add_run(engine)
    snap(engine, base, "https://e/a", b"v1")
    snap(engine, head, "https://e/a", b"v2", minutes=1)
    snap(engine, head, "https://e/a", b"v1", minutes=5)
    snap(engine, base, "https://e/c", b"v1", minutes=5)
    snap(engine, base, "https://e/c", b"v0", minutes=1)
    snap(engine, head, "https://e/c", b"v1")
    assert statuses(diff_entity_runs(engine, 1).urls) == [
        ("unchanged", "https://e/a"),
        ("unchanged", "https://e/c"),
    ]


def test_default_uses_two_most_recent_runs_with_snapshots_for_entity(engine):
    r1, r2 = add_run(engine), add_run(engine)
    other = add_run(engine)  # only another entity's snapshots
    empty = add_run(engine)  # no snapshots at all
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://e/a", b"2")
    snap(engine, other, "https://o/a", b"z", entity_id=2)
    diff = diff_entity_runs(engine, 1)
    assert (diff.base_run_id, diff.head_run_id) == (r1, r2)
    assert empty > r2


def test_explicit_runs(engine):
    r1, r2, r3 = add_run(engine), add_run(engine), add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://e/a", b"2")
    snap(engine, r3, "https://e/a", b"1")
    diff = diff_entity_runs(engine, 1, base_run_id=r1, head_run_id=r3)
    assert statuses(diff.urls) == [("unchanged", "https://e/a")]
    diff = diff_entity_runs(engine, 1, base_run_id=r1)
    assert diff.head_run_id == r3
    diff = diff_entity_runs(engine, 1, head_run_id=r2)
    assert (diff.base_run_id, diff.head_run_id) == (r1, r2)
    assert statuses(diff.urls) == [("changed", "https://e/a")]


def test_other_entities_snapshots_ignored(engine):
    base, head = add_run(engine), add_run(engine)
    snap(engine, base, "https://e/a", b"1")
    snap(engine, head, "https://e/a", b"1")
    snap(engine, base, "https://e/a", b"other", entity_id=2, minutes=9)
    snap(engine, head, "https://o/x", b"x", entity_id=2)
    assert statuses(diff_entity_runs(engine, 1).urls) == [("unchanged", "https://e/a")]


def test_too_few_runs(engine):
    r1 = add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    with pytest.raises(SnapshotDiffError, match="fewer than two runs"):
        diff_entity_runs(engine, 1)
    with pytest.raises(SnapshotDiffError, match="fewer than two runs"):
        diff_entity_runs(engine, 99)


def test_named_run_without_snapshots_for_entity(engine):
    r1, r2 = add_run(engine), add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://o/a", b"1", entity_id=2)
    with pytest.raises(SnapshotDiffError, match=f"run {r2} has no snapshots for entity 1"):
        diff_entity_runs(engine, 1, base_run_id=r1, head_run_id=r2)
    with pytest.raises(SnapshotDiffError, match="run 777 has no snapshots"):
        diff_entity_runs(engine, 1, base_run_id=777, head_run_id=r1)
