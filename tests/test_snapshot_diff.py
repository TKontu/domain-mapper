"""Tests for ``core.snapshot_diff`` (SD-01, SD-02); offline SQLite under a temp dir."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

import core.snapshot_store  # noqa: F401  (registers the snapshot table)
from adapters.fetcher import FetchResult
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.chunk_store import ChunkRow
from core.runs import Run, create_all
from core.snapshot_diff import (
    SnapshotDiffError,
    UrlDiff,
    diff_entity_runs,
    stale_assessments,
)
from core.verify import Route
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


def snap(engine, run_id, url, body, entity_id=1, minutes=0, final_url=None) -> int:
    result = FetchResult(
        final_url=final_url or url,
        status=200,
        content_type="text/html",
        body_bytes=body,
        headers={},
        fetched_at=T0 + timedelta(minutes=minutes),
    )
    return record_snapshot(
        engine, entity_id=entity_id, url=url, result=result, source_kind="crawl", run_id=run_id
    )


def chunk(engine, snapshot_id: int) -> int:
    with Session(engine) as session, session.begin():
        row = ChunkRow(
            snapshot_id=snapshot_id,
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


def assess(engine, run_id, criterion, chunk_ids, entity_id=1, rubric="v1") -> int:
    quotes = tuple(Quote(c, "q", True, "exact", (0, 1)) for c in chunk_ids)
    ext = Extraction(status="strong", quotes=quotes, rationale="r", checks_passed=True, failures=())
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=ext,
        route=Route("accept", "high", "ok"),
        rubric_version=rubric,
        extractor_model="x",
        verifier_model="y",
        run_id=run_id,
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


def test_equal_base_and_head_is_its_own_error(engine):
    r1, r2 = add_run(engine), add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://e/a", b"2")
    with pytest.raises(SnapshotDiffError, match="^base and head run are the same$"):
        diff_entity_runs(engine, 1, base_run_id=r1, head_run_id=r1)
    with pytest.raises(SnapshotDiffError, match="^base and head run are the same$"):
        stale_assessments(engine, 1, base_run_id=r2, head_run_id=r2)


def test_base_alone_at_newest_run_never_diffs_in_reverse(engine):
    r1, r2 = add_run(engine), add_run(engine)
    snap(engine, r1, "https://e/a", b"1")
    snap(engine, r2, "https://e/a", b"2")
    msg = f"^no run newer than base run {r2} has snapshots for entity 1$"
    with pytest.raises(SnapshotDiffError, match=msg):
        diff_entity_runs(engine, 1, base_run_id=r2)
    with pytest.raises(SnapshotDiffError, match=msg):
        stale_assessments(engine, 1, base_run_id=r2)


def stale_fixture(engine):
    base, head = add_run(engine), add_run(engine)
    ids = {}
    for name, old, new in [
        ("changed", b"1", b"2"),
        ("same", b"s", b"s"),
        ("gone", b"g", None),
        ("other", b"o", b"p"),
    ]:
        url = f"https://e/{name}"
        ids[name] = chunk(engine, snap(engine, base, url, old))
        if new is not None:
            snap(engine, head, url, new)
    ids["added"] = chunk(engine, snap(engine, head, "https://e/added", b"n"))
    return base, head, ids


def test_stale_assessments_changed_and_removed_only(engine):
    base, head, c = stale_fixture(engine)
    a_changed = assess(engine, base, "c2", [c["changed"]])
    a_removed = assess(engine, base, "c1", [c["gone"]], rubric="v2")
    assess(engine, base, "c3", [c["same"], c["added"]])
    assess(engine, base, "c4", [])  # no evidence at all
    a_multi = assess(engine, base, "c1", [c["other"], c["same"], c["changed"], c["other"]])
    other_snap = snap(engine, base, "https://e/changed", b"zz", entity_id=2)
    assess(engine, base, "c9", [chunk(engine, other_snap)], entity_id=2)
    plan = stale_assessments(engine, 1)
    assert (plan.diff.base_run_id, plan.diff.head_run_id) == (base, head)
    assert plan.current == 5
    got = [
        (s.criterion_id, s.rubric_version, s.assessment_id, [(u.status, u.url) for u in s.urls])
        for s in plan.stale
    ]
    assert got == [
        ("c1", "v1", a_multi, [("changed", "https://e/changed"), ("changed", "https://e/other")]),
        ("c1", "v2", a_removed, [("removed", "https://e/gone")]),
        ("c2", "v1", a_changed, [("changed", "https://e/changed")]),
    ]


def test_evidence_without_url_ignored(engine):
    base, _head, _c = stale_fixture(engine)
    orphan = chunk(engine, 99999)  # chunk whose snapshot row is absent -> url None
    assess(engine, base, "c1", [orphan, 424242])  # and a chunk id with no row at all
    plan = stale_assessments(engine, 1)
    assert plan.stale == []
    assert plan.current == 1


def test_stale_matches_evidence_citing_final_url(engine):
    base, head = add_run(engine), add_run(engine)
    # changed URL redirected in base run; evidence cites the base final_url
    c_changed = chunk(engine, snap(engine, base, "https://e/a", b"1", final_url="https://e/a2"))
    snap(engine, head, "https://e/a", b"2", final_url="https://e/a3")
    # removed URL matched via its base-run final_url
    c_gone = chunk(engine, snap(engine, base, "https://e/g", b"g", final_url="https://e/g2"))
    # unchanged and added URLs with final_urls: never flagged
    c_same = chunk(engine, snap(engine, base, "https://e/s", b"s", final_url="https://e/s2"))
    snap(engine, head, "https://e/s", b"s", final_url="https://e/s2")
    c_added = chunk(engine, snap(engine, head, "https://e/n", b"n", final_url="https://e/n2"))
    # final_url equal to url: listed once
    c_plain = chunk(engine, snap(engine, base, "https://e/p", b"p"))
    snap(engine, head, "https://e/p", b"q")
    a1 = assess(engine, base, "c1", [c_changed])
    a2 = assess(engine, base, "c2", [c_gone])
    assess(engine, base, "c3", [c_same, c_added])
    a4 = assess(engine, base, "c4", [c_plain, c_plain])
    # head-run final_url of the changed URL also matches, deduped with the snapshot url
    c_head = chunk(engine, snap(engine, head, "https://e/x", b"x", final_url="https://e/a3"))
    a5 = assess(engine, head, "c5", [c_head, c_changed])
    plan = stale_assessments(engine, 1)
    got = [(s.criterion_id, s.assessment_id, statuses(s.urls)) for s in plan.stale]
    assert got == [
        ("c1", a1, [("changed", "https://e/a")]),
        ("c2", a2, [("removed", "https://e/g")]),
        ("c4", a4, [("changed", "https://e/p")]),
        ("c5", a5, [("changed", "https://e/a")]),
    ]
    assert [d.url for d in plan.diff.urls] == [
        "https://e/a", "https://e/g", "https://e/n", "https://e/p", "https://e/s", "https://e/x"
    ]
