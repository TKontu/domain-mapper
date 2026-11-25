"""Tests for ``core.candidate_store`` (DS-02); SQLite in-memory only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.candidate_store import Candidate, NewCandidate, record_candidates
from core.runs import create_all

T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone(timedelta(hours=2)))


@pytest.fixture
def engine() -> sa.Engine:
    eng = sa.create_engine("sqlite://")
    create_all(eng)
    return eng


def _row(i: int, **kw: object) -> NewCandidate:
    base: dict[str, object] = {
        "name": f"Title {i}",
        "url": f"https://example.org/{i}",
        "channel": "search",
        "query": "web crawler",
        "found_at": T0,
    }
    base.update(kw)
    return NewCandidate(**base)  # type: ignore[arg-type]


def _all(engine: sa.Engine) -> list[Candidate]:
    with Session(engine) as s:
        return list(s.scalars(sa.select(Candidate).order_by(Candidate.id)))


def test_writes_rows_in_order_with_nulls(engine: sa.Engine) -> None:
    assert record_candidates(engine, [_row(1), _row(2)]) == 2
    rows = _all(engine)
    assert [r.url for r in rows] == ["https://example.org/1", "https://example.org/2"]
    r = rows[0]
    assert (r.name, r.channel, r.query) == ("Title 1", "search", "web crawler")
    assert r.found_at == T0 and r.found_at.tzinfo == UTC
    assert r.country_hint is None and r.segment_hint is None
    assert r.source_id is None and r.resolved_entity_id is None


def test_duplicates_are_kept(engine: sa.Engine) -> None:
    record_candidates(engine, [_row(1)])
    record_candidates(engine, [_row(1)])
    assert len(_all(engine)) == 2


def test_empty_input_writes_nothing(engine: sa.Engine) -> None:
    with mock.patch("core.candidate_store.Session", side_effect=AssertionError("no session")):
        assert record_candidates(engine, []) == 0
    assert _all(engine) == []


@pytest.mark.parametrize("field", ["url", "channel"])
@pytest.mark.parametrize("value", ["", "  "])
def test_blank_url_or_channel_rejected_before_session(
    engine: sa.Engine, field: str, value: str
) -> None:
    rows = [_row(1), _row(2, **{field: value})]
    with mock.patch("core.candidate_store.Session", side_effect=AssertionError("no session")):
        with pytest.raises(ValueError, match=field):
            record_candidates(engine, rows)
    assert _all(engine) == []


def test_naive_found_at_rejected_nothing_written(engine: sa.Engine) -> None:
    with pytest.raises(Exception):
        record_candidates(engine, [_row(1), _row(2, found_at=datetime(2026, 1, 1))])
    assert _all(engine) == []


def test_resolved_entity_id_indexed(engine: sa.Engine) -> None:
    idx = {tuple(i["column_names"]) for i in sa.inspect(engine).get_indexes("candidate")}
    assert ("resolved_entity_id",) in idx
