"""Tests for core.assessment_store (AS-10): append-only assessment and evidence rows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.assess import Extraction, Quote
from core.assessment_store import (
    AssessmentRow,
    CurrentEvidence,
    EvidenceRow,
    current_assessments,
    record_assessment,
)
from core.entity_store import Entity
from core.runs import Run, create_all
from core.verify import Route

CRIT = "public_terms"
RUBRIC_VERSION = "0.0.1"


@pytest.fixture
def engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path}/dm.sqlite")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def add_entity(engine, name="Acme") -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example",
            name=name,
            primary_domain=f"{name.lower()}.example",
            status="in_scope",
        )
        session.add(entity)
        session.flush()
        return entity.id


def add_run(engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="assess", status="running", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def extraction(*quotes: Quote, status: str = "strong") -> Extraction:
    return Extraction(
        status=status,
        quotes=quotes,
        rationale="Published.",
        checks_passed=True,
        failures=(),
    )


def record(engine, entity_id: int, run_id: int, ext: Extraction, routed: Route) -> int:
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=CRIT,
        extraction=ext,
        route=routed,
        rubric_version=RUBRIC_VERSION,
        extractor_model="extractor-x",
        verifier_model="verifier-y",
        run_id=run_id,
    )


def assert_rows(engine) -> tuple[list[AssessmentRow], list[EvidenceRow]]:
    with Session(engine) as session:
        assessments = list(session.scalars(sa.select(AssessmentRow).order_by(AssessmentRow.id)))
        evidence = list(session.scalars(sa.select(EvidenceRow).order_by(EvidenceRow.id)))
        return assessments, evidence


def test_create_all_creates_assessment_tables(engine) -> None:
    assert {"assessment", "evidence"} <= set(sa.inspect(engine).get_table_names())


def test_record_assessment_round_trip(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    quoted = Quote(7, "general terms and conditions", True, "exact", (0, 29))
    invented = Quote(9, "invented text", False, None, None)
    assessment_id = record(
        engine,
        entity_id,
        run_id,
        extraction(quoted, invented),
        Route("adjudicate", None, "verifier disagrees: too generic"),
    )

    assessments, evidence = assert_rows(engine)
    assert len(assessments) == 1
    row = assessments[0]
    assert assessment_id == row.id
    assert row.entity_id == entity_id
    assert row.criterion_id == CRIT
    assert row.status == "strong"
    assert row.confidence is None
    assert row.rationale == "Published."
    assert row.rubric_version == RUBRIC_VERSION
    assert row.extractor_model == "extractor-x"
    assert row.verifier_model == "verifier-y"
    assert row.decided_by == "auto"
    assert row.route_kind == "adjudicate"
    assert row.run_id == run_id
    assert row.created_at.tzinfo == UTC

    assert [(e.assessment_id, e.chunk_id, e.quote, e.grounded) for e in evidence] == [
        (assessment_id, 7, "general terms and conditions", True),
        (assessment_id, 9, "invented text", False),
    ]


@pytest.mark.parametrize("route_kind", ["accept", "adjudicate", "silent"])
def test_route_kind_is_stored(engine, route_kind: str) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    confidence = "high" if route_kind == "accept" else None
    record(
        engine,
        entity_id,
        run_id,
        extraction(),
        Route(route_kind, confidence, "reason"),  # type: ignore[arg-type]
    )
    assessments, evidence = assert_rows(engine)
    assert assessments[0].route_kind == route_kind
    assert assessments[0].confidence is None
    assert evidence == []


def test_two_calls_append_two_rows(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    quote = Quote(3, "terms", True, "exact", (0, 5))
    first = record(engine, entity_id, run_id, extraction(quote), Route("accept", "high", "ok"))
    second = record(engine, entity_id, run_id, extraction(quote), Route("accept", "high", "ok"))

    assessments, evidence = assert_rows(engine)
    assert first != second
    assert [row.id for row in assessments] == [first, second]
    # Append-only: the first assessment and its evidence are untouched.
    assert [e.assessment_id for e in evidence] == [first, second]


def test_silent_outcome_stores_no_evidence(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    assessment_id = record(
        engine,
        entity_id,
        run_id,
        Extraction("silent", (), "No evidence chunks.", True, ()),
        Route("silent", None, "No relevant evidence."),
    )
    assessments, evidence = assert_rows(engine)
    assert [row.id for row in assessments] == [assessment_id]
    assert assessments[0].status == "silent"
    assert evidence == []


def test_failure_writes_nothing_in_one_transaction(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    bad = Quote(None, "terms", True, "exact", (0, 5))  # type: ignore[arg-type]
    with pytest.raises(IntegrityError):
        record(engine, entity_id, run_id, extraction(bad), Route("accept", "high", "ok"))
    assessments, evidence = assert_rows(engine)
    assert assessments == []
    assert evidence == []


# --- AS-11: current view ---------------------------------------------------------------------


def record_v(
    engine,
    entity_id: int,
    run_id: int,
    *quotes: Quote,
    criterion_id: str = CRIT,
    rubric_version: str = RUBRIC_VERSION,
    status: str = "strong",
    kind: str = "accept",
) -> int:
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion_id,
        extraction=extraction(*quotes, status=status),
        route=Route(kind, None, "r"),  # type: ignore[arg-type]
        rubric_version=rubric_version,
        extractor_model="extractor-x",
        verifier_model="verifier-y",
        run_id=run_id,
    )


def set_created(engine, assessment_id: int, when: datetime) -> None:
    with Session(engine) as session, session.begin():
        session.get(AssessmentRow, assessment_id).created_at = when  # type: ignore[union-attr]


def test_current_empty_db(engine) -> None:
    assert current_assessments(engine) == []


def test_current_returns_newer_with_own_evidence(engine) -> None:
    eid, rid = add_entity(engine), add_run(engine)
    old = record_v(engine, eid, rid, Quote(1, "old", True, "exact", (0, 3)), status="weak")
    new = record_v(engine, eid, rid, Quote(2, "new", False, None, None))
    set_created(engine, old, datetime(2026, 1, 1, tzinfo=UTC))
    set_created(engine, new, datetime(2026, 1, 2, tzinfo=UTC))
    rows = current_assessments(engine)
    assert [r.id for r in rows] == [new]
    assert rows[0].status == "strong"
    assert rows[0].evidence == (CurrentEvidence(2, "new", False),)


def test_current_older_id_with_later_time_wins(engine) -> None:
    eid, rid = add_entity(engine), add_run(engine)
    first = record_v(engine, eid, rid)
    second = record_v(engine, eid, rid)
    set_created(engine, first, datetime(2026, 1, 3, tzinfo=UTC))
    set_created(engine, second, datetime(2026, 1, 2, tzinfo=UTC))
    assert [r.id for r in current_assessments(engine)] == [first]


def test_current_tie_resolves_by_id(engine) -> None:
    eid, rid = add_entity(engine), add_run(engine)
    same = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [record_v(engine, eid, rid) for _ in range(3)]
    for i in ids:
        set_created(engine, i, same)
    assert [r.id for r in current_assessments(engine)] == [max(ids)]


def test_current_per_rubric_version_and_filters(engine) -> None:
    a, b, rid = add_entity(engine, "Acme"), add_entity(engine, "Bolt"), add_run(engine)
    a1 = record_v(engine, a, rid, rubric_version="1")
    a2 = record_v(engine, a, rid, rubric_version="2")
    a3 = record_v(engine, a, rid, criterion_id="other", rubric_version="1")
    b1 = record_v(engine, b, rid, rubric_version="1")
    rows = current_assessments(engine)
    assert [(r.entity_id, r.criterion_id, r.rubric_version) for r in rows] == [
        (a, "other", "1"),
        (a, CRIT, "1"),
        (a, CRIT, "2"),
        (b, CRIT, "1"),
    ]
    assert [r.id for r in rows] == [a3, a1, a2, b1]
    assert [r.id for r in current_assessments(engine, entity_id=b)] == [b1]
    assert [r.id for r in current_assessments(engine, criterion_id="other")] == [a3]
    assert [r.id for r in current_assessments(engine, rubric_version="2")] == [a2]
    assert current_assessments(engine, entity_id=b, rubric_version="2") == []


def test_current_silent_row_has_no_evidence(engine) -> None:
    eid, rid = add_entity(engine), add_run(engine)
    aid = record_v(engine, eid, rid, status="unknown", kind="silent")
    (row,) = current_assessments(engine)
    assert (row.id, row.route_kind, row.evidence) == (aid, "silent", ())
    assert row.run_id == rid and row.decided_by == "auto" and row.created_at.tzinfo == UTC
