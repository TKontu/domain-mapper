"""Tests for core.attribute_store (AT-01): typed attribute values and the current view."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.attribute_store import (
    AttributeValueRow,
    current_attribute_values,
    record_attribute_value,
)
from core.runs import Run, create_all
from core.scope_pack import Attribute, load_scope_pack

EXAMPLE = Path(__file__).resolve().parent.parent / "scopes" / "_example"

INT_ATTR = Attribute(id="headcount", label="Headcount", type="int")
FLOAT_ATTR = Attribute(id="rating", label="Rating", type="float", unit="points")
ENUM_ATTR = Attribute(id="pricing", label="Pricing", type="enum", values=["per_word", "per_page"])
BOOL_ATTR = Attribute(id="certified", label="Certified", type="bool")
DATE_ATTR = Attribute(id="founded", label="Founded", type="date")


@pytest.fixture
def engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path}/dm.sqlite")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def current(engine, entity_id: int) -> dict[str, object]:
    with Session(engine) as session:
        return current_attribute_values(session, entity_id)


def row_count(engine) -> int:
    with Session(engine) as session:
        return session.scalar(sa.select(sa.func.count(AttributeValueRow.id))) or 0


def test_create_all_creates_attribute_value_table(engine) -> None:
    assert "attribute_value" in sa.inspect(engine).get_table_names()


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [
        (INT_ATTR, 42, 42),
        (FLOAT_ATTR, 2.5, 2.5),
        (FLOAT_ATTR, 3, 3.0),
        (ENUM_ATTR, "per_page", "per_page"),
        (BOOL_ATTR, True, True),
        (BOOL_ATTR, False, False),
        (DATE_ATTR, "2014-03-02", "2014-03-02"),
        (DATE_ATTR, date(2014, 3, 2), "2014-03-02"),
    ],
)
def test_each_type_round_trips(engine, attribute, value, expected) -> None:
    with Session(engine) as session, session.begin():
        row_id = record_attribute_value(session, 7, attribute, value)
    assert row_id > 0

    with Session(engine) as session:
        row = session.get(AttributeValueRow, row_id)
        assert row is not None
        assert row.entity_id == 7
        assert row.attribute_id == attribute.id
        assert row.value_json == expected
        assert row.unit == attribute.unit
        assert row.confidence is None
        assert row.run_id is None
        assert current_attribute_values(session, 7) == {attribute.id: expected}


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        (INT_ATTR, "42"),
        (INT_ATTR, True),
        (INT_ATTR, 4.2),
        (FLOAT_ATTR, "2.5"),
        (FLOAT_ATTR, True),
        (ENUM_ATTR, "hourly"),
        (ENUM_ATTR, 2),
        (BOOL_ATTR, 1),
        (BOOL_ATTR, "true"),
        (DATE_ATTR, "2014-02-30"),
        (DATE_ATTR, "not-a-date"),
        (DATE_ATTR, 20140302),
        (DATE_ATTR, datetime(2014, 3, 2, 12, 0, tzinfo=UTC)),
    ],
)
def test_invalid_value_raises_and_writes_nothing(engine, attribute, value) -> None:
    with Session(engine) as session:
        with pytest.raises(ValueError):
            record_attribute_value(session, 7, attribute, value)
        assert row_count(engine) == 0
        assert current_attribute_values(session, 7) == {}


def test_latest_value_per_attribute_wins(engine) -> None:
    with Session(engine) as session, session.begin():
        record_attribute_value(session, 7, INT_ATTR, 10)
        record_attribute_value(session, 7, ENUM_ATTR, "per_word")
        record_attribute_value(session, 7, INT_ATTR, 12)

    assert current(engine, 7) == {"headcount": 12, "pricing": "per_word"}


def test_other_entities_values_are_not_returned(engine) -> None:
    with Session(engine) as session, session.begin():
        record_attribute_value(session, 7, INT_ATTR, 10)
        record_attribute_value(session, 8, INT_ATTR, 99)
        record_attribute_value(session, 8, BOOL_ATTR, True)

    assert current(engine, 7) == {"headcount": 10}
    assert current(engine, 8) == {"headcount": 99, "certified": True}


def test_entity_without_values_returns_empty_dict(engine) -> None:
    assert current(engine, 404) == {}


def test_confidence_and_run_id_are_stored(engine) -> None:
    with Session(engine) as session, session.begin():
        run = Run(stage="attributes", status="running", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        run_id = run.id
        row_id = record_attribute_value(session, 7, FLOAT_ATTR, 1.5, confidence=0.8, run_id=run_id)

    with Session(engine) as session:
        row = session.get(AttributeValueRow, row_id)
        assert row is not None
        assert row.confidence == 0.8
        assert row.run_id == run_id


def test_example_pack_attribute_stores_its_unit(engine) -> None:
    pack = load_scope_pack(EXAMPLE)
    attributes = {a.id: a for a in pack.attributes.attributes}

    with Session(engine) as session, session.begin():
        record_attribute_value(session, 7, attributes["language_pairs"], 14)
        with pytest.raises(ValueError):
            record_attribute_value(session, 7, attributes["pricing_model"], "hourly")

    with Session(engine) as session:
        assert current_attribute_values(session, 7) == {"language_pairs": 14}
        row = session.scalars(sa.select(AttributeValueRow)).one()
        assert row.unit == "pairs"


def test_current_values_many_matches_per_entity_and_omits_empty(engine) -> None:
    from core.attribute_store import current_attribute_values_many

    with Session(engine) as session, session.begin():
        record_attribute_value(session, 1, INT_ATTR, 1)
        record_attribute_value(session, 2, INT_ATTR, 5)
        record_attribute_value(session, 1, INT_ATTR, 2)
        record_attribute_value(session, 1, ENUM_ATTR, "per_word")
        record_attribute_value(session, 2, BOOL_ATTR, True)
        record_attribute_value(session, 2, INT_ATTR, 6)
        record_attribute_value(session, 1, ENUM_ATTR, "per_page")
    with Session(engine) as session:
        many = current_attribute_values_many(session, [1, 2, 3])
        assert many == {
            1: {"headcount": 2, "pricing": "per_page"},
            2: {"certified": True, "headcount": 6},
        }
        for entity_id in (1, 2):
            assert many[entity_id] == current_attribute_values(session, entity_id)
        assert current_attribute_values(session, 3) == {}


def test_current_values_many_empty_input_does_not_query(engine) -> None:
    from core.attribute_store import current_attribute_values_many

    statements: list[str] = []
    sa.event.listen(engine, "before_cursor_execute", lambda *a: statements.append(a[2]))
    with Session(engine) as session:
        assert current_attribute_values_many(session, []) == {}
    assert statements == []


def test_value_with_evidence_stores_rows_returned_in_id_order(engine) -> None:
    from core.assessment_store import EvidenceRow
    from core.attribute_store import attribute_evidence_many

    with Session(engine) as session, session.begin():
        with_ev = record_attribute_value(
            session,
            1,
            INT_ATTR,
            12,
            evidence=[(7, "twelve staff", True), (3, "about a dozen", False)],
        )
        without_ev = record_attribute_value(session, 2, INT_ATTR, 5)
    with Session(engine) as session:
        rows = session.scalars(sa.select(EvidenceRow).order_by(EvidenceRow.id)).all()
        assert [(r.attribute_value_id, r.assessment_id) for r in rows] == [
            (with_ev, None),
            (with_ev, None),
        ]
        found = attribute_evidence_many(session, [with_ev, without_ev])
    assert list(found) == [with_ev]
    assert [(e.chunk_id, e.quote, e.grounded) for e in found[with_ev]] == [
        (7, "twelve staff", True),
        (3, "about a dozen", False),
    ]


def test_invalid_value_writes_no_value_and_no_evidence(engine) -> None:
    from core.assessment_store import EvidenceRow

    with Session(engine) as session, session.begin():
        with pytest.raises(ValueError):
            record_attribute_value(session, 1, INT_ATTR, "x", evidence=[(1, "q", True)])
    with Session(engine) as session:
        assert session.scalars(sa.select(AttributeValueRow)).all() == []
        assert session.scalars(sa.select(EvidenceRow)).all() == []


def test_attribute_evidence_many_empty_input_does_not_query(engine) -> None:
    from core.attribute_store import attribute_evidence_many

    statements: list[str] = []
    sa.event.listen(engine, "before_cursor_execute", lambda *a: statements.append(a[2]))
    with Session(engine) as session:
        assert attribute_evidence_many(session, []) == {}
    assert statements == []


def test_current_attribute_value_rows_many_returns_row_unit(engine) -> None:
    from core.attribute_store import current_attribute_value_rows_many

    with Session(engine) as session, session.begin():
        hc_id = record_attribute_value(session, 1, INT_ATTR, 1)  # no unit
        record_attribute_value(session, 1, FLOAT_ATTR, 2.5)  # superseded, unit points
        rating_id = record_attribute_value(session, 1, FLOAT_ATTR, 3.5)  # current, unit points
    with Session(engine) as session:
        many = current_attribute_value_rows_many(session, [1, 2])
    assert many == {1: {"headcount": (hc_id, 1, None), "rating": (rating_id, 3.5, "points")}}
