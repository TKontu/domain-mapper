"""Tests for ``core.presence_store`` (MP-01); SQLite only."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from core.entity_store import Entity
from core.presence_store import (
    MarketPresenceRow,
    layer_ids_for,
    presences_many,
    record_presence,
)
from core.runs import create_all
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


@pytest.fixture
def engine(tmp_path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path}/p.sqlite")
    create_all(engine)
    with Session(engine) as session, session.begin():
        for i in (1, 2, 3):
            session.add(
                Entity(id=i, scope_id="s", name=f"e{i}", primary_domain=f"e{i}.example",
                       status="unscreened")
            )  # fmt: skip
    try:
        yield engine
    finally:
        engine.dispose()


def count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(sa.select(sa.func.count()).select_from(MarketPresenceRow)).scalar_one()


def test_record_and_read_back_idempotent(engine):
    first = record_presence(engine, 1, "XA", "hq")
    assert record_presence(engine, 1, "XA", "hq") == first
    with Session(engine) as session, session.begin():
        record_presence(session, 1, "XB", "sales")
    assert count(engine) == 2
    assert presences_many(engine, [1]) == {1: [("XA", "hq"), ("XB", "sales")]}


@pytest.mark.parametrize(
    ("country", "kind"), [("XA", "office"), ("xa", "hq"), ("XAA", "hq"), ("", "hq")]
)
def test_invalid_values_raise(engine, country, kind):
    with pytest.raises(ValueError):
        record_presence(engine, 1, country, kind)
    assert count(engine) == 0


def test_presences_many_batches_and_omits_missing(engine):
    record_presence(engine, 1, "XA", "hq")
    record_presence(engine, 2, "XC", "remote")
    record_presence(engine, 2, "XB", "site")
    assert presences_many(engine, [1, 2, 3, 99]) == {
        1: [("XA", "hq")],
        2: [("XB", "site"), ("XC", "remote")],
    }
    assert presences_many(engine, []) == {}


def test_presences_many_is_one_query(engine):
    record_presence(engine, 1, "XA", "hq")
    statements: list[str] = []
    sa.event.listen(engine, "before_cursor_execute", lambda *a: statements.append(a[2]))
    presences_many(engine, [1, 2, 3])
    assert len(statements) == 1


@pytest.mark.parametrize(
    ("countries", "expected"),
    [
        (["XA"], ["home", "region", "global"]),
        (["XB"], ["region", "global"]),
        (["ZZ"], ["global"]),
        ([], []),
    ],
)
def test_layer_ids_for_example_layers(countries, expected):
    layers = load_scope_pack(EXAMPLE_SCOPE).scope.layers
    assert layer_ids_for(layers, countries) == expected
