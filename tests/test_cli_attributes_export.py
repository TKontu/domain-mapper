"""Tests for ``landscape attributes-export`` (AT-03); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.attribute_store import record_attribute_value
from core.entity_store import Entity
from core.runs import create_all
from core.scope_pack import Attribute

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
HEADCOUNT = Attribute(id="headcount", label="Headcount", type="int")
PRICING = Attribute(
    id="pricing_model", label="Pricing model", type="enum", values=["per_word", "per_page"]
)
REVENUE = Attribute(id="revenue", label="Revenue", type="float", unit="MEUR")
CERTIFIED = Attribute(id="certified", label="Certified", type="bool")
FOUNDED = Attribute(id="founded", label="Founded", type="date")


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


def add_entity(engine: Engine, name: str, scope_id: str = "example") -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id=scope_id, name=name, primary_domain=f"{name}.example", status="in_scope"
        )
        session.add(entity)
        session.flush()
        return entity.id


def add_value(engine: Engine, entity_id: int, attribute: Attribute, value: object) -> None:
    with Session(engine) as session, session.begin():
        record_attribute_value(session, entity_id, attribute, value)


def export(out: Path):
    return CliRunner().invoke(
        cli.app, ["attributes-export", "--scope", EXAMPLE_SCOPE, "--out", str(out)]
    )


def test_entities_with_values_one_skipped_and_keys_sorted(engine, tmp_path) -> None:
    a = add_entity(engine, "alpha")
    b = add_entity(engine, "beta")
    add_entity(engine, "gamma")
    add_entity(engine, "other", scope_id="elsewhere")
    add_value(engine, a, HEADCOUNT, 42)
    add_value(engine, a, PRICING, "per_word")
    add_value(engine, b, HEADCOUNT, 7)
    out = tmp_path / "attributes.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {"entity_id": a, "attributes": {"headcount": 42, "pricing_model": "per_word"}},
        {"entity_id": b, "attributes": {"headcount": 7}},
    ]
    # The attributes object is written with attribute ids sorted, not in insertion order.
    assert list(lines[0]["attributes"]) == ["headcount", "pricing_model"]
    assert result.stdout == f"exported 2 entities to {out} (skipped 1 without attributes)\n"
    assert not Path(f"{out}.tmp").exists()


def test_latest_value_wins(engine, tmp_path) -> None:
    a = add_entity(engine, "alpha")
    add_value(engine, a, HEADCOUNT, 10)
    add_value(engine, a, HEADCOUNT, 12)
    out = tmp_path / "attributes.jsonl"
    assert export(out).exit_code == 0
    (line,) = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {"entity_id": a, "attributes": {"headcount": 12}}


def test_values_are_written_json_decoded(engine, tmp_path) -> None:
    a = add_entity(engine, "alpha")
    add_value(engine, a, REVENUE, 1.5)
    add_value(engine, a, CERTIFIED, True)
    add_value(engine, a, FOUNDED, "2001-05-17")
    out = tmp_path / "attributes.jsonl"
    assert export(out).exit_code == 0
    (line,) = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "entity_id": a,
        "attributes": {"certified": True, "founded": "2001-05-17", "revenue": 1.5},
    }


def test_empty_database_writes_empty_file(engine, tmp_path) -> None:
    out = tmp_path / "attributes.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == ""
    assert result.stdout == f"exported 0 entities to {out} (skipped 0 without attributes)\n"


def test_unwritable_directory_exits_2_and_leaves_nothing(engine, tmp_path) -> None:
    out = tmp_path / "missing" / "attributes.jsonl"
    result = export(out)
    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()
