"""Tests for ``landscape attributes-export --format csv`` (XP-03); SQLite under a temp dir."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.attribute_store import record_attribute_value
from core.entity_store import Entity
from core.runs import create_all
from core.scope_pack import Attribute

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
HEADCOUNT = Attribute(id="headcount", label="Headcount", type="int")
PAIRS = Attribute(id="language_pairs", label="Pairs", type="float", unit="pairs")
PRICING = Attribute(id="pricing_model", label="Pricing", type="enum", values=["per_word", "=x"])
OFF_PACK = Attribute(id="revenue", label="Revenue", type="float", unit="MEUR")
HEADER = [
    "entity_id",
    *(
        f"{attr}.{field}"
        for attr in ("headcount", "language_pairs", "pricing_model")
        for field in ("value", "unit", "evidence_count", "grounded_count")
    ),
]


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


def add_entity(engine: Engine, name: str) -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example", name=name, primary_domain=f"{name}.example", status="in_scope"
        )
        session.add(entity)
        session.flush()
        return entity.id


def add_value(engine: Engine, entity_id: int, attribute: Attribute, value: object, **kw) -> None:
    with Session(engine) as session, session.begin():
        record_attribute_value(session, entity_id, attribute, value, **kw)


def export(out: Path, *extra: str):
    return CliRunner().invoke(
        cli.app, ["attributes-export", "--scope", EXAMPLE_SCOPE, "--out", str(out), *extra]
    )


def read_rows(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_csv_rows_in_pack_order_with_counts_and_empties(engine, tmp_path) -> None:
    a = add_entity(engine, "alpha")
    b = add_entity(engine, "beta")
    add_entity(engine, "gamma")
    add_value(engine, a, HEADCOUNT, 5, evidence=[(1, "q1", True)])
    add_value(engine, a, HEADCOUNT, 12, evidence=[(9, "q9", True), (4, "q4", False)])
    add_value(engine, a, PAIRS, -3.5)
    add_value(engine, a, OFF_PACK, 1.5)
    add_value(engine, b, PRICING, "=x")
    out = tmp_path / "attributes.csv"
    result = export(out, "--format", "csv")
    assert result.exit_code == 0, result.output
    assert read_rows(out) == [
        HEADER,
        [str(a), "12", "", "2", "1", "-3.5", "pairs", "0", "0", "", "", "", ""],
        [str(b), "", "", "", "", "", "", "", "", "'=x", "", "0", "0"],
    ]
    assert "revenue" not in out.read_text(encoding="utf-8")
    assert f"exported 2 entities to {out} (skipped 1 without attributes)" in result.output
    assert not Path(f"{out}.tmp").exists()


def test_csv_header_written_with_zero_entities(engine, tmp_path) -> None:
    out = tmp_path / "attributes.csv"
    result = export(out, "--format", "csv")
    assert result.exit_code == 0, result.output
    assert read_rows(out) == [HEADER]


def test_unknown_format_exits_2_before_setup_or_write(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("setup reached")

    monkeypatch.setattr(cli, "_load", boom)
    out = tmp_path / "attributes.csv"
    result = export(out, "--format", "parquet")
    assert result.exit_code == 2
    assert "unknown format" in result.output
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_jsonl_format_byte_identical_to_default(engine, tmp_path) -> None:
    a = add_entity(engine, "alpha")
    add_value(engine, a, HEADCOUNT, 12, evidence=[(9, "q9", True)])
    add_value(engine, a, OFF_PACK, 1.5)
    default = tmp_path / "default.jsonl"
    explicit = tmp_path / "explicit.jsonl"
    assert export(default).exit_code == 0
    assert export(explicit, "--format", "jsonl").exit_code == 0
    assert default.read_bytes() == explicit.read_bytes()
    assert b"revenue" in default.read_bytes()


def test_csv_streams_in_batches(engine, tmp_path, monkeypatch) -> None:
    ids = [add_entity(engine, f"e{i}") for i in range(5)]
    for i, entity_id in enumerate(ids):
        add_value(engine, entity_id, HEADCOUNT, i)
    calls: list[list[int]] = []
    real = cli.current_attribute_value_rows_many

    def counting(session, entity_ids):
        calls.append(list(entity_ids))
        return real(session, entity_ids)

    monkeypatch.setattr(cli, "ATTRIBUTES_EXPORT_BATCH", 2)
    monkeypatch.setattr(cli, "current_attribute_value_rows_many", counting)
    out = tmp_path / "attributes.csv"
    result = export(out, "--format", "csv")
    assert result.exit_code == 0, result.output
    assert calls == [ids[0:2], ids[2:4], ids[4:5]]
    assert [row[:2] for row in read_rows(out)[1:]] == [[str(e), str(i)] for i, e in enumerate(ids)]


def test_csv_unwritable_directory_exits_2(engine, tmp_path) -> None:
    out = tmp_path / "missing" / "attributes.csv"
    result = export(out, "--format", "csv")
    assert result.exit_code == 2
    assert not out.exists()
