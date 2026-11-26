"""Tests for ``landscape presence-set`` / ``landscape presences`` (MP-01); SQLite, CliRunner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.entity_store import Entity
from core.presence_store import MarketPresenceRow
from core.runs import create_all
from core.scope_pack import load_scope_pack

SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


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
    scope_id = load_scope_pack(Path(SCOPE)).scope.id
    with Session(engine) as session, session.begin():
        session.add(Entity(id=1, scope_id=scope_id, name="One", primary_domain="one.example",
                           status="unscreened"))  # fmt: skip
        session.add(Entity(id=2, scope_id="other", name="Two", primary_domain="two.example",
                           status="unscreened"))  # fmt: skip
    try:
        yield engine
    finally:
        engine.dispose()


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def pset(entity: int, country: str, kind: str):
    return invoke(
        "presence-set", "--scope", SCOPE, "--entity", str(entity),
        "--country", country, "--type", kind,
    )  # fmt: skip


def count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(sa.select(sa.func.count()).select_from(MarketPresenceRow)).scalar_one()


def test_set_then_list_prints_sorted_rows_and_layers(engine):
    for country, kind in [("XB", "sales"), ("XA", "site"), ("XA", "hq"), ("XA", "hq")]:
        result = pset(1, country, kind)
        assert result.exit_code == 0, result.output
    assert count(engine) == 3
    result = invoke("presences", "--scope", SCOPE, "--entity", "1")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "XA hq",
        "XA site",
        "XB sales",
        "layers: home, region, global",
    ]


def test_list_without_presence_prints_empty_layers(engine):
    result = invoke("presences", "--scope", SCOPE, "--entity", "1")
    assert result.exit_code == 0
    assert result.output.splitlines() == ["layers: "]


@pytest.mark.parametrize(
    ("entity", "country", "kind"),
    [(1, "XA", "office"), (1, "xa", "hq"), (1, "XAA", "hq"), (99, "XA", "hq"), (2, "XA", "hq")],
)
def test_set_invalid_exits_2_writing_nothing(engine, entity, country, kind):
    result = pset(entity, country, kind)
    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert count(engine) == 0


def test_list_unknown_entity_exits_2(engine):
    result = invoke("presences", "--scope", SCOPE, "--entity", "99")
    assert result.exit_code == 2
