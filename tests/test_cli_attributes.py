"""Tests for ``landscape attribute-set`` / ``landscape attributes`` (AT-02); SQLite, CliRunner."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from typer.testing import CliRunner

from core import cli
from core.attribute_store import AttributeValueRow
from core.runs import create_all

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"

# The example pack has int and enum only; float, bool and date are added test-locally.
EXTRA_ATTRIBUTES = """
  - id: revenue
    label: Revenue
    type: float
    unit: MEUR
  - id: certified
    label: Certified
    type: bool
  - id: founded
    label: Founded
    type: date
"""


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


@pytest.fixture
def scope(tmp_path) -> str:
    target = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, target)
    with (target / "attributes.yaml").open("a") as fh:
        fh.write(EXTRA_ATTRIBUTES)
    return str(target)


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def set_attr(scope: str, entity: int, attribute: str, value: str):
    return invoke(
        "attribute-set", "--scope", scope, "--entity", str(entity),
        "--attribute", attribute, "--value", value,
    )  # fmt: skip


def row_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(sa.select(sa.func.count()).select_from(AttributeValueRow)).scalar_one()


def test_round_trip_each_type(engine, scope):
    cases = [
        ("headcount", "42"),
        ("language_pairs", "7"),
        ("pricing_model", "per_word"),
        ("revenue", "1.5"),
        ("certified", "true"),
        ("founded", "2001-05-17"),
    ]
    for i, (attribute, value) in enumerate(cases, start=1):
        result = set_attr(scope, 5, attribute, value)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == f"attribute_value {i} entity 5 attribute {attribute}"
    result = invoke("attributes", "--scope", scope, "--entity", "5")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "certified true -",
        'founded "2001-05-17" -',
        "headcount 42 -",
        "language_pairs 7 pairs",
        'pricing_model "per_word" -',
        "revenue 1.5 MEUR",
    ]


def test_bool_false(engine, scope):
    assert set_attr(scope, 1, "certified", "false").exit_code == 0
    assert invoke("attributes", "--scope", scope, "--entity", "1").output == "certified false -\n"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("nope", "1"),
        ("headcount", "many"),
        ("headcount", "1.5"),
        ("revenue", "nan"),
        ("certified", "yes"),
        ("founded", "2001-13-40"),
        ("pricing_model", "per_minute"),
    ],
)
def test_invalid_exits_2_and_writes_nothing(engine, scope, attribute, value):
    result = set_attr(scope, 1, attribute, value)
    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert result.output.startswith("error: ")
    assert row_count(engine) == 0


def test_latest_set_wins(engine, scope):
    set_attr(scope, 3, "headcount", "10")
    set_attr(scope, 3, "headcount", "12")
    result = invoke("attributes", "--scope", scope, "--entity", "3")
    assert result.output == "headcount 12 -\n"


def test_other_entity_and_empty(engine, scope):
    set_attr(scope, 3, "headcount", "10")
    result = invoke("attributes", "--scope", scope, "--entity", "4")
    assert result.exit_code == 0
    assert result.output == "no attributes\n"


def test_database_unavailable(tmp_path, monkeypatch, scope):
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DM_DATABASE_URL", f"sqlite:///{tmp_path}/missing/dir/dm.sqlite")
    result = invoke("attributes", "--scope", scope, "--entity", "1")
    assert result.exit_code == 2
    assert "database unavailable" in result.output
