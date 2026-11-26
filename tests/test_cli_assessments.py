"""Tests for ``landscape assessments`` (AS-11); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.assess import Extraction, Quote
from core.assessment_store import AssessmentRow, record_assessment
from core.runs import Run, create_all
from core.verify import Route

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


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


def add_run(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="assess", status="finished", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def add(engine, run_id, entity_id, criterion="c1", version="1", quotes=(), kind="accept") -> int:
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status="strong", quotes=tuple(quotes), rationale="x", checks_passed=True, failures=()
        ),
        route=Route(kind, None, "r"),
        rubric_version=version,
        extractor_model="e",
        verifier_model="v",
        run_id=run_id,
    )


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["assessments", "--scope", EXAMPLE_SCOPE, *args])


def test_empty_prints_no_assessments(engine) -> None:
    result = invoke()
    assert result.exit_code == 0, result.output
    assert result.output == "no assessments\n"


def test_lines_and_latest_only(engine) -> None:
    rid = add_run(engine)
    old = add(engine, rid, 5, quotes=[Quote(9, "old", True, "exact", (0, 3))])
    new = add(
        engine,
        rid,
        5,
        quotes=[Quote(3, "a", True, "exact", (0, 1)), Quote(4, "b", False, None, None)],
    )
    silent = add(engine, rid, 6, kind="silent")
    with Session(engine) as session, session.begin():
        session.get(AssessmentRow, old).created_at = datetime(2020, 1, 1, tzinfo=UTC)  # type: ignore[union-attr]
    result = invoke()
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"assessment {new} entity 5 criterion c1 rubric 1 status strong route accept"
        f" decided_by auto run {rid}",
        "  evidence chunk 3 grounded yes",
        "  evidence chunk 4 grounded no",
        f"assessment {silent} entity 6 criterion c1 rubric 1 status strong route silent"
        f" decided_by auto run {rid}",
    ]


def test_filters_pass_through(engine) -> None:
    rid = add_run(engine)
    add(engine, rid, 1)
    b = add(engine, rid, 2, criterion="c2", version="2")
    add(engine, rid, 2, criterion="c1", version="2")
    for args in (
        ["--entity", "2", "--criterion", "c2"],
        ["--criterion", "c2", "--rubric-version", "2"],
    ):
        result = invoke(*args)
        assert result.exit_code == 0, result.output
        assert [ln.split()[1] for ln in result.output.splitlines()] == [str(b)]
    assert invoke("--rubric-version", "9").output == "no assessments\n"


def test_missing_scope_exits_2(engine, tmp_path) -> None:
    result = CliRunner().invoke(cli.app, ["assessments", "--scope", str(tmp_path / "nope")])
    assert result.exit_code == 2
