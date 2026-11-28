"""Tests for ``landscape eval`` (EV-02); temp SQLite, neutral ``scopes/_example`` pack."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.assess import Extraction
from core.assessment_store import record_assessment
from core.entity_store import Entity
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


def entity(engine, domain: str, scope_id: str = "example") -> int:
    with Session(engine) as session, session.begin():
        row = Entity(scope_id=scope_id, name=domain, primary_domain=domain, status="in_scope")
        session.add(row)
        session.flush()
        return row.id


def assess(engine, entity_id: int, criterion: str, status: str, version: str = "1") -> None:
    with Session(engine) as session, session.begin():
        run = Run(stage="assess", status="finished", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        run_id = run.id
    record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status=status, quotes=(), rationale="x", checks_passed=True, failures=()
        ),
        route=Route("accept", None, "r"),
        rubric_version=version,
        extractor_model="e",
        verifier_model="v",
        run_id=run_id,
    )


def gold(tmp_path, rows) -> str:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def label(domain, criterion, status):
    return {"primary_domain": domain, "criterion_id": criterion, "status": status}


def run(*args):
    return CliRunner().invoke(cli.app, ["eval", "--scope", EXAMPLE_SCOPE, *args])


def test_metrics_per_criterion_then_overall_with_unmatched(engine, tmp_path):
    a = entity(engine, "a.example")
    b = entity(engine, "b.example")
    entity(engine, "a.example", scope_id="other")
    assess(engine, a, "iso17100", "strong")
    assess(engine, b, "iso17100", "weak")
    assess(engine, a, "public_terms", "silent")
    path = gold(
        tmp_path,
        [
            label("a.example", "iso17100", "strong"),
            label("b.example", "iso17100", "standard"),
            label("a.example", "public_terms", "standard"),
            label("zz.example", "iso17100", "strong"),
            label("b.example", "public_terms", "weak"),
        ],
    )
    result = run("--gold", path)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines == [
        "unmatched 2",
        "iso17100 n 2 accuracy 0.500 kappa 0.333 silent_rate 0.000 false_strong_rate 0.000",
        "public_terms n 1 accuracy 0.000 kappa 0.000 silent_rate 1.000 false_strong_rate 0.000",
        "overall n 3 accuracy 0.333 kappa 0.250 silent_rate 0.333 false_strong_rate 0.000",
    ]


def test_empty_match_prints_n_zero(engine, tmp_path):
    path = gold(tmp_path, [label("a.example", "iso17100", "strong")])
    result = run("--gold", path)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "unmatched 1",
        "overall n 0 accuracy - kappa - silent_rate - false_strong_rate -",
    ]


def test_rubric_version_filter(engine, tmp_path):
    a = entity(engine, "a.example")
    assess(engine, a, "iso17100", "weak", version="1")
    assess(engine, a, "iso17100", "strong", version="2")
    path = gold(tmp_path, [label("a.example", "iso17100", "strong")])
    out1 = run("--gold", path, "--rubric-version", "1").output.splitlines()
    out2 = run("--gold", path, "--rubric-version", "2").output.splitlines()
    assert out1[-1].startswith("overall n 1 accuracy 0.000")
    assert out2[-1].startswith("overall n 1 accuracy 1.000")


def test_missing_gold_file_exits_2(engine, tmp_path):
    result = run("--gold", str(tmp_path / "nope.jsonl"))
    assert result.exit_code == 2
    assert "gold" in result.stderr


def test_default_gold_path_is_scope_gold_labels(engine):
    result = run()
    assert result.exit_code == 2
    assert "labels.jsonl" in result.stderr


def test_gold_set_error_exits_2_with_line(engine, tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"primary_domain": "a.example"}\n', encoding="utf-8")
    result = run("--gold", str(path))
    assert result.exit_code == 2
    assert "line 1" in result.stderr


def _gate_fixture(engine, tmp_path):
    """iso17100: 10 correct pairs (passes); public_terms: 10 pairs, all wrong (fails)."""
    rows = []
    for i in range(10):
        d = f"e{i}.example"
        eid = entity(engine, d)
        assess(engine, eid, "iso17100", "standard")
        assess(engine, eid, "public_terms", "weak")
        rows.append(label(d, "iso17100", "standard"))
        rows.append(label(d, "public_terms", "standard"))
    return gold(tmp_path, rows)


def test_without_gate_no_gate_lines(engine, tmp_path):
    result = run("--gold", _gate_fixture(engine, tmp_path))
    assert result.exit_code == 0, result.output
    assert not any(line.startswith("gate ") for line in result.output.splitlines())


def test_gate_fails_one_criterion_exit_1(engine, tmp_path):
    result = run("--gold", _gate_fixture(engine, tmp_path), "--gate")
    assert result.exit_code == 1, result.output
    gate = [line for line in result.output.splitlines() if line.startswith("gate ")]
    assert gate == [
        "gate iso17100 pass",
        "gate public_terms fail accuracy<0.8,kappa<0.6",
        "gate overall fail",
    ]


def test_gate_override_file_makes_it_pass(engine, tmp_path):
    path = _gate_fixture(engine, tmp_path)
    override = tmp_path / "t.yaml"
    override.write_text("criteria: {public_terms: {min_accuracy: 0.0, min_kappa: 0.0}}\n", encoding="utf-8")
    result = run("--gold", path, "--gate", "--thresholds", str(override))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "gate overall pass"
    assert "gate public_terms pass" in result.output.splitlines()


def test_gate_scope_override_file_is_used(engine, tmp_path):
    import shutil

    scope = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, scope)
    (scope / "eval").mkdir()
    (scope / "eval" / "thresholds.yaml").write_text(
        "criteria: {public_terms: {min_accuracy: 0.0, min_kappa: 0.0}}\n", encoding="utf-8"
    )
    path = _gate_fixture(engine, tmp_path)
    result = CliRunner().invoke(
        cli.app, ["eval", "--scope", str(scope), "--gold", path, "--gate"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "gate overall pass"


def test_gate_thresholds_error_exit_2(engine, tmp_path):
    path = _gate_fixture(engine, tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("default: {min_accuracy: 7}\n", encoding="utf-8")
    result = run("--gold", path, "--gate", "--thresholds", str(bad))
    assert result.exit_code == 2
    assert "min_accuracy" in result.output


def test_gate_missing_engine_file_exit_2(engine, tmp_path, monkeypatch):
    path = _gate_fixture(engine, tmp_path)
    monkeypatch.setattr(cli, "ENGINE_THRESHOLDS", tmp_path / "none.yaml")
    result = run("--gold", path, "--gate")
    assert result.exit_code == 2
    assert "thresholds file not found" in result.output


def test_gate_explicit_missing_override_exit_2(engine, tmp_path):
    path = _gate_fixture(engine, tmp_path)
    result = run("--gold", path, "--gate", "--thresholds", str(tmp_path / "nope.yaml"))
    assert result.exit_code == 2
    assert "thresholds file not found" in result.output
