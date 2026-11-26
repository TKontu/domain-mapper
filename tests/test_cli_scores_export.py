"""Tests for ``landscape scores-export`` (AG-04); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.entity_store import Entity
from core.runs import Run, create_all
from core.screen_store import ScreenResultRow
from core.verify import Route

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
VERSION = "0.0.1"  # scopes/_example/scope.yaml version


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


def add_entity(engine: Engine, name: str, scope_id: str = "example") -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id=scope_id, name=name, primary_domain=f"{name}.example", status="in_scope"
        )
        session.add(entity)
        session.flush()
        return entity.id


def add(engine, run_id, entity_id, criterion, status, grounded=True) -> None:
    quotes = (Quote(1, "q", grounded, "exact" if grounded else None, (0, 1) if grounded else None),)
    record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status=status, quotes=quotes, rationale="x", checks_passed=True, failures=()
        ),
        route=Route("accept", None, "r"),
        rubric_version=VERSION,
        extractor_model="e",
        verifier_model="v",
        run_id=run_id,
    )


def add_screen(engine, run_id, entity_id, segments) -> None:
    with Session(engine) as session, session.begin():
        session.add(
            ScreenResultRow(
                entity_id=entity_id,
                run_id=run_id,
                in_scope=True,
                segments=segments,
                confidence=0.9,
                reason="r",
                borderline=False,
            )
        )


def export(out: Path):
    return CliRunner().invoke(
        cli.app, ["scores-export", "--scope", EXAMPLE_SCOPE, "--out", str(out)]
    )


def scores_json(entity_id: int) -> dict:
    result = CliRunner().invoke(
        cli.app,
        ["scores", "--scope", EXAMPLE_SCOPE, "--entity", str(entity_id), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_two_screened_entities_match_scores_json(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    b = add_entity(engine, "beta")
    add_screen(engine, rid, a, ["agency"])
    add_screen(engine, rid, b, ["freelancer_network"])
    add(engine, rid, a, "iso17100", "strong")
    add(engine, rid, a, "public_terms", "weak", grounded=False)
    add(engine, rid, b, "public_terms", "standard")
    out = tmp_path / "scores.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [scores_json(a), scores_json(b)]
    assert result.stdout == f"exported 2 entities to {out} (skipped 0 without screen result)\n"
    assert not Path(f"{out}.tmp").exists()


def test_unscreened_entity_is_skipped_and_counted(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    add_entity(engine, "beta")
    add_entity(engine, "other", scope_id="elsewhere")
    add_screen(engine, rid, a, ["agency"])
    out = tmp_path / "scores.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["entity_id"] for line in lines] == [a]
    assert result.stdout == f"exported 1 entities to {out} (skipped 1 without screen result)\n"


def test_screened_entity_without_assessments_has_null_scores(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    add_screen(engine, rid, a, ["agency"])
    out = tmp_path / "scores.jsonl"
    assert export(out).exit_code == 0
    (line,) = out.read_text(encoding="utf-8").splitlines()
    topics = json.loads(line)["topics"]
    assert topics and all(t["score"] is None for t in topics)
    assert all(t["coverage"] in (None, 0.0) for t in topics)
    assert json.loads(line) == scores_json(a)


def test_empty_scope_writes_empty_file(engine, tmp_path) -> None:
    out = tmp_path / "scores.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == ""
    assert result.stdout == f"exported 0 entities to {out} (skipped 0 without screen result)\n"


def test_unwritable_directory_exits_2_and_leaves_nothing(engine, tmp_path) -> None:
    out = tmp_path / "missing" / "scores.jsonl"
    result = export(out)
    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_failure_mid_stream_leaves_no_output(engine, tmp_path, monkeypatch) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    b = add_entity(engine, "beta")
    add_screen(engine, rid, a, ["agency"])
    add_screen(engine, rid, b, ["agency"])
    real = cli._scores_object
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "_scores_object", boom)
    out = tmp_path / "scores.jsonl"
    result = export(out)
    assert result.exit_code != 0
    assert len(calls) == 2
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_export_batches_queries_and_matches_single_batch(engine, tmp_path, monkeypatch) -> None:
    rid = add_run(engine)
    ids = [add_entity(engine, f"e{i}") for i in range(5)]
    for i, entity_id in enumerate(ids):
        if i == 2:
            continue  # unscreened: skipped in both runs
        add_screen(engine, rid, entity_id, ["agency"])
        add(engine, rid, entity_id, "iso17100", "strong")

    monkeypatch.setattr(cli, "SCORES_EXPORT_BATCH", 500)
    out_all = tmp_path / "all.jsonl"
    full = export(out_all)
    assert full.exit_code == 0, full.output

    calls: list[list[int]] = []
    real = cli.latest_screen_segments_many

    def counting(engine_, entity_ids):
        calls.append(list(entity_ids))
        return real(engine_, entity_ids)

    assessment_calls: list[list[int]] = []
    real_assessments = cli.iter_current_assessments

    def counting_assessments(engine_, **kwargs):
        assessment_calls.append(list(kwargs["entity_ids"]))
        return real_assessments(engine_, **kwargs)

    monkeypatch.setattr(cli, "SCORES_EXPORT_BATCH", 2)
    monkeypatch.setattr(cli, "latest_screen_segments_many", counting)
    monkeypatch.setattr(cli, "iter_current_assessments", counting_assessments)
    out_batched = tmp_path / "batched.jsonl"
    batched = export(out_batched)
    assert batched.exit_code == 0, batched.output
    assert calls == [ids[0:2], ids[2:4], ids[4:5]]
    assert assessment_calls == calls
    assert out_batched.read_bytes() == out_all.read_bytes()
    assert batched.stdout == (
        f"exported 4 entities to {out_batched} (skipped 1 without screen result)\n"
    )
    assert full.stdout == (
        f"exported 4 entities to {out_all} (skipped 1 without screen result)\n"
    )
