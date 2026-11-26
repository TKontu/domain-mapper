"""Tests for ``landscape scores`` (AG-02, AG-03); SQLite under a temp dir, CLI via CliRunner."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.runs import Run, create_all
from core.scope_pack import load_scope_pack
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


def add(engine, run_id, entity_id, criterion, status, version=VERSION, grounded=True) -> int:
    quotes = (Quote(1, "q", grounded, "exact" if grounded else None, (0, 1) if grounded else None),)
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status=status, quotes=quotes, rationale="x", checks_passed=True, failures=()
        ),
        route=Route("accept", None, "r"),
        rubric_version=version,
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


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["scores", "--scope", EXAMPLE_SCOPE, *args])


def test_scores_from_stored_screen_segments(engine) -> None:
    rid = add_run(engine)
    add_screen(engine, rid, 5, ["agency"])
    add(engine, rid, 5, "iso17100", "strong")
    add(engine, rid, 5, "public_terms", "weak", grounded=False)
    result = invoke("--entity", "5")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"entity 5 segments agency rubric {VERSION}",
        "topic certification score 3.0000 coverage 1.0000 applicable 1 scored 1 evidenced 1",
        "topic terms score 1.0000 coverage 0.0000 applicable 1 scored 1 evidenced 0",
    ]


def test_segment_option_overrides_stored_result(engine) -> None:
    rid = add_run(engine)
    add_screen(engine, rid, 5, ["agency"])
    add(engine, rid, 5, "iso17100", "strong")
    add(engine, rid, 5, "public_terms", "standard")
    result = invoke("--entity", "5", "--segment", "freelancer_network")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"entity 5 segments freelancer_network rubric {VERSION}",
        "topic certification score - coverage - applicable 0 scored 0 evidenced 0",
        "topic terms score 2.0000 coverage 1.0000 applicable 1 scored 1 evidenced 1",
    ]


def test_unknown_segment_exits_2(engine) -> None:
    result = invoke("--entity", "5", "--segment", "agency", "--segment", "bogus")
    assert result.exit_code == 2
    assert "bogus" in result.stderr


def test_no_screen_result_exits_2(engine) -> None:
    result = invoke("--entity", "7")
    assert result.exit_code == 2
    assert result.stderr == "error: entity 7 has no screen result; pass --segment\n"


def test_other_rubric_version_ignored(engine) -> None:
    rid = add_run(engine)
    add(engine, rid, 5, "iso17100", "strong", version="9.9")
    result = invoke("--entity", "5", "--segment", "agency")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1] == (
        "topic certification score - coverage 0.0000 applicable 1 scored 0 evidenced 0"
    )


def test_no_assessments_prints_dash_scores(engine) -> None:
    result = invoke("--entity", "5", "--segment", "agency")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"entity 5 segments agency rubric {VERSION}",
        "topic certification score - coverage 0.0000 applicable 1 scored 0 evidenced 0",
        "topic terms score - coverage 0.0000 applicable 1 scored 0 evidenced 0",
    ]


def test_score_topics_value_error_exits_1(engine) -> None:
    rid = add_run(engine)
    add(engine, rid, 5, "iso17100", "bogus_status")
    result = invoke("--entity", "5", "--segment", "agency")
    assert result.exit_code == 1
    assert result.stderr.startswith("error: criterion 'iso17100' has unknown status")


def test_json_prints_one_object_of_topic_scores(engine) -> None:
    rid = add_run(engine)
    add_screen(engine, rid, 5, ["agency"])
    add(engine, rid, 5, "iso17100", "strong")
    add(engine, rid, 5, "public_terms", "weak", grounded=False)
    result = invoke("--entity", "5", "--format", "json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {
        "entity_id": 5,
        "segments": ["agency"],
        "rubric_version": VERSION,
        "topics": [
            {
                "topic_id": "certification",
                "score": 3.0,
                "coverage": 1.0,
                "applicable": 1,
                "scored": 1,
                "evidenced": 1,
            },
            {
                "topic_id": "terms",
                "score": 1.0,
                "coverage": 0.0,
                "applicable": 1,
                "scored": 1,
                "evidenced": 0,
            },
        ],
    }


def test_json_topics_follow_rubric_order(engine) -> None:
    pack = load_scope_pack(Path(EXAMPLE_SCOPE))
    result = invoke("--entity", "5", "--segment", "agency", "--format", "json")
    assert result.exit_code == 0, result.output
    topics = json.loads(result.stdout)["topics"]
    assert [topic["topic_id"] for topic in topics] == [t.id for t in pack.rubric.topics]


def test_json_without_assessments_has_null_score(engine) -> None:
    result = invoke("--entity", "5", "--segment", "agency", "--format", "json")
    assert result.exit_code == 0, result.output
    topics = {topic["topic_id"]: topic for topic in json.loads(result.stdout)["topics"]}
    assert topics["certification"]["score"] is None
    assert topics["certification"]["coverage"] == 0.0
    assert topics["terms"]["score"] is None
    assert topics["terms"]["coverage"] == 0.0


def test_json_zero_applicable_topic_has_null_score_and_coverage(engine) -> None:
    result = invoke("--entity", "5", "--segment", "freelancer_network", "--format", "json")
    assert result.exit_code == 0, result.output
    topics = {topic["topic_id"]: topic for topic in json.loads(result.stdout)["topics"]}
    assert topics["certification"] == {
        "topic_id": "certification",
        "score": None,
        "coverage": None,
        "applicable": 0,
        "scored": 0,
        "evidenced": 0,
    }


def test_format_text_is_default_output(engine) -> None:
    rid = add_run(engine)
    add_screen(engine, rid, 5, ["agency"])
    add(engine, rid, 5, "iso17100", "strong")
    add(engine, rid, 5, "public_terms", "standard")
    default = invoke("--entity", "5")
    text = invoke("--entity", "5", "--format", "text")
    assert default.exit_code == text.exit_code == 0
    assert text.stdout == default.stdout
    assert text.stdout.splitlines()[0] == f"entity 5 segments agency rubric {VERSION}"


def test_unknown_format_exits_2(engine) -> None:
    result = invoke("--entity", "5", "--format", "yaml")
    assert result.exit_code == 2
    assert result.stderr == "error: unknown format 'yaml'\n"


def test_database_unavailable_exits_2(engine, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OperationalError("select", {}, Exception("down"))

    monkeypatch.setattr(cli, "latest_screen_segments", boom)
    result = invoke("--entity", "5")
    assert result.exit_code == 2
    assert result.stderr == "error: database unavailable (OperationalError)\n"
