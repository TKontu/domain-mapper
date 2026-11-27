"""RT-01: single-pair ``landscape assess`` runs through StageRunner (assignment 137/A1)."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session

from adapters.llm import LlmError
from core.runs import DONE, FAILED, FINISHED, WorkUnit
from tests.test_cli_assess import (  # noqa: F401  (engine fixture)
    CRIT,
    FakeLLM,
    _grounded,
    assessments,
    engine,
    invoke,
    runs,
    seed,
)


def units(engine) -> list[tuple[str, str, str]]:
    with Session(engine) as session:
        return [
            (u.stage, u.unit_key, u.status)
            for u in session.scalars(sa.select(WorkUnit).order_by(WorkUnit.id))
        ]


TEXT = "Our general terms and conditions are published here."


def test_first_run_one_done_unit_then_rerun_skips(engine) -> None:
    eid, ids = seed(engine, "acme", [TEXT])
    first = invoke(FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT)
    assert first.exit_code == 0, first.output
    assert first.output.splitlines()[0] == "status strong checks pass chunks 1"
    assert runs(engine) == [("assess", FINISHED)]
    assert units(engine) == [("assess", f"{eid}:{CRIT}", DONE)]

    llm = FakeLLM(_grounded(ids))
    again = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert again.exit_code == 0, again.output
    assert again.output.splitlines() == ["skipped 1"]
    assert llm.prompts == []
    assert len(assessments(engine)) == 1
    assert runs(engine) == [("assess", FINISHED), ("assess", FINISHED)]


def test_changed_chunk_text_reruns(engine) -> None:
    eid, ids = seed(engine, "acme", [TEXT])
    assert invoke(FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT).exit_code == 0
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE chunk SET text = :t WHERE id = :i"),
            {"t": TEXT + " Updated.", "i": ids[0]},
        )
    llm = FakeLLM(_grounded(ids))
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert "skipped 1" not in result.output
    assert len(llm.prompts) == 1 and len(assessments(engine)) == 2


def test_changed_n_reruns(engine) -> None:
    eid, ids = seed(engine, "acme", [TEXT])
    assert invoke(FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT).exit_code == 0
    llm = FakeLLM(_grounded(ids))
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT, "--n", "3")
    assert result.exit_code == 0, result.output
    assert len(llm.prompts) == 1 and len(assessments(engine)) == 2


def test_changed_criterion_reruns(engine, monkeypatch) -> None:
    from app import cli

    eid, ids = seed(engine, "acme", [TEXT])
    assert invoke(FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT).exit_code == 0
    real = cli._find_criterion

    def edited(rubric, cid):
        found = real(rubric, cid)
        return found.model_copy(update={"question": "Edited question?"}) if found else found

    monkeypatch.setattr(cli, "_find_criterion", edited)
    llm = FakeLLM(_grounded(ids))
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert "skipped 1" not in result.output
    assert len(llm.prompts) == 1 and len(assessments(engine)) == 2


def test_failed_checks_unit_done_exit_1(engine) -> None:
    eid, ids = seed(engine, "acme", [TEXT])
    bad = {"status": "strong", "quotes": [{"chunk_id": ids[0], "quote": "nope"}], "rationale": "r"}
    result = invoke(FakeLLM(bad), "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert units(engine) == [("assess", f"{eid}:{CRIT}", DONE)]
    assert runs(engine) == [("assess", FINISHED)]


def test_llm_error_unit_and_run_failed_then_retried(engine) -> None:
    eid, ids = seed(engine, "acme", [TEXT])
    result = invoke(FakeLLM(error=LlmError("boom")), "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert units(engine) == [("assess", f"{eid}:{CRIT}", FAILED)]
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []
    llm = FakeLLM(_grounded(ids))
    retry = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert retry.exit_code == 0, retry.output
    assert len(llm.prompts) == 1
    assert units(engine) == [("assess", f"{eid}:{CRIT}", DONE)]


def test_setup_error_no_run_or_unit(engine) -> None:
    eid, _ = seed(engine, "acme", [TEXT])
    result = invoke(FakeLLM(), "--entity", str(eid), "--criterion", "no_such")
    assert result.exit_code == 2
    result = invoke(FakeLLM(), "--entity", str(eid), "--criterion", CRIT, "--n", "0")
    assert result.exit_code == 2
    assert runs(engine) == [] and units(engine) == []
