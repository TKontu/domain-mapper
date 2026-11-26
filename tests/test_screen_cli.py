"""Tests for ``landscape screen`` (SC-02); SQLite under a temp dir, fake local LLM."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.llm import LlmError, LlmResult
from core import cli
from core.chunk_store import ChunkRow
from core.entity_store import STATUS_UNSCREENED, Entity, iter_entities
from core.runs import Run, create_all
from core.screen_store import ScreenResultRow
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
SEGMENTS = ("agency", "freelancer_network")


@pytest.fixture
def db_url(tmp_path, monkeypatch) -> str:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no stray .env
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    return url


@pytest.fixture
def engine(db_url: str):
    engine = sa.create_engine(db_url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def seed_entity(
    engine,
    name: str,
    groups: list[list[tuple[int, str]]] | None = None,
    *,
    status: str = STATUS_UNSCREENED,
) -> int:
    """Insert one entity and optional snapshot/chunk groups; return the entity id."""
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example",
            name=name,
            primary_domain=f"{name.lower()}.example",
            status=status,
        )
        session.add(entity)
        session.flush()
        entity_id = entity.id
    for group in groups or []:
        with Session(engine) as session, session.begin():
            snapshot = SnapshotRow(
                entity_id=entity_id,
                url="https://acme.example/",
                final_url="https://acme.example/",
                content_type="text/html",
                sha256="0" * 64,
                fetched_at=datetime.now(UTC),
                source_kind="web",
            )
            session.add(snapshot)
            session.flush()
            for seq, text in group:
                session.add(
                    ChunkRow(
                        snapshot_id=snapshot.id,
                        seq=seq,
                        text=text,
                        lang=None,
                        page=None,
                        heading_path=[],
                        char_start=0,
                        char_end=len(text),
                    )
                )
    return entity_id


def answer(
    *,
    in_scope: bool = True,
    segments: list[str] | None = None,
    confidence: float = 0.9,
    reason: str = "because",
) -> dict[str, object]:
    return {
        "in_scope": in_scope,
        "segments": list(SEGMENTS[:1]) if segments is None else segments,
        "confidence": confidence,
        "reason": reason,
    }


class FakeLLM:
    model = "fake-small"

    def __init__(self, answers: dict[str, dict[str, object] | Exception]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        name = prompt.split("Entity: ", 1)[1].split("\n", 1)[0]
        self.calls.append(
            {"prompt": prompt, "purpose": purpose, "run_id": run_id, "schema": json_schema}
        )
        outcome = self.answers[name]
        if isinstance(outcome, Exception):
            raise outcome
        return LlmResult(
            text="{}",
            parsed=outcome,
            tokens_in=1,
            tokens_out=1,
            cost=0.0,
            cache_hit=False,
        )


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["screen", "--scope", EXAMPLE_SCOPE, *args])


def use_llm(monkeypatch, llm: FakeLLM) -> None:
    monkeypatch.setattr(cli, "_make_screen_llm", lambda settings, db: llm)


def result_rows(engine) -> list[tuple[int, int, bool, list[str], float, bool]]:
    with Session(engine) as session:
        return [
            (r.entity_id, r.run_id, r.in_scope, r.segments, r.confidence, r.borderline)
            for r in session.scalars(sa.select(ScreenResultRow).order_by(ScreenResultRow.id))
        ]


def entity_status(engine, entity_id: int) -> str:
    with Session(engine) as session:
        entity = session.get(Entity, entity_id)
        assert entity is not None
        return entity.status


def test_pager_filters_by_status_across_keyset_pages(engine) -> None:
    ids = [
        seed_entity(engine, "Acme"),
        seed_entity(engine, "Beta", status="in_scope"),
        seed_entity(engine, "Gamma"),
        seed_entity(engine, "Delta"),
    ]
    refs = list(iter_entities(engine, "example", batch=1, status=STATUS_UNSCREENED))
    assert [ref.id for ref in refs] == [ids[0], ids[2], ids[3]]
    assert [ref.name for ref in refs] == ["Acme", "Gamma", "Delta"]


def test_screens_entities_and_records_results(engine, monkeypatch) -> None:
    ids = {
        name: seed_entity(engine, name, [[(0, f"{name} page text")]])
        for name in ("Acme", "Beta", "Gamma")
    }
    fake = FakeLLM(
        {
            "Acme": answer(segments=["agency"], confidence=0.9),
            "Beta": answer(in_scope=False, segments=[], confidence=0.9),
            "Gamma": answer(segments=["freelancer_network"], confidence=0.5),
        }
    )
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert f"ok {ids['Acme']} in_scope 0.90 agency" in lines
    assert f"ok {ids['Beta']} out_of_scope 0.90 -" in lines
    assert f"ok {ids['Gamma']} borderline 0.50 freelancer_network" in lines
    assert "screened 3 skipped 0 errors 0" in lines
    assert [(r[0], r[2], r[3], r[4], r[5]) for r in result_rows(engine)] == [
        (ids["Acme"], True, ["agency"], 0.9, False),
        (ids["Beta"], False, [], 0.9, False),
        (ids["Gamma"], True, ["freelancer_network"], 0.5, True),
    ]
    assert entity_status(engine, ids["Acme"]) == "in_scope"
    assert entity_status(engine, ids["Beta"]) == "out_of_scope"
    assert entity_status(engine, ids["Gamma"]) == "borderline"
    with Session(engine) as session:
        runs = list(session.scalars(sa.select(Run)))
    assert len(runs) == 1 and runs[0].stage == "screen" and runs[0].status == "finished"
    run_id = runs[0].id
    assert all(r[1] == run_id for r in result_rows(engine))
    assert [call["purpose"] for call in fake.calls] == ["screen", "screen", "screen"]
    assert [call["run_id"] for call in fake.calls] == [run_id, run_id, run_id]


def test_entity_without_chunks_is_skipped_not_failed(engine, monkeypatch) -> None:
    entity_id = seed_entity(engine, "NoText")
    fake = FakeLLM({})
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 0, result.output
    assert f"skipped {entity_id} no-text" in result.output.splitlines()
    assert "screened 0 skipped 1 errors 0" in result.output
    assert result_rows(engine) == []
    assert entity_status(engine, entity_id) == STATUS_UNSCREENED
    assert fake.calls == []


def test_unknown_segment_is_an_error_and_keeps_entity_unscreened(engine, monkeypatch) -> None:
    entity_id = seed_entity(engine, "Acme", [[(0, "page text")]])
    fake = FakeLLM({"Acme": answer(segments=["not-a-segment"])})
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 1, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("error ")]
    assert len(errors) == 1 and errors[0].startswith(f"error {entity_id} ScreenError: ")
    assert "screened 0 skipped 0 errors 1" in result.output
    assert result_rows(engine) == []
    assert entity_status(engine, entity_id) == STATUS_UNSCREENED
    with Session(engine) as session:
        assert session.scalar(sa.select(Run)).status == "failed"


def test_llm_error_is_reported_and_keeps_entity_unscreened(engine, monkeypatch) -> None:
    entity_id = seed_entity(engine, "Acme", [[(0, "page text")]])
    fake = FakeLLM({"Acme": LlmError("gateway down")})
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 1, result.output
    assert f"error {entity_id} LlmError: gateway down" in result.output
    assert "screened 0 skipped 0 errors 1" in result.output
    assert result_rows(engine) == []
    assert entity_status(engine, entity_id) == STATUS_UNSCREENED


def test_errors_do_not_stop_later_entities(engine, monkeypatch) -> None:
    first = seed_entity(engine, "Acme", [[(0, "page text")]])
    second = seed_entity(engine, "Beta", [[(0, "page text")]])
    fake = FakeLLM(
        {
            "Acme": answer(segments=["not-a-segment"]),
            "Beta": answer(segments=["agency"], confidence=0.9),
        }
    )
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 1, result.output
    assert f"error {first} ScreenError:" in result.output
    assert f"ok {second} in_scope 0.90 agency" in result.output
    assert "screened 1 skipped 0 errors 1" in result.output
    assert entity_status(engine, first) == STATUS_UNSCREENED
    assert entity_status(engine, second) == "in_scope"


def test_rerun_does_not_rescreen_screened_entities(engine, monkeypatch) -> None:
    entity_id = seed_entity(engine, "Acme", [[(0, "page text")]])
    fake = FakeLLM({"Acme": answer(segments=["agency"], confidence=0.9)})
    use_llm(monkeypatch, fake)

    first = invoke()
    second = invoke()

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "screened 0 skipped 0 errors 0" in second.output
    assert len(result_rows(engine)) == 1
    assert len(fake.calls) == 1
    assert entity_status(engine, entity_id) == "in_scope"


def test_limit_bounds_entities_in_id_order(engine, monkeypatch) -> None:
    ids = [seed_entity(engine, name, [[(0, "page text")]]) for name in ("Acme", "Beta", "Gamma")]
    fake = FakeLLM(
        {name: answer(segments=["agency"], confidence=0.9) for name in ("Acme", "Beta", "Gamma")}
    )
    use_llm(monkeypatch, fake)

    result = invoke("--limit", "2")

    assert result.exit_code == 0, result.output
    assert "screened 2 skipped 0 errors 0" in result.output
    assert [r[0] for r in result_rows(engine)] == ids[:2]
    assert entity_status(engine, ids[2]) == STATUS_UNSCREENED
    assert len(fake.calls) == 2


def test_text_is_ordered_by_snapshot_then_seq(engine, monkeypatch) -> None:
    entity_id = seed_entity(
        engine,
        "Acme",
        [
            [(1, "SECOND"), (0, "FIRST")],  # written out of seq order on purpose
            [(0, "THIRD")],
        ],
    )
    fake = FakeLLM({"Acme": answer(segments=["agency"], confidence=0.9)})
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 0, result.output
    prompt = fake.calls[0]["prompt"]
    assert "FIRST\n\nSECOND\n\nTHIRD" in prompt
    assert entity_status(engine, entity_id) == "in_scope"


def test_text_passed_to_llm_is_bounded_by_max_chars(engine, monkeypatch) -> None:
    seed_entity(engine, "Acme", [[(0, "A" * 9000 + "TAIL")]])
    fake = FakeLLM({"Acme": answer(segments=["agency"], confidence=0.9)})
    use_llm(monkeypatch, fake)

    result = invoke()

    assert result.exit_code == 0, result.output
    prompt = fake.calls[0]["prompt"]
    assert "A" * 8000 in prompt
    assert "TAIL" not in prompt


def test_dry_run_prints_plan_and_calls_nothing(engine, monkeypatch, db_url) -> None:
    seed_entity(engine, "Acme", [[(0, "page text")]])
    fake = FakeLLM({"Acme": answer()})
    use_llm(monkeypatch, fake)

    result = invoke("--dry-run")

    assert result.exit_code == 0, result.output
    assert "stage: screen" in result.output
    assert "limit: (none)" in result.output
    assert fake.calls == []
    assert result_rows(engine) == []


def test_invalid_pack_exits_2(tmp_path, monkeypatch, db_url) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(EXAMPLE_SCOPE, broken)
    scope_yaml = broken / "scope.yaml"
    scope_yaml.write_text(scope_yaml.read_text() + "\nunexpected_field: 1\n")
    fake = FakeLLM({})
    use_llm(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["screen", "--scope", str(broken)])

    assert result.exit_code == 2, result.output
    assert "scope.yaml" in result.output
    assert fake.calls == []


def test_unbuildable_llm_exits_2(engine, monkeypatch) -> None:
    seed_entity(engine, "Acme", [[(0, "page text")]])
    monkeypatch.setattr(
        cli, "_make_screen_llm", mock.Mock(side_effect=ValueError("no local model here"))
    )

    result = invoke()

    assert result.exit_code == 2, result.output
    assert "no local model here" in result.output
    with Session(engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(Run)) == 0


def test_builds_local_backend_with_the_database_engine(engine, monkeypatch) -> None:
    seed_entity(engine, "Acme", [[(0, "page text")]])
    fake = FakeLLM({"Acme": answer(segments=["agency"], confidence=0.9)})
    with mock.patch.object(cli, "build_llm", return_value=fake) as build:
        result = invoke()

    assert result.exit_code == 0, result.output
    assert build.call_count == 1
    _settings, backend = build.call_args.args
    assert backend == "local"
    assert isinstance(build.call_args.kwargs["db"], sa.Engine)


def test_budget_exceeded_stops_screening_and_keeps_rest_unscreened(engine, monkeypatch) -> None:
    import adapters.llm as llm_module

    first = seed_entity(engine, "Acme", [[(0, "page text")]])
    second = seed_entity(engine, "Beta", [[(0, "page text")]])
    third = seed_entity(engine, "Gamma", [[(0, "page text")]])
    fake = FakeLLM({name: answer(segments=["agency"]) for name in ("Acme", "Beta", "Gamma")})
    original = fake.complete

    def costly(prompt: str, **kwargs: Any) -> LlmResult:
        result = original(prompt, **kwargs)
        return LlmResult(
            text=result.text,
            parsed=result.parsed,
            tokens_in=1,
            tokens_out=1,
            cost=0.75,
            cache_hit=False,
        )

    fake.complete = costly  # type: ignore[method-assign]
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "0.5")
    monkeypatch.setattr(llm_module, "OpenAICompatibleLLM", lambda *a, **k: fake)

    result = invoke()

    assert result.exit_code == 1, result.output
    assert f"ok {first} in_scope" in result.output
    assert f"error {second} LlmBudgetExceeded:" in result.output
    lines = result.output.splitlines()
    assert not [line for line in lines if line.split()[1:2] == [str(third)]]  # never attempted
    assert len(fake.calls) == 1
    assert "screened 1 skipped 0 errors 1" in result.output
    assert entity_status(engine, second) == STATUS_UNSCREENED
    assert entity_status(engine, third) == STATUS_UNSCREENED


# --- AB-01: abort-close guard for the screen stage -----------------------------


def _screen_runs(engine) -> list[tuple[str, str]]:
    with Session(engine) as session:
        return [(r.stage, r.status) for r in session.scalars(sa.select(Run).order_by(Run.id))]


def _db_error(statement: str) -> sa.exc.OperationalError:
    return sa.exc.OperationalError(statement, {}, Exception("gone"))


def _assert_one_db_line(result) -> None:
    """One stderr line names the close failure; the database URL never appears in the output."""
    assert result.stderr.count("error: database unavailable") == 1
    assert "error: database unavailable (OperationalError)" in result.stderr


def test_runtime_error_with_close_db_error_propagates_original(
    engine, monkeypatch
) -> None:  # AB-01
    seed_entity(engine, "Acme", [[(0, "page text")]])
    use_llm(monkeypatch, FakeLLM({}))
    with mock.patch.object(cli, "screen_entity", side_effect=RuntimeError("boom")):
        with mock.patch.object(cli, "_close_run", side_effect=_db_error("UPDATE")):
            result = invoke()

    assert type(result.exception) is RuntimeError
    _assert_one_db_line(result)


def test_keyboard_interrupt_with_close_db_error_propagates_original(
    engine, monkeypatch
) -> None:  # AB-01
    seed_entity(engine, "Acme", [[(0, "page text")]])
    use_llm(monkeypatch, FakeLLM({}))
    with mock.patch.object(cli, "screen_entity", side_effect=KeyboardInterrupt()):
        with mock.patch.object(cli, "_close_run", side_effect=_db_error("UPDATE")):
            result = invoke()

    assert result.exit_code == 130, result.output
    _assert_one_db_line(result)


def test_db_error_with_close_db_error_exits_2_one_line(
    engine, monkeypatch, db_url
) -> None:  # AB-01
    seed_entity(engine, "Acme", [[(0, "page text")]])
    use_llm(monkeypatch, FakeLLM({}))
    with mock.patch.object(cli, "screen_entity", side_effect=_db_error("SELECT")):
        with mock.patch.object(cli, "_close_run", side_effect=_db_error("UPDATE")):
            result = invoke()

    assert result.exit_code == 2, result.output
    _assert_one_db_line(result)
    assert db_url not in result.output
    assert db_url not in result.stderr


def test_db_error_with_close_ok_exits_2_one_line_run_failed(engine, monkeypatch) -> None:  # AB-01
    seed_entity(engine, "Acme", [[(0, "page text")]])
    use_llm(monkeypatch, FakeLLM({}))
    with mock.patch.object(cli, "screen_entity", side_effect=_db_error("SELECT")):
        result = invoke()

    assert result.exit_code == 2, result.output
    _assert_one_db_line(result)
    assert _screen_runs(engine) == [("screen", "failed")]
