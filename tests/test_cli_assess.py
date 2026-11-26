"""Tests for ``landscape assess`` (AS-03); SQLite under a temp dir, fake local LLM."""

from __future__ import annotations

import os
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
from core.chunk_store import store_chunks
from core.chunking import Chunk
from core.entity_store import Entity
from core.runs import FAILED, FINISHED, Run, create_all
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
CRIT = "public_terms"
OTHER_MARKER = "OTHER-ENTITY-MARKER-zz9"


class FakeLLM:
    model = "fake-local"

    def __init__(self, parsed: object = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return LlmResult(
            text="{}",
            parsed=self.parsed,  # type: ignore[arg-type]
            tokens_in=1,
            tokens_out=1,
            cost=0.0,
            cache_hit=False,
        )


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    engine = sa.create_engine(url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def seed(engine, name: str, texts: list[str]) -> tuple[int, list[int]]:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example", name=name, primary_domain=f"{name}.example", status="in_scope"
        )
        session.add(entity)
        session.flush()
        entity_id = entity.id
        snap = SnapshotRow(
            entity_id=entity_id,
            url=f"https://{name}.example/",
            final_url=f"https://{name}.example/",
            content_type="text/html",
            sha256="0" * 64,
            fetched_at=datetime.now(UTC),
            source_kind="web",
        )
        session.add(snap)
        session.flush()
        snap_id = snap.id
    chunks = [
        Chunk(seq=i, text=t, page=None, heading_path=("S",), char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]
    store_chunks(engine, snap_id, chunks)
    with engine.connect() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                sa.text("SELECT id FROM chunk WHERE snapshot_id = :s ORDER BY seq"), {"s": snap_id}
            )
        ]
    return entity_id, ids


def runs(engine) -> list[tuple[str, str]]:
    with Session(engine) as session:
        return [(r.stage, r.status) for r in session.query(Run).all()]


def invoke(llm: FakeLLM | None, *args: str) -> Any:
    fake = llm or FakeLLM()
    with mock.patch.object(cli, "_make_assess_llm", return_value=fake):
        return CliRunner().invoke(cli.app, ["assess", "--scope", EXAMPLE_SCOPE, *args])


def test_happy_path_grounded(engine) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    seed(engine, "other", [f"Terms and conditions {OTHER_MARKER}."])
    llm = FakeLLM(
        {
            "status": "strong",
            "quotes": [{"chunk_id": ids[0], "quote": "general terms and conditions"}],
            "rationale": "Published.",
        }
    )
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "status strong checks pass chunks 1"
    assert lines[1] == f'quote chunk {ids[0]} grounded exact "general terms and conditions"'
    assert lines[2] == "rationale Published."
    assert runs(engine) == [("assess", FINISHED)]
    # (f) only this entity's chunks reach the prompt
    assert len(llm.prompts) == 1 and OTHER_MARKER not in llm.prompts[0]


def test_ungrounded_fails_checks(engine) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    llm = FakeLLM(
        {
            "status": "strong",
            "quotes": [{"chunk_id": ids[0], "quote": "invented liability text"}],
            "rationale": "r",
        }
    )
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert "status strong checks fail chunks 1" in result.output
    assert f'quote chunk {ids[0]} ungrounded - "invented liability text"' in result.output
    assert "failure quote 0: not found verbatim" in result.output
    assert "failure evidence_min:" in result.output
    assert runs(engine) == [("assess", FINISHED)]


def test_no_chunks_is_silent(engine) -> None:
    eid, _ = seed(engine, "acme", ["Nothing relevant at all."])
    llm = FakeLLM()
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "status silent checks pass chunks 0"
    assert llm.prompts == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--criterion", "nope"],
        ["--criterion", CRIT, "--n", "0"],
        ["--criterion", CRIT, "--n", "5", "--k", "4"],
        [],
    ],
)
def test_bad_args_exit_2_no_run(engine, extra: list[str]) -> None:
    result = invoke(None, "--entity", "1", *extra)
    assert result.exit_code == 2, result.output
    assert runs(engine) == []
    if extra[:2] == ["--criterion", "nope"]:
        assert "error: unknown criterion nope" in result.output


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(error=LlmError("boom")), FakeLLM({"status": "bogus", "quotes": [], "rationale": ""})],
)
def test_llm_failure_exit_1_run_failed(engine, llm: FakeLLM) -> None:
    eid, _ = seed(engine, "acme", ["Our general terms and conditions are published here."])
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert "error LlmError: boom" in result.output or "error AssessError:" in result.output
    assert runs(engine) == [("assess", FAILED)]


def test_database_unavailable(tmp_path, monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/missing/dir/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    result = invoke(None, "--entity", "1", "--criterion", CRIT)
    assert result.exit_code == 2, result.output
    assert "database unavailable" in result.output
    assert url not in result.output


def test_dry_run_still_works(engine) -> None:
    result = CliRunner().invoke(cli.app, ["assess", "--scope", EXAMPLE_SCOPE, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "stage: assess" in result.output
