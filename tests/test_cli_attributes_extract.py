"""Tests for ``landscape assess`` (AT-07); SQLite under a temp dir, fake local LLM."""

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
from core.assessment_store import EvidenceRow
from core.attribute_extract import attribute_query
from core.attribute_store import AttributeValueRow
from core.scope_pack import Attribute
from core.chunk_store import store_chunks
from core.chunking import Chunk
from core.entity_store import Entity
from core.runs import FAILED, FINISHED, Run, create_all
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
ATTR = "headcount"


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


TEXT = "Our agency has a headcount of 42 staff translators."


def values(engine) -> list[AttributeValueRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(AttributeValueRow).order_by(AttributeValueRow.id)))


def evidence(engine) -> list[EvidenceRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(EvidenceRow).order_by(EvidenceRow.id)))


def invoke(fake: FakeLLM, *args: str):
    with mock.patch.object(cli, "_make_assess_llm", return_value=fake) as make:
        result = CliRunner().invoke(
            cli.app, ["attributes-extract", "--scope", EXAMPLE_SCOPE, *args]
        )
    return result, make


def test_grounded_int_value_is_stored_with_evidence(engine) -> None:
    entity_id, ids = seed(engine, "acme", [TEXT, "Unrelated words about weather."])
    fake = FakeLLM({"value": 42, "quotes": [{"chunk_id": ids[0], "quote": "headcount of 42"}]})
    result, _ = invoke(fake, "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 0, result.output
    rows = values(engine)
    assert len(rows) == 1 and rows[0].entity_id == entity_id
    ev = evidence(engine)
    assert len(ev) == 1 and ev[0].attribute_value_id == rows[0].id and ev[0].chunk_id == ids[0]
    lines = result.output.splitlines()
    assert lines[0] == "value 42 checks pass chunks 1"
    assert lines[1] == f'quote chunk {ids[0]} grounded "headcount of 42"'
    assert lines[-1] == f"attribute_value {rows[0].id}"
    assert runs(engine) == [("attributes_extract", FINISHED)]
    assert len(fake.prompts) == 1


def test_ungrounded_quote_fails_without_write(engine) -> None:
    entity_id, ids = seed(engine, "acme", [TEXT])
    fake = FakeLLM({"value": 42, "quotes": [{"chunk_id": ids[0], "quote": "invented words"}]})
    result, _ = invoke(fake, "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 1, result.output
    assert values(engine) == [] and evidence(engine) == []
    assert "value 42 checks fail chunks 1" in result.output
    assert f'quote chunk {ids[0]} ungrounded "invented words"' in result.output
    assert "failure quote 0: not found verbatim" in result.output
    assert result.output.splitlines()[-1] == "attribute_value -"
    assert runs(engine) == [("attributes_extract", FINISHED)]


def test_null_value_exits_zero_without_write(engine) -> None:
    entity_id, _ = seed(engine, "acme", [TEXT])
    result, _ = invoke(
        FakeLLM({"value": None, "quotes": []}), "--entity", str(entity_id), "--attribute", ATTR
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["value null checks pass chunks 1", "attribute_value -"]
    assert values(engine) == []


def test_unknown_attribute_exits_2_without_run(engine) -> None:
    entity_id, _ = seed(engine, "acme", [TEXT])
    result, make = invoke(FakeLLM(), "--entity", str(entity_id), "--attribute", "nope")
    assert result.exit_code == 2
    assert runs(engine) == []
    make.assert_not_called()


def test_n_below_one_exits_2(engine) -> None:
    entity_id, _ = seed(engine, "acme", [TEXT])
    result, _ = invoke(FakeLLM(), "--entity", str(entity_id), "--attribute", ATTR, "--n", "0")
    assert result.exit_code == 2
    assert runs(engine) == []


def test_entity_without_chunks_makes_no_llm_call(engine) -> None:
    entity_id, _ = seed(engine, "acme", [])
    fake = FakeLLM({"value": 1, "quotes": []})
    result, _ = invoke(fake, "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 0, result.output
    assert fake.prompts == []
    assert result.output.splitlines() == ["value null checks pass chunks 0", "attribute_value -"]


def test_other_entity_chunks_are_not_used(engine) -> None:
    entity_id, _ = seed(engine, "acme", [])
    seed(engine, "other", [TEXT])
    fake = FakeLLM({"value": 1, "quotes": []})
    result, _ = invoke(fake, "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 0 and fake.prompts == []


def test_llm_error_closes_run_failed(engine) -> None:
    entity_id, _ = seed(engine, "acme", [TEXT])
    fake = FakeLLM(error=LlmError("boom"))
    result, _ = invoke(fake, "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 1
    assert runs(engine) == [("attributes_extract", FAILED)]
    assert values(engine) == []


def test_invalid_output_closes_run_failed(engine) -> None:
    entity_id, _ = seed(engine, "acme", [TEXT])
    result, _ = invoke(FakeLLM("not an object"), "--entity", str(entity_id), "--attribute", ATTR)
    assert result.exit_code == 1
    assert runs(engine) == [("attributes_extract", FAILED)]


def test_dry_run_writes_nothing(engine) -> None:
    result, make = invoke(FakeLLM(), "--entity", "1", "--attribute", ATTR, "--dry-run")
    assert result.exit_code == 0
    assert "stage: attributes-extract" in result.output
    assert runs(engine) == []
    make.assert_not_called()


def test_attribute_query_includes_enum_values() -> None:
    attribute = Attribute(
        id="pricing_model", label="Pricing model", type="enum", values=["per_word", "per_page"]
    )
    assert attribute_query(attribute) == "Pricing model per_word per_page"
    assert attribute_query(Attribute(id="h", label="Headcount", type="int")) == "Headcount"
