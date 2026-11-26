"""Tests for ``landscape attributes-extract`` (AT-07, AT-08); SQLite temp dir, fake local LLM."""

from __future__ import annotations

import os
import re
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
from app import cli
from core.assessment_store import EvidenceRow
from core.attribute_extract import attribute_query
from core.attribute_store import AttributeValueRow, record_attribute_value
from core.scope_pack import Attribute
from core.chunk_store import ChunkRow, store_chunks
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


# --- AT-08: batch mode (--all) -------------------------------------------------

BATCH_TEXT = "Our agency has a headcount of 42 staff translators."
QUOTE = "headcount of 42"
SCOPE_ID = "example"


class GroundingFake:
    """Fake LLM returning a grounded quote citing the first chunk id in the prompt."""

    model = "fake-local"

    def __init__(self, value: object = 42, quote: str = QUOTE) -> None:
        self.value = value
        self.quote = quote
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
        match = re.search(r"\[chunk (\d+)\]", prompt)
        assert match is not None
        chunk_id = int(match.group(1))
        return LlmResult(
            text="{}",
            parsed={"value": self.value, "quotes": [{"chunk_id": chunk_id, "quote": self.quote}]},
            tokens_in=1,
            tokens_out=1,
            cost=0.0,
            cache_hit=False,
        )


def batch_scope(tmp_path: Path, attributes_yaml: str) -> Path:
    """Copy the neutral ``_example`` pack to tmp and replace ``attributes.yaml``."""
    dst = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, dst)
    (dst / "attributes.yaml").write_text(attributes_yaml)
    return dst


TWO_ATTRS = (
    "attributes:\n"
    "  - id: attr_a\n    label: Headcount\n    type: int\n"
    "  - id: attr_b\n    label: staff translators\n    type: int\n"
)


def seed_status(
    engine, name: str, texts: list[str], status: str, scope_id: str = SCOPE_ID
) -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id=scope_id,
            name=name,
            primary_domain=f"{name}.example",
            status=status,
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
    return entity_id


def invoke_batch(fake, scope: Path, *args: str):
    with mock.patch.object(cli, "_make_assess_llm", return_value=fake) as make:
        result = CliRunner().invoke(cli.app, ["attributes-extract", "--scope", str(scope), *args])
    return result, make


def test_all_two_in_scope_two_attributes_makes_four_calls(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    out_id = seed_status(engine, "gamma", [BATCH_TEXT], "out_of_scope")
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 4
    assert sum("gamma.example" in p for p in fake.prompts) == 0
    # none of the out-of-scope entity's chunks are cited
    out_chunks = [p for p in fake.prompts if f"[chunk {out_id}]" in p]
    assert out_chunks == []
    rows = values(engine)
    assert len(rows) == 4
    assert {r.attribute_id for r in rows} == {"attr_a", "attr_b"}
    assert len(evidence(engine)) == 4
    assert runs(engine) == [("attributes_extract", FINISHED)]
    lines = result.output.splitlines()
    pair_lines = [ln for ln in lines if ln.startswith("entity ")]
    assert len(pair_lines) == 4
    for ln in pair_lines:
        assert re.match(
            r"entity \d+ attribute attr_[ab] value 42 checks pass chunks 1 attribute_value \d+", ln
        ), ln
    assert lines[-1] == "pairs 4 stored 4 null 0 failed 0 errors 0 skipped 0"


def test_all_attribute_filter_makes_two_calls(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all", "--attribute", "attr_a")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 2
    rows = values(engine)
    assert len(rows) == 2 and {r.attribute_id for r in rows} == {"attr_a"}
    assert result.output.splitlines()[-1] == "pairs 2 stored 2 null 0 failed 0 errors 0 skipped 0"


def test_all_limit_caps_entities_not_pairs(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all", "--limit", "1")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 2  # one entity × two attributes
    entities_seen = {r.entity_id for r in values(engine)}
    assert len(entities_seen) == 1
    assert result.output.splitlines()[-1] == "pairs 2 stored 2 null 0 failed 0 errors 0 skipped 0"


def test_all_limit_below_one_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, _ = invoke_batch(GroundingFake(), scope, "--all", "--limit", "0")
    assert result.exit_code == 2
    assert runs(engine) == []


def test_all_together_with_entity_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, make = invoke_batch(GroundingFake(), scope, "--all", "--entity", "1")
    assert result.exit_code == 2
    assert runs(engine) == []
    make.assert_not_called()


def test_neither_all_nor_entity_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, make = invoke_batch(GroundingFake(), scope, "--attribute", "attr_a")
    assert result.exit_code == 2
    assert runs(engine) == []
    make.assert_not_called()


def test_all_unknown_attribute_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, make = invoke_batch(GroundingFake(), scope, "--all", "--attribute", "nope")
    assert result.exit_code == 2
    assert runs(engine) == []
    make.assert_not_called()


def test_all_n_below_one_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, _ = invoke_batch(GroundingFake(), scope, "--all", "--n", "0")
    assert result.exit_code == 2
    assert runs(engine) == []


def test_all_llm_error_one_pair_is_recorded_and_batch_continues(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    second = seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    first_chunk = _first_chunk(engine, first)

    class FlakyFake(GroundingFake):
        def complete(self, prompt, *, purpose, run_id, json_schema=None):
            # fail exactly the first pair (lowest entity id, first attribute in pack order)
            if f"[chunk {first_chunk}]" in prompt and not self.prompts:
                self.prompts.append(prompt)
                raise LlmError("boom")
            return super().complete(prompt, purpose=purpose, run_id=run_id, json_schema=json_schema)

    fake = FlakyFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    error_lines = [ln for ln in lines if " error " in ln]
    assert error_lines == [f"entity {first} attribute attr_a error LlmError: boom"]
    # later pairs still stored
    stored = values(engine)
    assert len(stored) == 3
    assert any(r.entity_id == second for r in stored)
    assert any(r.entity_id == first and r.attribute_id == "attr_b" for r in stored)
    assert runs(engine) == [("attributes_extract", FINISHED)]
    assert lines[-1] == "pairs 4 stored 3 null 0 failed 0 errors 1 skipped 0"


def test_all_invalid_output_one_pair_is_recorded_and_batch_continues(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")

    class BadOnce(GroundingFake):
        def __init__(self):
            super().__init__(value=42)
            self.calls = 0

        def complete(self, prompt, *, purpose, run_id, json_schema=None):
            self.calls += 1
            if self.calls == 1:
                self.prompts.append(prompt)
                return LlmResult(
                    text="{}", parsed="not an object", tokens_in=1, tokens_out=1,
                    cost=0.0, cache_hit=False,
                )
            return super().complete(prompt, purpose=purpose, run_id=run_id, json_schema=json_schema)

    result, _ = invoke_batch(BadOnce(), scope, "--all")
    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    assert any(" error AttributeExtractError:" in ln for ln in lines)
    assert len(values(engine)) == 3
    assert runs(engine) == [("attributes_extract", FINISHED)]
    assert lines[-1] == "pairs 4 stored 3 null 0 failed 0 errors 1 skipped 0"


def test_all_ungrounded_pair_exits_1_not_written_others_written(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    first_chunk = _first_chunk(engine, first)

    class UngroundedOnce(GroundingFake):
        def complete(self, prompt, *, purpose, run_id, json_schema=None):
            match = re.search(r"\[chunk (\d+)\]", prompt)
            assert match is not None
            cid = int(match.group(1))
            self.prompts.append(prompt)
            quote = "invented words" if cid == first_chunk and len(self.prompts) == 1 else QUOTE
            return LlmResult(
                text="{}",
                parsed={"value": 42, "quotes": [{"chunk_id": cid, "quote": quote}]},
                tokens_in=1, tokens_out=1, cost=0.0, cache_hit=False,
            )

    result, _ = invoke_batch(UngroundedOnce(), scope, "--all")
    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    # the ungrounded pair is a checks-fail line, not an error line, and writes nothing
    fail_lines = [ln for ln in lines if " checks fail " in ln]
    assert len(fail_lines) == 1
    assert fail_lines[0].endswith("attribute_value -")
    stored = values(engine)
    assert len(stored) == 3
    assert not any(r.entity_id == first and r.attribute_id == "attr_a" for r in stored)
    assert runs(engine) == [("attributes_extract", FINISHED)]
    assert lines[-1] == "pairs 4 stored 3 null 0 failed 1 errors 0 skipped 0"


def test_all_null_value_counts_null_exit_0(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    # null value with no quotes -> checks pass, nothing stored
    class NullFake(GroundingFake):
        def complete(self, prompt, *, purpose, run_id, json_schema=None):
            self.prompts.append(prompt)
            return LlmResult(
                text="{}", parsed={"value": None, "quotes": []},
                tokens_in=1, tokens_out=1, cost=0.0, cache_hit=False,
            )

    result, _ = invoke_batch(NullFake(), scope, "--all")
    assert result.exit_code == 0, result.output
    assert values(engine) == []
    assert result.output.splitlines()[-1] == "pairs 2 stored 0 null 2 failed 0 errors 0 skipped 0"


def test_all_dry_run_writes_nothing(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, make = invoke_batch(GroundingFake(), scope, "--all", "--dry-run")
    assert result.exit_code == 0
    assert "stage: attributes-extract" in result.output
    assert runs(engine) == []
    make.assert_not_called()


def test_all_one_run_row_covers_whole_batch(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    result, _ = invoke_batch(GroundingFake(value=42), scope, "--all")
    assert result.exit_code == 0, result.output
    assert runs(engine) == [("attributes_extract", FINISHED)]


def _first_chunk(engine, entity_id: int) -> int:
    with engine.connect() as conn:
        row = conn.execute(
            sa.text(
                "SELECT c.id FROM chunk c JOIN snapshot s ON c.snapshot_id = s.id "
                "WHERE s.entity_id = :e ORDER BY c.id LIMIT 1"
            ),
            {"e": entity_id},
        ).first()
    assert row is not None
    return int(row[0])


# --- AT-09: --skip-existing (batch mode only) ---------------------------------


def _prestore(engine, entity_id: int, attribute: Attribute, value: int) -> None:
    with Session(engine) as session, session.begin():
        record_attribute_value(session, entity_id, attribute, value)


def test_all_skip_existing_skips_prestored_pair_three_calls(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    prestored = Attribute(id="attr_a", label="Headcount", type="int")
    _prestore(engine, first, prestored, 7)
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all", "--skip-existing")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 3
    lines = result.output.splitlines()
    assert f"entity {first} attribute attr_a skipped existing" in lines
    # no retrieval/LLM for the skipped pair: no pair line for it, no prompt citing its chunks
    assert not any(ln.startswith(f"entity {first} attribute attr_a value ") for ln in lines)
    assert sum(f"entity {first} attribute attr_a" in p for p in fake.prompts) == 0
    assert lines[-1] == "pairs 4 stored 3 null 0 failed 0 errors 0 skipped 1"
    # pre-stored value unchanged: no new row for the skipped pair
    rows = [r for r in values(engine) if r.entity_id == first and r.attribute_id == "attr_a"]
    assert len(rows) == 1 and rows[0].value_json == 7
    assert len(values(engine)) == 4  # 1 pre-stored + 3 stored this run
    assert runs(engine) == [("attributes_extract", FINISHED)]


def test_all_without_skip_existing_makes_four_calls_skipped_zero(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    prestored = Attribute(id="attr_a", label="Headcount", type="int")
    _prestore(engine, first, prestored, 7)
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 4
    lines = result.output.splitlines()
    assert not any(ln.endswith("skipped existing") for ln in lines)
    assert lines[-1] == "pairs 4 stored 4 null 0 failed 0 errors 0 skipped 0"
    # without the flag the pre-stored pair is re-extracted: a new row is appended
    rows = [r for r in values(engine) if r.entity_id == first and r.attribute_id == "attr_a"]
    assert len(rows) == 2 and rows[-1].value_json == 42


def test_skip_existing_with_entity_exits_2_no_run(engine, tmp_path) -> None:
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, make = invoke_batch(
        GroundingFake(),
        scope,
        "--entity",
        str(first),
        "--attribute",
        "attr_a",
        "--skip-existing",
    )
    assert result.exit_code == 2, result.output
    assert runs(engine) == []
    make.assert_not_called()
    assert values(engine) == []


# --- RT-01: batch runs each pair through StageRunner -----------------------------


def _extract_units(engine) -> list[tuple[str, str, int, int]]:
    from core.runs import WorkUnit

    with Session(engine) as session:
        return [
            (u.unit_key, u.status, u.attempts, u.run_id)
            for u in session.scalars(
                sa.select(WorkUnit)
                .where(WorkUnit.stage == "attributes_extract")
                .order_by(WorkUnit.id)
            )
        ]


def _run_ids(engine) -> list[int]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(Run.id).order_by(Run.id)))


def _add_chunk(engine, entity_id: int, text: str) -> None:
    with Session(engine) as session, session.begin():
        snap = SnapshotRow(
            entity_id=entity_id,
            url="https://extra.example/",
            final_url="https://extra.example/",
            content_type="text/html",
            sha256="1" * 64,
            fetched_at=datetime.now(UTC),
            source_kind="web",
        )
        session.add(snap)
        session.flush()
        snap_id = snap.id
    store_chunks(
        engine,
        snap_id,
        [Chunk(seq=0, text=text, page=None, heading_path=("S",), char_start=0, char_end=len(text))],
    )


def test_all_records_done_work_units_under_the_run(engine, tmp_path) -> None:  # RT-01
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    second = seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    result, _ = invoke_batch(GroundingFake(value=42), scope, "--all")
    assert result.exit_code == 0, result.output
    run_id = _run_ids(engine)[0]
    assert _extract_units(engine) == [
        (f"{first}:attr_a", "done", 1, run_id),
        (f"{first}:attr_b", "done", 1, run_id),
        (f"{second}:attr_a", "done", 1, run_id),
        (f"{second}:attr_b", "done", 1, run_id),
    ]


def test_all_rerun_skips_done_and_retries_errored_pair(engine, tmp_path) -> None:  # RT-01
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    second = seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    first_chunk = _first_chunk(engine, first)

    class FlakyFake(GroundingFake):
        def complete(self, prompt, *, purpose, run_id, json_schema=None):
            if f"[chunk {first_chunk}]" in prompt and not self.prompts:
                self.prompts.append(prompt)
                raise LlmError("boom")
            return super().complete(prompt, purpose=purpose, run_id=run_id, json_schema=json_schema)

    result, _ = invoke_batch(FlakyFake(value=42), scope, "--all")
    assert result.exit_code == 1, result.output
    assert (f"{first}:attr_a", "failed", 1) in [u[:3] for u in _extract_units(engine)]

    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 1
    pair_lines = [ln for ln in result.output.splitlines() if ln.startswith("entity ")]
    assert len(pair_lines) == 1 and pair_lines[0].startswith(f"entity {first} attribute attr_a ")
    assert result.output.splitlines()[-1] == "pairs 4 stored 1 null 0 failed 0 errors 0 skipped 3"
    first_run, second_run = _run_ids(engine)
    assert _extract_units(engine) == [
        (f"{first}:attr_a", "done", 2, second_run),
        (f"{first}:attr_b", "done", 1, first_run),
        (f"{second}:attr_a", "done", 1, first_run),
        (f"{second}:attr_b", "done", 1, first_run),
    ]


def test_all_failed_checks_pair_is_done_not_retried(engine, tmp_path) -> None:  # RT-01
    scope = batch_scope(tmp_path, TWO_ATTRS)
    seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    result, _ = invoke_batch(GroundingFake(value=42, quote="not in the text"), scope, "--all")
    assert result.exit_code == 1, result.output
    assert {u[1] for u in _extract_units(engine)} == {"done"}
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert fake.prompts == []
    assert result.output.splitlines()[-1] == "pairs 2 stored 0 null 0 failed 0 errors 0 skipped 2"


def test_all_new_chunk_re_extracts_entity_pairs(engine, tmp_path) -> None:  # RT-01
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    assert invoke_batch(GroundingFake(value=42), scope, "--all")[0].exit_code == 0
    _add_chunk(engine, first, BATCH_TEXT)
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 2
    pair_lines = [ln for ln in result.output.splitlines() if ln.startswith("entity ")]
    assert all(ln.startswith(f"entity {first} ") for ln in pair_lines) and len(pair_lines) == 2
    assert result.output.splitlines()[-1] == "pairs 4 stored 2 null 0 failed 0 errors 0 skipped 2"


def test_all_chunk_text_edited_in_place_re_extracts_entity_pairs(engine, tmp_path) -> None:  # RT-01
    scope = batch_scope(tmp_path, TWO_ATTRS)
    first = seed_status(engine, "acme", [BATCH_TEXT], "in_scope")
    seed_status(engine, "beta", [BATCH_TEXT], "in_scope")
    assert invoke_batch(GroundingFake(value=42), scope, "--all")[0].exit_code == 0
    chunk_id = _first_chunk(engine, first)
    with Session(engine) as session, session.begin():
        session.get(ChunkRow, chunk_id).text = "Edited: our headcount of 42 staff translators."
    fake = GroundingFake(value=42)
    result, _ = invoke_batch(fake, scope, "--all")
    assert result.exit_code == 0, result.output
    assert len(fake.prompts) == 2
    pair_lines = [ln for ln in result.output.splitlines() if ln.startswith("entity ")]
    assert all(ln.startswith(f"entity {first} ") for ln in pair_lines) and len(pair_lines) == 2
    assert result.output.splitlines()[-1] == "pairs 4 stored 2 null 0 failed 0 errors 0 skipped 2"
