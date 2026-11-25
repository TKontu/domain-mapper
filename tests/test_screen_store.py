"""Tests for core.screen_store (SC-02): screen result rows and bounded entity text."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.entity_store import STATUS_UNSCREENED, Entity
from core.runs import Run, create_all
from core.screen import ScreenResult
from core.screen_store import (
    DEFAULT_MAX_CHARS,
    STATUS_BORDERLINE,
    STATUS_IN_SCOPE,
    STATUS_OUT_OF_SCOPE,
    ScreenResultRow,
    load_entity_text,
    record_screen_result,
    status_for_result,
)
from core.snapshot_store import SnapshotRow


@pytest.fixture
def engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path}/dm.sqlite")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def add_entity(engine, name="Acme", *, status=STATUS_UNSCREENED) -> int:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example",
            name=name,
            primary_domain=f"{name.lower()}.example",
            status=status,
        )
        session.add(entity)
        session.flush()
        return entity.id


def add_run(engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="screen", status="running", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def add_chunks(engine, entity_id: int, chunks: list[list[tuple[int, str]]]) -> list[int]:
    """Insert one snapshot per ``chunks`` group, in the given order; return snapshot ids."""
    snapshot_ids: list[int] = []
    with Session(engine) as session, session.begin():
        for group in chunks:
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
            snapshot_ids.append(snapshot.id)
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
    return snapshot_ids


def test_create_all_creates_screen_result_table(engine) -> None:
    assert "screen_result" in sa.inspect(engine).get_table_names()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            ScreenResult(
                in_scope=True,
                segments=("agency",),
                confidence=0.9,
                reason="fits",
                borderline=False,
            ),
            STATUS_IN_SCOPE,
        ),
        (
            ScreenResult(
                in_scope=False,
                segments=(),
                confidence=0.9,
                reason="does not fit",
                borderline=False,
            ),
            STATUS_OUT_OF_SCOPE,
        ),
        (
            ScreenResult(
                in_scope=True,
                segments=("agency",),
                confidence=0.5,
                reason="unknown",
                borderline=True,
            ),
            STATUS_BORDERLINE,
        ),
    ],
)
def test_status_for_result(result: ScreenResult, expected: str) -> None:
    assert status_for_result(result) == expected


def test_record_writes_one_row_and_updates_entity_status(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    result = ScreenResult(
        in_scope=True,
        segments=("agency", "freelancer_network"),
        confidence=0.82,
        reason="sells translation",
        borderline=False,
    )

    status = record_screen_result(engine, entity_id=entity_id, run_id=run_id, result=result)

    assert status == STATUS_IN_SCOPE
    with Session(engine) as session:
        rows = list(session.scalars(sa.select(ScreenResultRow)))
        entity = session.get(Entity, entity_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.entity_id == entity_id
    assert row.run_id == run_id
    assert row.in_scope is True
    assert row.segments == ["agency", "freelancer_network"]
    assert row.confidence == 0.82
    assert row.reason == "sells translation"
    assert row.borderline is False
    assert row.created_at.tzinfo is not None
    assert row.created_at.utcoffset() == timedelta(0)
    assert entity is not None and entity.status == STATUS_IN_SCOPE


def test_out_of_scope_status(engine) -> None:
    entity_id = add_entity(engine)
    run_id = add_run(engine)
    result = ScreenResult(
        in_scope=False, segments=(), confidence=0.95, reason="off topic", borderline=False
    )
    assert (
        record_screen_result(engine, entity_id=entity_id, run_id=run_id, result=result)
        == STATUS_OUT_OF_SCOPE
    )
    with Session(engine) as session:
        assert session.get(Entity, entity_id).status == STATUS_OUT_OF_SCOPE


def test_missing_entity_rolls_back_the_result_row(engine) -> None:
    run_id = add_run(engine)
    result = ScreenResult(
        in_scope=True, segments=("agency",), confidence=0.9, reason="ok", borderline=False
    )
    with pytest.raises(ValueError, match="entity 999"):
        record_screen_result(engine, entity_id=999, run_id=run_id, result=result)
    with Session(engine) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(ScreenResultRow)) == 0


def test_load_entity_text_orders_snapshot_then_seq(engine) -> None:
    entity_id = add_entity(engine)
    add_chunks(
        engine,
        entity_id,
        [
            [(1, "two"), (0, "one")],  # written out of seq order on purpose
            [(0, "three")],
        ],
    )
    assert load_entity_text(engine, entity_id) == "one\n\ntwo\n\nthree"


def test_load_entity_text_caps_at_max_chars(engine) -> None:
    entity_id = add_entity(engine)
    add_chunks(engine, entity_id, [[(0, "x" * 50), (1, "MIDDLE"), (2, "y" * 50)]])
    text = load_entity_text(engine, entity_id, max_chars=60)
    assert len(text) == 60
    assert "MIDDLE" in text
    assert "y" * 50 not in text


def test_load_entity_text_empty_without_chunks(engine) -> None:
    entity_id = add_entity(engine)
    assert load_entity_text(engine, entity_id) == ""


def test_load_entity_text_ignores_other_entities(engine) -> None:
    other = add_entity(engine, "Other")
    entity_id = add_entity(engine)
    add_chunks(engine, other, [[(0, "other text")]])
    add_chunks(engine, entity_id, [[(0, "mine")]])
    assert load_entity_text(engine, entity_id) == "mine"


def test_default_max_chars_matches_screen_default() -> None:
    assert DEFAULT_MAX_CHARS == 8000


@pytest.mark.parametrize("bad", [0, -1])
def test_load_entity_text_rejects_non_positive_max_chars(engine, bad: int) -> None:
    entity_id = add_entity(engine)
    with pytest.raises(ValueError):
        load_entity_text(engine, entity_id, max_chars=bad)
