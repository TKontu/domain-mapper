"""Tests for cosine vector search over stored chunk embeddings (AS-06). Offline, SQLite."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow, store_chunks
from core.chunking import Chunk
from core.runs import create_all
from core.snapshot_store import SnapshotRow
from core.vector_search import VectorHit, search_chunks_by_vector


def make_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'vs.db'}")
    create_all(engine)
    return engine


def add_snapshot(engine: Engine, entity_id: int | None) -> int:
    row = SnapshotRow(
        entity_id=entity_id,
        url="https://example.test/page",
        final_url="https://example.test/page",
        content_type="text/html",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_kind="web",
    )
    with Session(engine) as session, session.begin():
        session.add(row)
        session.flush()
        return row.id


def add_chunks(engine: Engine, snapshot_id: int, texts: list[str]) -> None:
    chunks = [
        Chunk(seq=i, text=t, page=None, heading_path=("S", f"T{i}"), char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]
    store_chunks(engine, snapshot_id, chunks)


def ids_by_text(engine: Engine) -> dict[str, int]:
    with Session(engine) as session:
        return {r.text: r.id for r in session.scalars(sa.select(ChunkRow))}


def set_embedding(
    engine: Engine,
    chunk_id: int,
    vector: list[float] | None,
    model: str | None,
) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            sa.update(ChunkRow)
            .where(ChunkRow.id == chunk_id)
            .values(embedding=vector, embedding_model=model)
        )


def test_ranks_by_cosine_and_caps_limit(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    texts = ["same direction", "diagonal", "orthogonal", "opposite"]
    add_chunks(engine, snap, texts)
    ids = ids_by_text(engine)
    set_embedding(engine, ids["same direction"], [1.0, 0.0, 0.0], "m1")
    set_embedding(engine, ids["diagonal"], [1.0, 1.0, 0.0], "m1")
    set_embedding(engine, ids["orthogonal"], [0.0, 1.0, 0.0], "m1")
    set_embedding(engine, ids["opposite"], [-1.0, 0.0, 0.0], "m1")

    hits = search_chunks_by_vector(engine, [1.0, 0.0, 0.0], model="m1", limit=3)

    assert [h.chunk_id for h in hits] == [
        ids["same direction"],
        ids["diagonal"],
        ids["orthogonal"],
    ]
    assert [h.score for h in hits] == pytest.approx([1.0, math.sqrt(0.5), 0.0], abs=1e-9)
    assert isinstance(hits[0], VectorHit)
    assert hits[0].snapshot_id == snap
    assert hits[0].heading_path == ("S", "T0")
    assert len(search_chunks_by_vector(engine, [1.0, 0.0, 0.0], model="m1", limit=10)) == 4


def test_model_entity_and_null_embedding_filters(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    own = add_snapshot(engine, 1)
    other = add_snapshot(engine, 2)
    add_chunks(engine, own, ["wanted model", "other model", "null vector", "null model"])
    add_chunks(engine, other, ["other entity"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["wanted model"], [1.0, 0.0], "m1")
    set_embedding(engine, ids["other model"], [1.0, 0.0], "m2")
    set_embedding(engine, ids["null vector"], None, "m1")
    set_embedding(engine, ids["null model"], [1.0, 0.0], None)
    set_embedding(engine, ids["other entity"], [1.0, 0.0], "m1")

    query = [1.0, 0.0]
    assert [
        h.chunk_id
        for h in search_chunks_by_vector(engine, query, model="m1", limit=10, entity_id=1)
    ] == [ids["wanted model"]]
    assert {h.chunk_id for h in search_chunks_by_vector(engine, query, model="m1", limit=10)} == {
        ids["wanted model"],
        ids["other entity"],
    }
    assert search_chunks_by_vector(engine, query, model="m1", limit=10, entity_id=99) == []
    assert search_chunks_by_vector(engine, query, model="missing", limit=10) == []


def test_ties_break_by_chunk_id_and_limit_keeps_lowest_ids(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["first tie", "second tie", "third tie"])
    ids = ids_by_text(engine)
    for text in ("first tie", "second tie", "third tie"):
        set_embedding(engine, ids[text], [1.0, 0.0], "m1")

    hits = search_chunks_by_vector(engine, [1.0, 0.0], model="m1", limit=2)
    assert [h.chunk_id for h in hits] == [ids["first tie"], ids["second tie"]]
    assert hits[0].score == hits[1].score == pytest.approx(1.0)


def test_dimension_mismatch_raises_value_error_naming_chunk(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["secret evidence phrase", "secret evidence phrase two"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["secret evidence phrase"], [1.0, 0.0], "m1")
    set_embedding(engine, ids["secret evidence phrase two"], [1.0, 0.0, 0.0], "m1")

    with pytest.raises(ValueError) as info:
        search_chunks_by_vector(engine, [1.0, 0.0], model="m1", limit=2)
    assert str(ids["secret evidence phrase two"]) in str(info.value)
    assert "secret evidence" not in str(info.value)


def test_batch_sizes_page_across_the_boundary(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, [f"neutral text {i}" for i in range(11)])
    ids = ids_by_text(engine)
    for i in range(11):
        set_embedding(engine, ids[f"neutral text {i}"], [1.0, i / 10, 0.0], "m1")

    query = [1.0, 0.0, 0.0]
    results = [
        search_chunks_by_vector(engine, query, model="m1", limit=4, batch_size=size)
        for size in (1, 2, 5, 500)
    ]
    assert results[0] == results[1] == results[2] == results[3]
    assert [h.chunk_id for h in results[0]] == [ids[f"neutral text {i}"] for i in range(4)]


def test_empty_corpus_and_unembedded_rows(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert search_chunks_by_vector(engine, [1.0], model="m1", limit=1) == []
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["never embedded"])
    assert search_chunks_by_vector(engine, [1.0], model="m1", limit=1) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "m1", "limit": 0},
        {"model": "m1", "limit": 1, "batch_size": 0},
        {"model": "", "limit": 1},
    ],
)
def test_invalid_arguments(tmp_path: Path, kwargs: dict[str, object]) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError):
        search_chunks_by_vector(engine, [1.0], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    [
        [],
        [math.nan],
        [math.inf],
        [True],
        ["x"],
        [0.0, 0.0],
    ],
)
def test_invalid_query_vectors(tmp_path: Path, query: list[object]) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError):
        search_chunks_by_vector(
            engine,
            query,  # type: ignore[arg-type]
            model="m1",
            limit=1,
        )
