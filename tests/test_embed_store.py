"""Tests for chunk embedding persistence (IX-01). Offline: fake embedder, SQLite."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import sqlalchemy as sa
from core.embed_store import EmbedStats, EmbedStoreError, embed_pending_chunks
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.embedding import EmbeddingError
from core.chunk_store import ChunkRow, store_chunks
from core.chunking import Chunk
from core.runs import create_all


class FakeEmbedder:
    def __init__(self, model: str = "fake-model", dim: int = 3) -> None:
        self.model = model
        self.dim = dim
        self.calls: list[list[str]] = []
        self.override: dict[int, list[list[float]] | Exception] = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        call = len(self.calls)
        result = self.override.get(call)
        if isinstance(result, Exception):
            raise result
        if result is not None:
            return result
        return [[float(len(t)), float(call), 0.5][: self.dim] for t in texts]


def engine_with_chunks(n: int) -> Engine:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    create_all(engine)
    chunks = [
        Chunk(seq=i, text=f"neutral text {i}", page=None, heading_path=(), char_start=0, char_end=5)
        for i in range(n)
    ]
    store_chunks(engine, 1, chunks)
    return engine


def rows(engine: Engine) -> list[ChunkRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(ChunkRow).order_by(ChunkRow.id)))


def test_embeds_all_in_id_order_batches() -> None:
    engine = engine_with_chunks(5)
    fake = FakeEmbedder()
    stats = embed_pending_chunks(engine, fake, model="m1", batch_size=2)
    assert stats == EmbedStats(embedded=5, batches=3)
    assert [len(c) for c in fake.calls] == [2, 2, 1]
    assert fake.calls[0] == ["neutral text 0", "neutral text 1"]
    for row in rows(engine):
        assert row.embedding_model == "m1"
        assert row.embedding is not None and len(row.embedding) == 3


def test_rerun_same_model_embeds_zero_and_new_model_reembeds() -> None:
    engine = engine_with_chunks(3)
    embed_pending_chunks(engine, FakeEmbedder(), model="m1")
    fake = FakeEmbedder()
    assert embed_pending_chunks(engine, fake, model="m1") == EmbedStats(0, 0)
    assert fake.calls == []
    assert embed_pending_chunks(engine, fake, model="m2") == EmbedStats(3, 1)
    assert {r.embedding_model for r in rows(engine)} == {"m2"}


def test_rechunking_clears_embeddings() -> None:
    engine = engine_with_chunks(2)
    embed_pending_chunks(engine, FakeEmbedder(), model="m1")
    store_chunks(
        engine, 1, [Chunk(seq=0, text="x", page=None, heading_path=(), char_start=0, char_end=1)]
    )
    (row,) = rows(engine)
    assert row.embedding is None and row.embedding_model is None


def test_limit_caps_total() -> None:
    engine = engine_with_chunks(5)
    fake = FakeEmbedder()
    assert embed_pending_chunks(engine, fake, model="m", batch_size=2, limit=3) == EmbedStats(3, 2)
    assert [len(c) for c in fake.calls] == [2, 1]
    assert embed_pending_chunks(engine, fake, model="m", limit=0) == EmbedStats(0, 0)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"limit": -1}, {"model": ""}])
def test_bad_arguments_raise_before_embedding(kwargs: dict[str, object]) -> None:
    engine = engine_with_chunks(1)
    fake = FakeEmbedder()
    args: dict[str, object] = {"model": "m", **kwargs}
    with pytest.raises(ValueError):
        embed_pending_chunks(engine, fake, **args)  # type: ignore[arg-type]
    assert fake.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 2.0, 3.0]],  # wrong count
        [[1.0, 2.0, 3.0], [1.0, 2.0]],  # dimension mismatch within batch
        [[1.0, 2.0], [1.0, 2.0]],  # dimension differs from earlier batch
        [[], []],  # empty
        [[1.0, math.nan, 3.0], [1.0, 2.0, 3.0]],  # non-finite
        [[1.0, "x", 3.0], [1.0, 2.0, 3.0]],  # non-float
    ],
)
def test_bad_vectors_raise_and_keep_earlier_batches(bad: list[list[float]]) -> None:
    engine = engine_with_chunks(4)
    fake = FakeEmbedder()
    fake.override[2] = bad
    with pytest.raises(EmbedStoreError) as info:
        embed_pending_chunks(engine, fake, model="m", batch_size=2)
    ids = [r.id for r in rows(engine)]
    assert str(ids[2]) in str(info.value)
    assert "neutral text" not in str(info.value)
    state = [r.embedding_model for r in rows(engine)]
    assert state == ["m", "m", None, None]


def test_embedding_error_propagates_and_batch_writes_nothing() -> None:
    engine = engine_with_chunks(4)
    fake = FakeEmbedder()
    fake.override[2] = EmbeddingError("service down", status=503)
    with pytest.raises(EmbeddingError):
        embed_pending_chunks(engine, fake, model="m", batch_size=2)
    assert [r.embedding_model for r in rows(engine)] == ["m", "m", None, None]
