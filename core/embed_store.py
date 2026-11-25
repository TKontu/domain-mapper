"""Fill ``chunk.embedding`` for pending chunks (IX-01, ARCHITECTURE.md §6 Index).

:func:`embed_pending_chunks` embeds every chunk whose ``embedding_model`` is NULL
or differs from ``model``. It is **unwired by design** until IX-02 (CLI
``landscape index --scope``).

Semantics:

* Keyset paging by ``chunk.id`` ascending, ``batch_size`` rows per page; one
  ``embed`` call and one transaction per batch. Memory is bounded to one batch
  of texts and vectors.
* ``limit`` caps the total chunks embedded (None = all).
* The embedder must return one vector per text, and every vector in a run must
  share one non-zero dimension of finite floats; otherwise
  :class:`EmbedStoreError` names the batch's chunk ids (never text). A failing
  batch — including an ``EmbeddingError`` from the embedder — writes nothing;
  earlier batches stay committed.
* Vectors are stored as a JSON list of floats; the pgvector type and migrations
  are later items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from adapters.embedding import Embedder
from core.chunk_store import ChunkRow

DEFAULT_BATCH_SIZE: Final = 64


class EmbedStoreError(RuntimeError):
    """The embedder's output cannot be stored; names chunk ids, never text."""


@dataclass(frozen=True)
class EmbedStats:
    embedded: int
    batches: int


def embed_pending_chunks(
    engine: Engine,
    embedder: Embedder,
    *,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> EmbedStats:
    """Embed chunks lacking a ``model`` embedding; return counts."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not model:
        raise ValueError("model must be non-empty")

    embedded = 0
    batches = 0
    last_id = 0
    dim: int | None = None
    pending = sa.or_(ChunkRow.embedding_model.is_(None), ChunkRow.embedding_model != model)
    while limit is None or embedded < limit:
        size = batch_size if limit is None else min(batch_size, limit - embedded)
        with Session(engine) as session, session.begin():
            page = session.execute(
                sa.select(ChunkRow.id, ChunkRow.text)
                .where(pending, ChunkRow.id > last_id)
                .order_by(ChunkRow.id)
                .limit(size)
            ).all()
            if not page:
                break
            ids = [row.id for row in page]
            vectors = embedder.embed([row.text for row in page])
            dim = _validate(vectors, ids, dim)
            for chunk_id, vector in zip(ids, vectors, strict=True):
                session.execute(
                    sa.update(ChunkRow)
                    .where(ChunkRow.id == chunk_id)
                    .values(embedding=[float(x) for x in vector], embedding_model=model)
                )
        embedded += len(ids)
        batches += 1
        last_id = ids[-1]
    return EmbedStats(embedded=embedded, batches=batches)


def _validate(vectors: object, ids: list[int], dim: int | None) -> int:
    span = f"chunk ids {ids[0]}..{ids[-1]} ({len(ids)} chunks)"
    if not isinstance(vectors, list) or len(vectors) != len(ids):
        got = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
        raise EmbedStoreError(f"embedder returned {got} vectors for {span}")
    for chunk_id, vector in zip(ids, vectors, strict=True):
        if not isinstance(vector, list) or not vector:
            raise EmbedStoreError(f"empty or non-list vector for chunk id {chunk_id} in {span}")
        if dim is None:
            dim = len(vector)
        if len(vector) != dim:
            raise EmbedStoreError(
                f"vector for chunk id {chunk_id} has dimension {len(vector)}, expected {dim}"
            )
        for x in vector:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise EmbedStoreError(f"non-finite or non-numeric value for chunk id {chunk_id}")
    assert dim is not None
    return dim
