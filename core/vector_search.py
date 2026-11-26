"""Cosine vector search over stored ``chunk`` embeddings (AS-06).

ARCHITECTURE.md §4 stage 7 retrieves evidence for a criterion as hybrid BM25 on
``retrieval_hints`` plus a vector leg on the question. This module is the vector
leg: :func:`search_chunks_by_vector` scores every stored chunk embedding for the
requested ``model`` against a query vector and returns the top ``limit`` by
cosine similarity.

Semantics:

* Only chunks whose ``embedding_model == model`` **and** whose ``embedding`` is
  non-null are considered; other models and unembedded chunks are skipped.
* Keyset paging by ``chunk.id`` ascending, ``batch_size`` rows per page; memory
  is one page plus the top-``limit`` heap. Optional ``entity_id`` restricts the
  corpus through the chunk's snapshot, as in :mod:`core.keyword_search`.
* Scores are cosine similarity; a stored zero-norm vector scores ``0.0``.
* A query vector that is empty, non-numeric, non-finite or zero-norm raises
  :class:`ValueError`; so does any stored vector whose dimension differs from the
  query, naming the chunk id (never its text). ``limit`` or ``batch_size`` below
  1 and an empty ``model`` raise before scanning.
* Ties break by ``chunk_id`` ascending; ``limit`` below the corpus size keeps the
  lowest ids among equal scores.

Caller: :func:`core.retrieve.retrieve_evidence` runs this as the vector leg of
hybrid retrieval. That module's ``embedder`` argument is unwired by design until
the later ``landscape assess`` row passes an embedder. pgvector and a vector
index are later work: this is an O(entity corpus) scan in keyset batches.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.snapshot_store import SnapshotRow

DEFAULT_BATCH_SIZE: Final = 500


@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    snapshot_id: int
    seq: int
    score: float
    heading_path: tuple[str, ...]


def search_chunks_by_vector(
    engine: Engine,
    query_vector: Sequence[float],
    *,
    model: str,
    limit: int,
    entity_id: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[VectorHit]:
    """Return up to ``limit`` chunks by cosine score, score desc then chunk_id asc."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not model:
        raise ValueError("model must be non-empty")
    query = _query_vector(query_vector)
    query_norm = math.sqrt(math.fsum(value * value for value in query))

    heap: list[tuple[float, int, VectorHit]] = []
    with Session(engine) as session:
        for page in _pages(session, entity_id, model, batch_size):
            for row in page:
                # The JSON column stores Python None as JSON ``null``, which is
                # not SQL NULL on SQLite, so unembedded rows are skipped here.
                if row.embedding is None:
                    continue
                vector = _stored_vector(row.embedding, row.id)
                if len(vector) != len(query):
                    raise ValueError(
                        f"embedding dimension mismatch for chunk id {row.id}: "
                        f"stored {len(vector)}, query {len(query)}"
                    )
                score = _cosine(query, query_norm, vector)
                hit = VectorHit(
                    chunk_id=row.id,
                    snapshot_id=row.snapshot_id,
                    seq=row.seq,
                    score=score,
                    heading_path=tuple(row.heading_path or ()),
                )
                # Min-heap on (score, -id): the root is the worst hit kept.
                item = (score, -row.id, hit)
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
    return [hit for _, _, hit in sorted(heap, key=lambda item: (-item[0], -item[1]))]


def _query_vector(query_vector: Sequence[float]) -> list[float]:
    if isinstance(query_vector, (str, bytes)) or not isinstance(query_vector, Sequence):
        raise ValueError("query_vector must be a non-empty sequence of finite numbers")
    values: list[float] = []
    for value in query_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("query_vector must contain only finite numbers")
        if not math.isfinite(value):
            raise ValueError("query_vector must contain only finite numbers")
        values.append(float(value))
    if not values:
        raise ValueError("query_vector must be non-empty")
    if math.fsum(value * value for value in values) == 0.0:
        raise ValueError("query_vector must have a non-zero norm")
    return values


def _stored_vector(value: Any, chunk_id: int) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"chunk id {chunk_id} has no usable embedding vector")
    vector: list[float] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise ValueError(f"chunk id {chunk_id} has a malformed embedding vector")
        if not math.isfinite(element):
            raise ValueError(f"chunk id {chunk_id} has a malformed embedding vector")
        vector.append(float(element))
    return vector


def _cosine(query: Sequence[float], query_norm: float, vector: Sequence[float]) -> float:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        return 0.0
    dot = math.fsum(left * right for left, right in zip(query, vector, strict=True))
    return dot / (query_norm * norm)


def _pages(
    session: Session, entity_id: int | None, model: str, batch_size: int
) -> Iterator[Sequence[Any]]:
    stmt = sa.select(
        ChunkRow.id,
        ChunkRow.snapshot_id,
        ChunkRow.seq,
        ChunkRow.heading_path,
        ChunkRow.embedding,
    ).where(ChunkRow.embedding_model == model)
    if entity_id is not None:
        stmt = stmt.join(SnapshotRow, SnapshotRow.id == ChunkRow.snapshot_id).where(
            SnapshotRow.entity_id == entity_id
        )
    last_id = 0
    while True:
        page = session.execute(
            stmt.where(ChunkRow.id > last_id).order_by(ChunkRow.id).limit(batch_size)
        ).fetchall()
        if not page:
            return
        yield page
        last_id = page[-1].id
