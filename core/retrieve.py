"""Evidence retrieval for one entity × criterion (AS-02, AS-06).

Query rule (:func:`build_query`): the criterion's ``retrieval_hints`` in their
declared order, followed by its ``question``, joined with single spaces. BM25
tokenisation then treats repeated terms once (see :mod:`core.keyword_search`).

:func:`retrieve_evidence` has two paths:

* ``embedder=None`` (default): keyword-only. It runs
  :func:`core.keyword_search.search_chunks` for the entity with ``limit=k``,
  keeps the first ``n`` hits in hit (score) order and loads their texts in one
  query, returning the ``(chunk_id, text)`` pairs that
  :func:`core.assess.extract_criterion` takes. This path is unchanged from AS-02.
* ``embedder`` given: hybrid. It embeds the criterion ``question`` once, runs the
  keyword leg on :func:`build_query` and the vector leg
  (:func:`core.vector_search.search_chunks_by_vector`, stored embeddings for
  ``model``) with ``limit=k`` each, fuses them by reciprocal-rank fusion
  (``k_rrf=60``, ties by ``chunk_id`` ascending) and cuts to ``n``.

ARCHITECTURE.md §4 stage 7 targets hybrid retrieval (BM25 on hints + vector on
question) → top-k → reranker → top-n. The reranker remains a follow-up item and
slots in between the fused top-``k`` and the top-``n`` cut; ``k`` is its
candidate pool.

Unwired by design: the production caller is the later ``landscape assess`` CLI row,
which passes an :class:`~adapters.embedding.Embedder`; until then only tests call
the hybrid path. Scale: inherits the O(entity corpus) scan of ``search_chunks`` and
the vector leg (IX-03/AS-06 residuals); the text load is one query for at most
``2k`` fused ids, cut to ``n``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from adapters.embedding import Embedder
from core.chunk_store import ChunkRow
from core.keyword_search import KeywordHit, search_chunks
from core.scope_pack import Criterion
from core.vector_search import VectorHit, search_chunks_by_vector

RRF_K: Final = 60


def build_query(criterion: Criterion) -> str:
    """Return ``retrieval_hints`` then ``question``, joined with single spaces."""
    return " ".join([*criterion.retrieval_hints, criterion.question])


def retrieve_evidence(
    engine: Engine,
    criterion: Criterion,
    *,
    entity_id: int,
    n: int = 6,
    k: int = 24,
    embedder: Embedder | None = None,
    model: str | None = None,
) -> list[tuple[int, str]]:
    """Return up to ``n`` ``(chunk_id, text)`` pairs for ``entity_id``, best first.

    With ``embedder`` given, ``model`` selects the stored embedding model
    (default: ``embedder.model``) and both legs are fused by reciprocal-rank
    fusion. Without it, the keyword-only AS-02 path runs and ``model`` is unused.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if k < n:
        raise ValueError("k must be at least n")
    if embedder is None:
        hits = search_chunks(engine, build_query(criterion), limit=k, entity_id=entity_id)[:n]
        return _with_texts(engine, [hit.chunk_id for hit in hits])

    resolved_model = model if model is not None else embedder.model
    if not resolved_model:
        raise ValueError("model must be non-empty")
    vectors = embedder.embed([criterion.question])
    if len(vectors) != 1:
        raise ValueError("embedder must return exactly one vector for the question")
    keyword_hits = search_chunks(engine, build_query(criterion), limit=k, entity_id=entity_id)
    vector_hits = search_chunks_by_vector(
        engine, vectors[0], model=resolved_model, limit=k, entity_id=entity_id
    )
    return _with_texts(engine, _fuse(keyword_hits, vector_hits, n=n))


def _fuse(keyword: Sequence[KeywordHit], vector: Sequence[VectorHit], *, n: int) -> list[int]:
    """Reciprocal-rank fusion; higher score first, ties by chunk id ascending."""
    scores: dict[int, float] = {}
    for rank, keyword_hit in enumerate(keyword, start=1):
        scores[keyword_hit.chunk_id] = scores.get(keyword_hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, vector_hit in enumerate(vector, start=1):
        scores[vector_hit.chunk_id] = scores.get(vector_hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in ranked[:n]]


def _with_texts(engine: Engine, ids: Sequence[int]) -> list[tuple[int, str]]:
    if not ids:
        return []
    with Session(engine) as session:
        rows = session.execute(sa.select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(ids)))
        texts = {row.id: row.text for row in rows}
    return [(chunk_id, texts[chunk_id]) for chunk_id in ids]
