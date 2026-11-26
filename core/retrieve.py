"""Keyword-only evidence retrieval for one entity × criterion (AS-02).

Query rule (:func:`build_query`): the criterion's ``retrieval_hints`` in their
declared order, followed by its ``question``, joined with single spaces. BM25
tokenisation then treats repeated terms once (see :mod:`core.keyword_search`).

:func:`retrieve_evidence` runs :func:`core.keyword_search.search_chunks` for the
entity with ``limit=k``, keeps the first ``n`` hits in hit (score) order and loads
their texts in one query, returning the ``(chunk_id, text)`` pairs that
:func:`core.assess.extract_criterion` takes.

Seam: ARCHITECTURE.md §4 stage 7 targets hybrid retrieval (BM25 on hints + vector
on question) → top-k → reranker → top-n. This module is the keyword leg only; the
vector leg and reranker are a follow-up item and slot in between the search and the
top-``n`` cut. ``k`` is kept for that seam (the reranker's candidate pool).

Unwired by design: the production caller is the later ``landscape assess`` CLI row.
Scale: inherits the O(entity corpus) scan of ``search_chunks`` (IX-03 residual); the
text load is one query for at most ``n`` ids.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.keyword_search import search_chunks
from core.scope_pack import Criterion


def build_query(criterion: Criterion) -> str:
    """Return ``retrieval_hints`` then ``question``, joined with single spaces."""
    return " ".join([*criterion.retrieval_hints, criterion.question])


def retrieve_evidence(
    engine: Engine, criterion: Criterion, *, entity_id: int, n: int = 6, k: int = 24
) -> list[tuple[int, str]]:
    """Return up to ``n`` ``(chunk_id, text)`` pairs for ``entity_id``, best score first."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if k < n:
        raise ValueError("k must be at least n")
    hits = search_chunks(engine, build_query(criterion), limit=k, entity_id=entity_id)[:n]
    if not hits:
        return []
    ids = [h.chunk_id for h in hits]
    with Session(engine) as session:
        rows = session.execute(sa.select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(ids)))
        texts = {row.id: row.text for row in rows}
    return [(cid, texts[cid]) for cid in ids]
