"""Portable Okapi BM25 keyword search over stored ``chunk`` rows (IX-03).

* :func:`tokenize` casefolds and splits into maximal runs of ``str.isalnum``
  characters; no stemming, no stopwords.
* :func:`search_chunks` scores with BM25 (k1=1.2, b=0.75) over the corpus of all
  chunks, or only chunks whose snapshot has ``entity_id``. IDF is
  ``ln(1 + (N - df + 0.5) / (df + 0.5))``; repeated query terms count once.
* Two keyset passes over ``chunk.id`` ascending, ``batch_size`` rows per page:
  pass 1 gathers N, total length and df of query terms; pass 2 scores and keeps
  a heap of size ``limit``. Memory is one batch plus the heap; dialect-independent.
* Residual: full scan per query (O(corpus)); a Postgres full-text index is later work.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.snapshot_store import SnapshotRow

K1: Final = 1.2
B: Final = 0.75
DEFAULT_BATCH_SIZE: Final = 500


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: int
    snapshot_id: int
    seq: int
    score: float
    heading_path: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Casefold ``text`` and return maximal runs of alphanumeric characters."""
    tokens: list[str] = []
    current: list[str] = []
    for ch in text.casefold():
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def search_chunks(
    engine: Engine,
    query: str,
    *,
    limit: int = 10,
    entity_id: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[KeywordHit]:
    """Return up to ``limit`` chunks with BM25 score > 0, score desc then chunk_id asc."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    terms = set(tokenize(query))
    if not terms:
        return []

    n_docs = 0
    total_len = 0
    df: Counter[str] = Counter()
    with Session(engine) as session:
        for page in _pages(session, entity_id, batch_size):
            for row in page:
                tokens = tokenize(row.text)
                n_docs += 1
                total_len += len(tokens)
                df.update(terms.intersection(tokens))
        if n_docs == 0 or not df:
            return []
        avgdl = total_len / n_docs
        idf = {t: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        heap: list[tuple[float, int, KeywordHit]] = []
        for page in _pages(session, entity_id, batch_size):
            for row in page:
                tokens = tokenize(row.text)
                score = _score(tokens, idf, avgdl)
                if score <= 0:
                    continue
                hit = KeywordHit(
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
    return [h for _, _, h in sorted(heap, key=lambda i: (-i[0], -i[1]))]


def _score(tokens: Sequence[str], idf: dict[str, float], avgdl: float) -> float:
    if not tokens:
        return 0.0
    tf = Counter(t for t in tokens if t in idf)
    norm = K1 * (1 - B + B * len(tokens) / avgdl)
    return sum(idf[t] * f * (K1 + 1) / (f + norm) for t, f in tf.items())


def _pages(session: Session, entity_id: int | None, batch_size: int) -> Iterator[Sequence[Any]]:
    stmt = sa.select(
        ChunkRow.id, ChunkRow.snapshot_id, ChunkRow.seq, ChunkRow.text, ChunkRow.heading_path
    )
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
