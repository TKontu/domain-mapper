"""Tests for portable BM25 keyword search over stored chunks (IX-03). Offline, SQLite."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from core.chunk_store import store_chunks
from core.chunking import Chunk
from core.keyword_search import KeywordHit, search_chunks, tokenize
from core.runs import create_all
from core.snapshot_store import SnapshotRow


def make_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'ks.db'}")
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
        Chunk(
            seq=i, text=t, page=None, heading_path=("Intro", f"S{i}"), char_start=0, char_end=len(t)
        )
        for i, t in enumerate(texts)
    ]
    store_chunks(engine, snapshot_id, chunks)


def test_tokenize_casefold_non_ascii_digits() -> None:
    assert tokenize("Straße MÜNCHEN, item-42 ok!") == ["strasse", "münchen", "item", "42", "ok"]
    assert tokenize("   ...  ") == []


def test_hand_computed_bm25_three_docs(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["apple banana", "apple apple cherry", "cherry date"])
    hits = search_chunks(engine, "Apple apple")  # repeated term counts once

    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 7 / 3

    def term(tf: int, dl: int) -> float:
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    assert [h.seq for h in hits] == [1, 0]
    assert hits[0].score == pytest.approx(term(2, 3), abs=1e-9)
    assert hits[1].score == pytest.approx(term(1, 2), abs=1e-9)
    assert hits[0].snapshot_id == snap
    assert hits[0].heading_path == ("Intro", "S1")
    assert isinstance(hits[0], KeywordHit)


def test_entity_filter_restricts_corpus_and_stats(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    a = add_snapshot(engine, 1)
    b = add_snapshot(engine, 2)
    add_chunks(engine, a, ["widget alpha", "gamma beta"])
    add_chunks(engine, b, ["widget widget", "widget omega"])
    hits = search_chunks(engine, "widget", entity_id=1)
    assert [h.snapshot_id for h in hits] == [a]
    # corpus of entity 1: N=2, df=1, avgdl=2
    idf = math.log(1 + (2 - 1 + 0.5) / 1.5)
    assert hits[0].score == pytest.approx(idf * 2.2 / (1 + 1.2), abs=1e-9)
    assert search_chunks(engine, "widget", entity_id=99) == []


def test_tie_order_by_chunk_id_and_limit(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["same words", "other thing", "same words", "same words"])
    hits = search_chunks(engine, "same")
    ids = [h.chunk_id for h in hits]
    assert ids == sorted(ids) and len(ids) == 3
    assert [h.chunk_id for h in search_chunks(engine, "same", limit=2)] == ids[:2]


def test_empty_queries_and_corpus(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert search_chunks(engine, "anything") == []
    snap = add_snapshot(engine, None)
    add_chunks(engine, snap, ["neutral text"])
    assert search_chunks(engine, "") == []
    assert search_chunks(engine, "   ") == []
    assert search_chunks(engine, "!!! ---") == []
    assert search_chunks(engine, "missing") == []


def test_invalid_arguments(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError):
        search_chunks(engine, "x", limit=0)
    with pytest.raises(ValueError):
        search_chunks(engine, "x", batch_size=0)


def test_paging_batch_sizes_agree(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    a = add_snapshot(engine, 1)
    b = add_snapshot(engine, 2)
    texts = [f"term{i % 3} common filler {'extra ' * (i % 4)}" for i in range(11)]
    add_chunks(engine, a, texts)
    add_chunks(engine, b, texts[:5])
    for entity in (None, 1):
        results = [
            search_chunks(engine, "common term1 extra", limit=4, entity_id=entity, batch_size=n)
            for n in (1, 2, 500)
        ]
        assert results[0] == results[1] == results[2]
        assert len(results[0]) == 4
