"""Tests for chunk row persistence (CH-03).

The suite runs offline against in-memory SQLite. The optional PostgreSQL test is
skipped unless ``DM_TEST_DATABASE_URL`` points at a disposable database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.parser import InMemoryParser, ParsedDocument
from core.chunk_store import ChunkRow, store_chunks
from core.chunking import Chunk
from core.parse_chunk import parse_and_chunk
from core.runs import create_all

POSTGRES_URL = os.environ.get("DM_TEST_DATABASE_URL")
URL = "https://example.test/doc"


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def make_chunk(
    seq: int,
    *,
    text: str | None = None,
    page: int | None = None,
    heading_path: tuple[str, ...] = (),
    char_start: int = 0,
    char_end: int | None = None,
) -> Chunk:
    body = text if text is not None else f"text {seq}"
    return Chunk(
        seq=seq,
        text=body,
        page=page,
        heading_path=heading_path,
        char_start=char_start,
        char_end=char_end if char_end is not None else char_start + len(body),
    )


def rows_for(engine: Engine, snapshot_id: int) -> list[ChunkRow]:
    with Session(engine) as session:
        return list(
            session.scalars(
                sa.select(ChunkRow)
                .where(ChunkRow.snapshot_id == snapshot_id)
                .order_by(ChunkRow.seq)
            )
        )


class StoreFixture:
    @pytest.fixture(autouse=True)
    def _engine(self) -> Iterator[Engine]:
        engine = memory_engine()
        create_all(engine)
        self.engine = engine
        yield engine
        engine.dispose()


class TestRoundTrip(StoreFixture):
    def test_rows_round_trip_all_fields(self) -> None:
        chunks = [
            make_chunk(
                0,
                text="first chunk",
                page=None,
                heading_path=("Alpha", "Beta"),
                char_start=3,
                char_end=14,
            ),
            make_chunk(1, text="second chunk", page=2, heading_path=(), char_start=20, char_end=32),
        ]

        written = store_chunks(self.engine, 7, chunks)

        assert written == 2
        rows = rows_for(self.engine, 7)
        assert [row.seq for row in rows] == [0, 1]
        assert rows[0].snapshot_id == 7
        assert rows[0].text == "first chunk"
        assert rows[0].page is None
        assert rows[0].lang is None  # language detection has not run yet
        assert rows[0].heading_path == ["Alpha", "Beta"]
        assert rows[0].char_start == 3
        assert rows[0].char_end == 14
        assert rows[1].page == 2
        assert rows[1].heading_path == []
        assert rows[1].char_start == 20
        assert rows[1].char_end == 32

    def test_create_all_creates_the_chunk_table(self) -> None:
        inspector = sa.inspect(self.engine)
        assert "chunk" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("chunk")}
        assert columns == {
            "id",
            "snapshot_id",
            "seq",
            "text",
            "lang",
            "page",
            "heading_path",
            "char_start",
            "char_end",
            "embedding",
            "embedding_model",
        }
        indexed_columns = {
            name for index in inspector.get_indexes("chunk") for name in index["column_names"]
        }
        assert "snapshot_id" in indexed_columns
        uniques = {
            tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints("chunk")
        }
        assert ("snapshot_id", "seq") in uniques


class TestIdempotency(StoreFixture):
    def test_restoring_a_snapshot_replaces_its_rows(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(0), make_chunk(1), make_chunk(2)])

        written = store_chunks(
            self.engine, 1, [make_chunk(0, text="new a"), make_chunk(1, text="new b")]
        )

        assert written == 2
        rows = rows_for(self.engine, 1)
        assert [row.text for row in rows] == ["new a", "new b"]

    def test_fewer_chunks_removes_extras(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(seq) for seq in range(4)])

        store_chunks(self.engine, 1, [make_chunk(0, text="only")])

        rows = rows_for(self.engine, 1)
        assert [(row.seq, row.text) for row in rows] == [(0, "only")]

    def test_two_snapshots_stay_independent(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(0, text="one")])
        store_chunks(self.engine, 2, [make_chunk(0, text="two"), make_chunk(1, text="two b")])

        store_chunks(self.engine, 1, [make_chunk(0, text="one again")])

        assert [row.text for row in rows_for(self.engine, 1)] == ["one again"]
        assert [row.text for row in rows_for(self.engine, 2)] == ["two", "two b"]

    def test_empty_iterable_clears_the_snapshot_and_returns_zero(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(0), make_chunk(1)])
        store_chunks(self.engine, 2, [make_chunk(0, text="other")])

        written = store_chunks(self.engine, 1, iter(()))

        assert written == 0
        assert rows_for(self.engine, 1) == []
        assert len(rows_for(self.engine, 2)) == 1

    def test_batch_size_below_one_raises_before_any_db_write(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(0, text="kept")])

        def untouched() -> Iterator[Chunk]:
            raise AssertionError("input must not be consumed")
            yield  # pragma: no cover

        with pytest.raises(ValueError):
            store_chunks(self.engine, 1, untouched(), batch_size=0)

        assert [row.text for row in rows_for(self.engine, 1)] == ["kept"]

    def test_exception_mid_iteration_rolls_back_to_the_previous_rows(self) -> None:
        store_chunks(self.engine, 1, [make_chunk(0, text="old a"), make_chunk(1, text="old b")])

        def broken() -> Iterator[Chunk]:
            yield make_chunk(0, text="new a")
            yield make_chunk(1, text="new b")
            yield make_chunk(2, text="new c")
            raise RuntimeError("source failed mid-stream")

        with pytest.raises(RuntimeError, match="source failed mid-stream"):
            store_chunks(self.engine, 1, broken(), batch_size=2)

        assert [row.text for row in rows_for(self.engine, 1)] == ["old a", "old b"]


class TestLaziness(StoreFixture):
    def test_generator_is_consumed_in_batches_not_materialised(self) -> None:
        chunks = [make_chunk(seq, text=f"chunk {seq}") for seq in range(5)]
        produced = 0
        batch_sizes_at_flush: list[int] = []

        def on_after_flush(session: Session, flush_context: object) -> None:
            batch_sizes_at_flush.append(produced)

        sa.event.listen(Session, "after_flush", on_after_flush)
        try:

            def source() -> Iterator[Chunk]:
                nonlocal produced
                for chunk in chunks:
                    produced += 1
                    yield chunk

            written = store_chunks(self.engine, 1, source(), batch_size=2)
        finally:
            sa.event.remove(Session, "after_flush", on_after_flush)

        assert written == 5
        # The first flush happened after two chunks, not after materialising all
        # five: the store batches inserts while it consumes the iterator.
        assert batch_sizes_at_flush == [2, 4, 5]
        assert [row.seq for row in rows_for(self.engine, 1)] == [0, 1, 2, 3, 4]


class TestEndToEnd(StoreFixture):
    def test_parse_and_chunk_rows_match(self) -> None:
        document = ParsedDocument(
            markdown="# Alpha\n\nintro text\n\n## Beta\n\nmore text\n",
            pages=(),
            title="Alpha",
            parser="test",
        )
        parser = InMemoryParser({"text/html": document})

        chunks = list(parse_and_chunk(b"body", content_type="text/html", url=URL, parser=parser))
        written = store_chunks(
            self.engine,
            42,
            parse_and_chunk(b"body", content_type="text/html", url=URL, parser=parser),
        )

        assert written == len(chunks)
        rows = rows_for(self.engine, 42)
        assert [
            (row.seq, row.text, row.page, row.heading_path, row.char_start, row.char_end)
            for row in rows
        ] == [
            (
                chunk.seq,
                chunk.text,
                chunk.page,
                list(chunk.heading_path),
                chunk.char_start,
                chunk.char_end,
            )
            for chunk in chunks
        ]


@pytest.mark.skipif(
    POSTGRES_URL is None, reason="DM_TEST_DATABASE_URL is not set; PostgreSQL path not exercised"
)
class TestPostgres:
    def test_store_replaces_rows_on_postgres(self) -> None:
        assert POSTGRES_URL is not None
        engine = create_engine(POSTGRES_URL)
        try:
            with engine.begin() as connection:
                ChunkRow.__table__.drop(connection, checkfirst=True)
            create_all(engine)
            try:
                store_chunks(engine, 1, [make_chunk(0), make_chunk(1)])
                written = store_chunks(engine, 1, [make_chunk(0, text="replaced")])
                assert written == 1
                assert [row.text for row in rows_for(engine, 1)] == ["replaced"]
            finally:
                with engine.begin() as connection:
                    ChunkRow.__table__.drop(connection, checkfirst=True)
        finally:
            engine.dispose()
