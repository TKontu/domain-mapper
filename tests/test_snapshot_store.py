"""Tests for snapshot row persistence (CR-01).

The suite runs offline against in-memory SQLite. The optional PostgreSQL test is
skipped unless ``DM_TEST_DATABASE_URL`` points at a disposable database.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.fetcher import FetchResult
from core.chunk_store import ChunkRow, store_chunks
from core.chunking import Chunk
from core.runs import create_all
from core.snapshot_store import SnapshotRow, record_snapshot

POSTGRES_URL = os.environ.get("DM_TEST_DATABASE_URL")
URL = "https://example.test/page"
FINAL_URL = "https://example.test/page/final"
BODY = b"<html><body>snapshot body</body></html>"
FETCHED_AT = datetime(2026, 11, 25, 12, 30, 45, tzinfo=UTC)


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def make_result(
    *,
    body: bytes = BODY,
    final_url: str = FINAL_URL,
    fetched_at: datetime = FETCHED_AT,
    content_type: str = "text/html",
) -> FetchResult:
    return FetchResult(
        final_url=final_url,
        status=200,
        content_type=content_type,
        body_bytes=body,
        headers={"content-language": "en"},
        fetched_at=fetched_at,
    )


def rows(engine: Engine) -> list[SnapshotRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(SnapshotRow).order_by(SnapshotRow.id)))


def row(engine: Engine, snapshot_id: int) -> SnapshotRow:
    with Session(engine) as session:
        stored = session.get(SnapshotRow, snapshot_id)
    assert stored is not None
    return stored


class StoreFixture:
    @pytest.fixture(autouse=True)
    def _engine(self) -> Iterator[Engine]:
        engine = memory_engine()
        create_all(engine)
        self.engine = engine
        yield engine
        engine.dispose()


class TestRoundTrip(StoreFixture):
    def test_row_round_trips_all_columns(self) -> None:
        result = make_result()

        snapshot_id = record_snapshot(
            self.engine, entity_id=7, url=URL, result=result, source_kind="homepage"
        )

        stored = row(self.engine, snapshot_id)
        assert stored.id == snapshot_id
        assert stored.entity_id == 7
        assert stored.url == URL
        assert stored.final_url == FINAL_URL
        assert stored.content_type == "text/html"
        assert stored.source_kind == "homepage"
        assert stored.fetched_at == FETCHED_AT
        assert stored.fetched_at.tzinfo is UTC

    def test_sha256_is_the_lowercase_hex_of_the_body_bytes(self) -> None:
        expected = hashlib.sha256(BODY).hexdigest()

        snapshot_id = record_snapshot(
            self.engine, entity_id=None, url=URL, result=make_result(), source_kind="page"
        )

        stored = row(self.engine, snapshot_id)
        assert stored.sha256 == expected
        assert len(stored.sha256) == 64
        assert stored.sha256 == stored.sha256.lower()

    def test_nullable_entity_id_is_stored_as_null(self) -> None:
        snapshot_id = record_snapshot(
            self.engine, entity_id=None, url=URL, result=make_result(), source_kind="unresolved"
        )

        assert row(self.engine, snapshot_id).entity_id is None

    def test_non_utc_fetched_at_is_normalised_to_aware_utc_on_sqlite(self) -> None:
        local = datetime(2026, 11, 25, 14, 30, 45, tzinfo=timezone(timedelta(hours=2)))

        snapshot_id = record_snapshot(
            self.engine,
            entity_id=None,
            url=URL,
            result=make_result(fetched_at=local),
            source_kind="page",
        )

        stored = row(self.engine, snapshot_id)
        assert stored.fetched_at == FETCHED_AT
        assert stored.fetched_at.tzinfo is UTC

    def test_returns_a_freshly_inserted_id(self) -> None:
        first = record_snapshot(
            self.engine, entity_id=None, url=URL, result=make_result(), source_kind="page"
        )
        second = record_snapshot(
            self.engine, entity_id=None, url=URL, result=make_result(), source_kind="page"
        )

        assert isinstance(first, int)
        assert isinstance(second, int)
        assert first != second


class TestInsertOnly(StoreFixture):
    def test_identical_input_makes_two_rows(self) -> None:
        result = make_result()

        first = record_snapshot(
            self.engine, entity_id=3, url=URL, result=result, source_kind="homepage"
        )
        second = record_snapshot(
            self.engine, entity_id=3, url=URL, result=result, source_kind="homepage"
        )

        assert first != second
        stored = rows(self.engine)
        assert [item.id for item in stored] == [first, second]
        assert [item.sha256 for item in stored] == [result.sha256, result.sha256]


class TestValidation(StoreFixture):
    def test_empty_url_raises_before_any_write(self) -> None:
        with pytest.raises(ValueError, match="url"):
            record_snapshot(
                self.engine, entity_id=None, url="", result=make_result(), source_kind="page"
            )

        assert rows(self.engine) == []

    def test_empty_source_kind_raises_before_any_write(self) -> None:
        with pytest.raises(ValueError, match="source_kind"):
            record_snapshot(
                self.engine, entity_id=None, url=URL, result=make_result(), source_kind=""
            )

        assert rows(self.engine) == []

    def test_valid_rows_are_kept_when_a_later_call_is_invalid(self) -> None:
        snapshot_id = record_snapshot(
            self.engine, entity_id=None, url=URL, result=make_result(), source_kind="page"
        )

        with pytest.raises(ValueError):
            record_snapshot(
                self.engine, entity_id=None, url="", result=make_result(), source_kind="page"
            )

        assert [item.id for item in rows(self.engine)] == [snapshot_id]


class TestSchema(StoreFixture):
    def test_create_all_creates_the_snapshot_table(self) -> None:
        inspector = sa.inspect(self.engine)
        assert "snapshot" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("snapshot")}
        assert columns == {
            "id",
            "entity_id",
            "url",
            "final_url",
            "content_type",
            "sha256",
            "fetched_at",
            "source_kind",
        }
        nullable = {
            column["name"]: column["nullable"] for column in inspector.get_columns("snapshot")
        }
        assert nullable["entity_id"] is True
        assert nullable["id"] is False

    def test_entity_id_is_indexed_without_a_foreign_key(self) -> None:
        inspector = sa.inspect(self.engine)
        indexed_columns = {
            name for index in inspector.get_indexes("snapshot") for name in index["column_names"]
        }
        assert "entity_id" in indexed_columns
        assert inspector.get_foreign_keys("snapshot") == []

    def test_sha256_is_indexed(self) -> None:
        inspector = sa.inspect(self.engine)
        indexed_columns = {
            name for index in inspector.get_indexes("snapshot") for name in index["column_names"]
        }
        assert "sha256" in indexed_columns


class TestEndToEnd(StoreFixture):
    def test_stored_snapshot_id_is_accepted_by_store_chunks(self) -> None:
        snapshot_id = record_snapshot(
            self.engine, entity_id=11, url=URL, result=make_result(), source_kind="document"
        )

        written = store_chunks(
            self.engine,
            snapshot_id=snapshot_id,
            chunks=[
                Chunk(
                    seq=0,
                    text="first chunk",
                    page=None,
                    heading_path=("Alpha",),
                    char_start=0,
                    char_end=11,
                )
            ],
        )

        assert written == 1
        with Session(self.engine) as session:
            stored = list(
                session.scalars(sa.select(ChunkRow).where(ChunkRow.snapshot_id == snapshot_id))
            )
        assert [(item.snapshot_id, item.seq, item.text) for item in stored] == [
            (snapshot_id, 0, "first chunk")
        ]


@pytest.mark.skipif(
    POSTGRES_URL is None, reason="DM_TEST_DATABASE_URL is not set; PostgreSQL path not exercised"
)
class TestPostgres:
    def test_row_round_trips_on_postgres(self) -> None:
        assert POSTGRES_URL is not None
        engine = create_engine(POSTGRES_URL)
        try:
            with engine.begin() as connection:
                SnapshotRow.__table__.drop(connection, checkfirst=True)
            create_all(engine)
            try:
                snapshot_id = record_snapshot(
                    engine, entity_id=None, url=URL, result=make_result(), source_kind="page"
                )
                stored = row(engine, snapshot_id)
                assert stored.fetched_at == FETCHED_AT
                assert stored.fetched_at.tzinfo is not None
                assert stored.sha256 == hashlib.sha256(BODY).hexdigest()
            finally:
                with engine.begin() as connection:
                    SnapshotRow.__table__.drop(connection, checkfirst=True)
        finally:
            engine.dispose()
