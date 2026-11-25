"""One immutable ``snapshot`` row per fetch (CR-01).

ARCHITECTURE.md §4 (Crawl & snapshot) stores each fetched body content-addressed
by ``sha256`` and links it to the entity, URL and time; §5 lists the ``snapshot``
table columns. :func:`record_snapshot` writes that metadata row and returns its
id, which :func:`core.chunk_store.store_chunks` takes as ``snapshot_id``.

Semantics:

* One row per call, never updated or deleted by this module. Two calls with
  identical input produce two rows (re-crawls are distinct observations).
* ``sha256`` is the lowercase hex digest of ``result.body_bytes``; no raw bytes
  are stored here (the raw-storage backend is an open decision).
* ``entity_id`` is nullable and indexed without a foreign key: the entity table
  lands in a later item.
* ``url`` or ``source_kind`` empty raises :class:`ValueError` before a session
  opens, so an invalid call cannot write anything.
* ``fetched_at`` is stored as aware UTC and read back aware UTC on SQLite too
  (see :class:`UtcDateTime`).

The declarative base is ``core.runs.Base`` (reused, not duplicated), so
``core.runs.create_all`` creates the ``snapshot`` table alongside the run and
chunk tables. Schema migrations remain a separate item. The module is
**unwired by design**: the Crawl row's CLI ``crawl`` stage is the production
caller.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy import Dialect, Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import TypeDecorator

from adapters.fetcher import FetchResult
from core.runs import Base


class UtcDateTime(TypeDecorator[datetime]):
    """Store an aware datetime as UTC; read it back as aware UTC.

    SQLite has no timezone-aware datetime type, so a plain
    ``DateTime(timezone=True)`` round-trips as naive wall time. Converting on
    bind and re-attaching UTC on read keeps the column's meaning identical on
    SQLite and PostgreSQL.
    """

    impl = sa.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
        utc = value.astimezone(UTC)
        # SQLite persists wall time only: hand it the UTC wall time without
        # tzinfo so a non-UTC input is not silently stored as local time.
        return utc.replace(tzinfo=None) if dialect.name == "sqlite" else utc

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class SnapshotRow(Base):
    """One fetched observation; append-only, no update or delete path."""

    __tablename__ = "snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed without a foreign key: the entity table row lands in a later item.
    entity_id: Mapped[int | None] = mapped_column(index=True)
    url: Mapped[str] = mapped_column(sa.Text)
    final_url: Mapped[str] = mapped_column(sa.Text)
    content_type: Mapped[str] = mapped_column(sa.String(255))
    sha256: Mapped[str] = mapped_column(sa.String(64), index=True)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime())
    source_kind: Mapped[str] = mapped_column(sa.String(64))


def record_snapshot(
    engine: Engine,
    *,
    entity_id: int | None,
    url: str,
    result: FetchResult,
    source_kind: str,
) -> int:
    """Insert one ``snapshot`` row for ``result``; return its id."""
    if not url:
        raise ValueError("url must not be empty")
    if not source_kind:
        raise ValueError("source_kind must not be empty")

    snapshot = SnapshotRow(
        entity_id=entity_id,
        url=url,
        final_url=result.final_url,
        content_type=result.content_type,
        sha256=hashlib.sha256(result.body_bytes).hexdigest(),
        fetched_at=result.fetched_at,
        source_kind=source_kind,
    )
    with Session(engine) as session, session.begin():
        session.add(snapshot)
        session.flush()
        snapshot_id = snapshot.id
    return snapshot_id
