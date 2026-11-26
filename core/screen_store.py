"""``screen_result`` rows and bounded entity text for the Screen stage (SC-02).

ARCHITECTURE.md §4 stage 3 (Screen & classify) and §5: one ``screen_result`` row records the
small local model's verdict for one entity — in scope, its scope segments, confidence, reason and
whether it needs review. :func:`record_screen_result` also moves the entity's status
(``in_scope``/``out_of_scope``/``borderline``) in the same transaction, so a result and its status
never disagree.

:func:`load_entity_text` reads the entity's stored chunks through ``chunk`` → ``snapshot`` in
``(snapshot_id, seq)`` order and stops once ``max_chars`` have been collected, so one entity's
whole chunk set is never held in memory. Rows are yielded in batches by the driver.

The declarative base is ``core.runs.Base`` (reused, not duplicated), so
``core.runs.create_all`` creates the ``screen_result`` table alongside the run, entity, snapshot
and chunk tables. Schema migrations remain a separate item. Consumption by the review queue and
the Assess stage is a later row (unwired here); the production writer is ``landscape screen``.
:func:`latest_screen_segments` is read by ``core.cli.run_scores`` (``landscape scores``, AG-02).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Dialect, Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import TypeDecorator

from core.chunk_store import ChunkRow
from core.entity_store import Entity
from core.runs import Base
from core.screen import ScreenResult
from core.snapshot_store import SnapshotRow

DEFAULT_MAX_CHARS: Final = 8000
"""Characters of entity text handed to ``screen_entity`` (matches its SC-01 default)."""

STATUS_IN_SCOPE: Final = "in_scope"
STATUS_OUT_OF_SCOPE: Final = "out_of_scope"
STATUS_BORDERLINE: Final = "borderline"

_TEXT_SEPARATOR: Final = "\n\n"
_TEXT_BATCH: Final = 100


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """Store an aware datetime as UTC; read it back as aware UTC.

    Same contract as :class:`core.snapshot_store.UtcDateTime`; SQLite persists wall time only, so
    the value is converted to UTC wall time on bind and re-tagged UTC on read.
    """

    impl = sa.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        utc = value.astimezone(UTC)
        return utc.replace(tzinfo=None) if dialect.name == "sqlite" else utc

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ScreenResultRow(Base):
    """One model screening verdict for one entity; append-only, one row per screening."""

    __tablename__ = "screen_result"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed without a foreign key, like snapshot.entity_id: the row outlives entity deletes.
    entity_id: Mapped[int] = mapped_column(index=True)
    run_id: Mapped[int] = mapped_column(sa.ForeignKey("run.id"), index=True)
    in_scope: Mapped[bool] = mapped_column()
    segments: Mapped[list[str]] = mapped_column(sa.JSON)
    confidence: Mapped[float] = mapped_column()
    reason: Mapped[str] = mapped_column(sa.Text)
    borderline: Mapped[bool] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utcnow)


def status_for_result(result: ScreenResult) -> str:
    """Map a screening verdict to the entity status stored in ``entity.status``."""
    if result.borderline:
        return STATUS_BORDERLINE
    return STATUS_IN_SCOPE if result.in_scope else STATUS_OUT_OF_SCOPE


def record_screen_result(
    engine: Engine,
    *,
    entity_id: int,
    run_id: int,
    result: ScreenResult,
) -> str:
    """Write ``result`` and set the entity's status in one transaction; return the status.

    Raises :class:`ValueError` without writing anything when ``entity_id`` does not exist, so a
    result row can never be stored without the matching status change.
    """
    status = status_for_result(result)
    with Session(engine) as session, session.begin():
        entity = session.get(Entity, entity_id)
        if entity is None:
            raise ValueError(f"entity {entity_id} not found")
        entity.status = status
        session.add(
            ScreenResultRow(
                entity_id=entity_id,
                run_id=run_id,
                in_scope=result.in_scope,
                segments=list(result.segments),
                confidence=result.confidence,
                reason=result.reason,
                borderline=result.borderline,
            )
        )
    return status


def latest_screen_segments(engine: Engine, entity_id: int) -> tuple[str, ...] | None:
    """Return the segments of ``entity_id``'s latest screen result; read-only.

    Latest is the highest ``created_at``, ties broken by the highest ``id``. ``None`` when the
    entity has no screen result. Production caller: ``core.cli.run_scores``.
    """
    stmt = (
        sa.select(ScreenResultRow.segments)
        .where(ScreenResultRow.entity_id == entity_id)
        .order_by(ScreenResultRow.created_at.desc(), ScreenResultRow.id.desc())
        .limit(1)
    )
    with Session(engine) as session:
        segments = session.execute(stmt).scalar_one_or_none()
    return None if segments is None else tuple(segments)


def load_entity_text(
    engine: Engine,
    entity_id: int,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Return up to ``max_chars`` of ``entity_id``'s stored chunk text.

    Chunks are read through their snapshot in ``(snapshot_id, seq)`` order and joined with blank
    lines. Reading stops as soon as ``max_chars`` have been collected; the result is truncated to
    exactly ``max_chars`` so the model prompt is bounded. An entity with no chunks yields ``""``.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    stmt = (
        sa.select(ChunkRow.text)
        .join(SnapshotRow, SnapshotRow.id == ChunkRow.snapshot_id)
        .where(SnapshotRow.entity_id == entity_id)
        .order_by(ChunkRow.snapshot_id, ChunkRow.seq)
    )
    parts: list[str] = []
    collected = 0
    with Session(engine) as session:
        rows = session.execute(stmt.execution_options(yield_per=_TEXT_BATCH))
        for (chunk_text,) in rows:
            parts.append(chunk_text)
            collected += len(chunk_text)
            if collected >= max_chars:
                break
    return _TEXT_SEPARATOR.join(parts)[:max_chars]
