"""Idempotent ``chunk`` row persistence for the Parse & chunk stage (CH-03).

ARCHITECTURE.md §4 (Parse & chunk) keeps ``snapshot_id``, page, heading path and
character offsets on every chunk; §5 lists the ``chunk`` table columns.
:func:`store_chunks` writes one snapshot's :class:`~core.chunking.Chunk`
iterator to that table so a retry replaces the snapshot's rows instead of
duplicating them. It is **unwired by design**: the CLI ``parse`` stage wires it
once snapshot rows exist (that row also supplies the snapshot table).

Semantics:

* One transaction per snapshot: delete the snapshot's existing rows, then
  insert the new ones. An error anywhere — including while the input iterator
  is being consumed — rolls the whole snapshot back to its previous rows.
* ``chunks`` is consumed lazily in batches of ``batch_size`` records, so at
  most one batch plus the current chunk is held in memory; the input is never
  materialised as a list.
* ``lang`` is written NULL until language detection runs; ``heading_path`` is
  stored as a JSON list of strings, matching the ``Chunk`` tuple on read.
* ``batch_size < 1`` raises :class:`ValueError` before the transaction opens,
  so a bad argument cannot delete or write anything.
* The return value is the number of rows written for the snapshot (0 when the
  input is empty, which still clears the snapshot's rows).

The declarative base is ``core.runs.Base`` (reused, not duplicated), so
``core.runs.create_all`` creates the ``chunk`` table alongside the run tables.
Schema migrations remain a separate item (residual below). No ``embedding``
column: the Index row adds it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.chunking import Chunk
from core.runs import Base

DEFAULT_BATCH_SIZE: Final = 500


class ChunkRow(Base):
    """One persisted chunk; ``(snapshot_id, seq)`` is unique per snapshot."""

    __tablename__ = "chunk"
    __table_args__ = (sa.UniqueConstraint("snapshot_id", "seq", name="uq_chunk_snapshot_seq"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed without a foreign key: the snapshot table row lands in a later item.
    snapshot_id: Mapped[int] = mapped_column(index=True)
    seq: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column(sa.Text)
    lang: Mapped[str | None] = mapped_column(sa.String(16))
    page: Mapped[int | None] = mapped_column()
    heading_path: Mapped[list[str]] = mapped_column(sa.JSON)
    char_start: Mapped[int] = mapped_column()
    char_end: Mapped[int] = mapped_column()


def store_chunks(
    engine: Engine,
    snapshot_id: int,
    chunks: Iterable[Chunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Replace ``snapshot_id``'s chunk rows with ``chunks``; return rows written."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    written = 0
    with Session(engine) as session, session.begin():
        session.execute(sa.delete(ChunkRow).where(ChunkRow.snapshot_id == snapshot_id))
        batch: list[ChunkRow] = []
        for chunk in chunks:
            batch.append(_row(snapshot_id, chunk))
            if len(batch) >= batch_size:
                written += _flush(session, batch)
                batch.clear()
        if batch:
            written += _flush(session, batch)
    return written


def _row(snapshot_id: int, chunk: Chunk) -> ChunkRow:
    return ChunkRow(
        snapshot_id=snapshot_id,
        seq=chunk.seq,
        text=chunk.text,
        lang=None,
        page=chunk.page,
        heading_path=list(chunk.heading_path),
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )


def _flush(session: Session, batch: list[ChunkRow]) -> int:
    session.add_all(batch)
    session.flush()
    return len(batch)
