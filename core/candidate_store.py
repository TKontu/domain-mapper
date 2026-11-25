"""``candidate`` rows: one per discovered name/URL observation (DS-02).

ARCHITECTURE.md §1 (Discover) outputs ``candidate(name, url, country_hint, segment_hint, channel,
query, found_at)``; §5 lists the table columns. :func:`record_candidates` inserts a batch in one
transaction and returns the row count.

Semantics:

* Append-only, no dedup: each search is a distinct observation; Resolve merges later.
* ``url`` or ``channel`` blank on any row raises :class:`ValueError` before a session opens, so
  nothing is written. Empty input returns 0 without opening a session.
* ``found_at`` must be timezone-aware; stored and read back as aware UTC (``UtcDateTime``).
* ``source_id`` (no FK to ``source`` yet) and ``resolved_entity_id`` (indexed, no FK) are
  nullable; country/segment hints are NULL for the search channel.

Production caller: ``landscape web-search --save`` (results bounded by ``--limit`` <= 100, so one
transaction is bounded). Consumer: the future Phase 1 Resolve row (unwired by design).
Migrations remain a separate item; ``core.runs.create_all`` creates the table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.runs import Base
from core.snapshot_store import UtcDateTime

_BATCH = 500


class Candidate(Base):
    """One discovered candidate observation."""

    __tablename__ = "candidate"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text)
    url: Mapped[str] = mapped_column(sa.Text)
    country_hint: Mapped[str | None] = mapped_column(sa.String(16))
    segment_hint: Mapped[str | None] = mapped_column(sa.String(255))
    channel: Mapped[str] = mapped_column(sa.String(64))
    # No foreign key yet: linking to the ``source`` catalogue is a later item.
    source_id: Mapped[int | None] = mapped_column()
    query: Mapped[str | None] = mapped_column(sa.Text)
    found_at: Mapped[datetime] = mapped_column(UtcDateTime())
    # Indexed without a foreign key: the entity table lands with Resolve.
    resolved_entity_id: Mapped[int | None] = mapped_column(index=True)


@dataclass(frozen=True, slots=True)
class NewCandidate:
    """Input for one ``candidate`` row."""

    name: str
    url: str
    channel: str
    query: str | None
    found_at: datetime
    country_hint: str | None = None
    segment_hint: str | None = None
    source_id: int | None = None


def record_candidates(engine: Engine, rows: Sequence[NewCandidate]) -> int:
    """Insert ``rows`` in order, in one transaction; return the number written."""
    for i, row in enumerate(rows):
        if not row.url.strip():
            raise ValueError(f"row {i}: url must not be blank")
        if not row.channel.strip():
            raise ValueError(f"row {i}: channel must not be blank")
    if not rows:
        return 0
    with Session(engine) as session, session.begin():
        for start in range(0, len(rows), _BATCH):
            session.add_all(
                Candidate(
                    name=r.name,
                    url=r.url,
                    country_hint=r.country_hint,
                    segment_hint=r.segment_hint,
                    channel=r.channel,
                    source_id=r.source_id,
                    query=r.query,
                    found_at=r.found_at,
                    resolved_entity_id=None,
                )
                for r in rows[start : start + _BATCH]
            )
            session.flush()
    return len(rows)
