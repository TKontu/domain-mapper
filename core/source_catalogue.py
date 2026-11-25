"""``source`` catalogue table seeded from a scope pack (ARCHITECTURE.md §5, AD-02).

The table lives on the shared :class:`core.runs.Base` metadata, so ``create_all``
creates it next to the run-tracking tables. :func:`seed_sources` upserts a loaded
pack's ``sources.yaml`` entries in one transaction: a new entry is inserted, a
changed entry updates its row, and an unchanged entry writes nothing.

Row identity is ``(scope_id, id)``: a pack's source ids are pack-local, and two
packs may legitimately name the same source (for example a public discovery
channel), so a bare ``id`` key would make one pack's seed overwrite another's.

The seeder is unwired by design: the Phase 1 Discover row wires it through the
CLI. It records the catalogue only; retrieval is the adapters' job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.runs import Base
from core.scope_pack import ScopePack
from core.scope_pack import Source as PackSource


class Source(Base):
    """One catalogue row: a scope pack's source, scoped to its pack."""

    __tablename__ = "source"
    __table_args__ = (sa.Index("ix_source_scope_id", "scope_id"),)

    # Pack-local id, unique only within its scope: see the module docstring.
    id: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    scope_id: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(256))
    # A pack source declares one role; the catalogue keeps the architecture's
    # list shape so a later item can add roles without a schema change.
    roles: Mapped[list[str]] = mapped_column(sa.JSON)
    access_class: Mapped[str | None] = mapped_column(sa.String(64))
    licence: Mapped[str | None] = mapped_column(sa.String(256))
    base_url: Mapped[str | None] = mapped_column(sa.String(1024))
    adapter: Mapped[str] = mapped_column(sa.String(64))


@dataclass(frozen=True)
class SeedSummary:
    """What one seed changed; the rows themselves stay in the database."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


def _values(entry: PackSource) -> dict[str, Any]:
    """Map one pack source onto catalogue columns.

    ``params.name`` / ``params.base_url`` win when present; otherwise the name
    falls back to the pack id and the URL stays NULL. ``countries`` is not
    persisted (ARCHITECTURE.md §5 has no column for it).
    """
    params = entry.params
    name = params.get("name")
    base_url = params.get("base_url")
    return {
        "name": str(name) if name else entry.id,
        "roles": [entry.role],
        "access_class": entry.access,
        "licence": entry.licence,
        "base_url": str(base_url) if base_url else None,
        "adapter": entry.adapter,
    }


def seed_sources(engine: Engine, pack: ScopePack) -> SeedSummary:
    """Upsert ``pack.sources`` into the ``source`` table idempotently.

    One transaction per seed covers every entry (source lists are tens, so this
    is bounded and keeps a half-seeded catalogue out of reach). Entries already
    present with identical values are left untouched, so re-seeding the same
    pack issues no INSERT or UPDATE. An entry removed from the pack keeps its
    row: candidates may reference it, and retirement is a later decision.
    """
    scope_id = pack.scope.id
    inserted = updated = unchanged = 0
    with Session(engine) as session, session.begin():
        existing = {
            row.id: row
            for row in session.scalars(sa.select(Source).where(Source.scope_id == scope_id))
        }
        for entry in pack.sources.sources:
            values = _values(entry)
            row = existing.get(entry.id)
            if row is None:
                session.add(Source(scope_id=scope_id, id=entry.id, **values))
                inserted += 1
            elif any(getattr(row, column) != value for column, value in values.items()):
                for column, value in values.items():
                    setattr(row, column, value)
                updated += 1
            else:
                unchanged += 1
    return SeedSummary(inserted=inserted, updated=updated, unchanged=unchanged)
