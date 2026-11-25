"""``entity`` rows: one resolved organisation per scope and primary domain (RV-02, CR-04).

ARCHITECTURE.md §5 lists ``entity(id, scope_id, group_id, name, primary_domain, status,
created_run)``. Rows are written by :func:`core.resolve.apply_resolution`; ``(scope_id,
primary_domain)`` is unique so a rerun reuses an existing entity instead of duplicating it.
:func:`iter_entities` streams the rows of one scope for the crawl stage, keyset-paged by id.

``scope_id`` is the scope pack id (``scope.yaml`` ``id``). ``group_id`` (no FK: groups/GLEIF are a
later item) and ``created_run`` are nullable. New rows start ``status = "unscreened"``.
Migrations remain a separate item; ``core.runs.create_all`` creates the table.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.runs import Base

STATUS_UNSCREENED = "unscreened"

DEFAULT_BATCH = 500
"""Entity rows fetched per page by :func:`iter_entities`."""


@dataclass(frozen=True, slots=True)
class EntityRef:
    """Identity and homepage host of one entity, as read by :func:`iter_entities`."""

    id: int
    primary_domain: str


class Entity(Base):
    """One resolved entity within a scope."""

    __tablename__ = "entity"
    __table_args__ = (sa.UniqueConstraint("scope_id", "primary_domain"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[str] = mapped_column(sa.String(255))
    group_id: Mapped[int | None] = mapped_column()
    name: Mapped[str] = mapped_column(sa.Text)
    primary_domain: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[str] = mapped_column(sa.String(32))
    created_run: Mapped[int | None] = mapped_column()


def iter_entities(
    engine: Engine,
    scope_id: str,
    *,
    batch: int = DEFAULT_BATCH,
    limit: int | None = None,
) -> Iterator[EntityRef]:
    """Yield the entities of ``scope_id`` in id order, at most ``limit`` when given.

    Rows are read by keyset (``id > last``, ``LIMIT batch``) in short-lived sessions, so memory
    holds one page, not the whole scope. ``batch`` and ``limit`` must be positive.

    Production caller: ``landscape crawl`` (:func:`core.cli.run_crawl`); with no ``--url`` it
    crawls one homepage per yielded entity.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    return _iter_entities(engine, scope_id, batch=batch, limit=limit)


def _iter_entities(
    engine: Engine, scope_id: str, *, batch: int, limit: int | None
) -> Iterator[EntityRef]:
    last_id = 0
    remaining = limit
    while remaining is None or remaining > 0:
        page_size = batch if remaining is None else min(batch, remaining)
        stmt = (
            sa.select(Entity.id, Entity.primary_domain)
            .where(Entity.scope_id == scope_id, Entity.id > last_id)
            .order_by(Entity.id)
            .limit(page_size)
        )
        with Session(engine) as session:
            rows = list(session.execute(stmt))
        if not rows:
            return
        for entity_id, primary_domain in rows:
            yield EntityRef(id=entity_id, primary_domain=primary_domain)
        last_id = rows[-1][0]
        if remaining is not None:
            remaining -= len(rows)
