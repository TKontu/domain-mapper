"""``entity`` rows: one resolved organisation per scope and primary domain (RV-02).

ARCHITECTURE.md §5 lists ``entity(id, scope_id, group_id, name, primary_domain, status,
created_run)``. Rows are written by :func:`core.resolve.apply_resolution`; ``(scope_id,
primary_domain)`` is unique so a rerun reuses an existing entity instead of duplicating it.

``scope_id`` is the scope pack id (``scope.yaml`` ``id``). ``group_id`` (no FK: groups/GLEIF are a
later item) and ``created_run`` are nullable. New rows start ``status = "unscreened"``.
Migrations remain a separate item; ``core.runs.create_all`` creates the table.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from core.runs import Base

STATUS_UNSCREENED = "unscreened"


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
