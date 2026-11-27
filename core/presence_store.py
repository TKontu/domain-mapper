"""``market_presence`` rows and layer membership derived from them (MP-01).

ARCHITECTURE.md §5 data model ``market_presence(entity_id, country, presence_type, evidence_id)``
and the Layer row ("membership derived from market presence"). :func:`record_presence` stores one
idempotent ``(entity_id, country, presence_type)`` row; :func:`presences_many` reads rows for many
entities in one query; :func:`layer_ids_for` derives the scope layers a set of countries belongs
to; :func:`presence_evidence_many` adds each row's ``evidence_id`` chunk id (MP-02). The
declarative base is ``core.runs.Base``, so ``core.runs.create_all`` creates the table.
Production callers: ``app.cli.run_presence_set`` and ``app.cli.run_presences``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.runs import Base
from core.scope_pack import Layer

PRESENCE_TYPES = ("hq", "site", "field_service", "sales", "remote")
_COUNTRY = re.compile(r"[A-Z]{2}")


class MarketPresenceRow(Base):
    """One country in which one entity is present, and how."""

    __tablename__ = "market_presence"
    __table_args__ = (sa.UniqueConstraint("entity_id", "country", "presence_type"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_id: Mapped[int] = mapped_column(sa.ForeignKey("entity.id"), index=True)
    country: Mapped[str] = mapped_column(sa.String(2))
    presence_type: Mapped[str] = mapped_column(sa.String(32))
    evidence_id: Mapped[int | None] = mapped_column()


def check_presence(country: str, presence_type: str) -> None:
    """Raise ``ValueError`` unless ``country`` is uppercase alpha-2 and the type is known."""
    if not isinstance(country, str) or not _COUNTRY.fullmatch(country):
        raise ValueError(f"country must be an uppercase ISO-3166 alpha-2 code, got {country!r}")
    if presence_type not in PRESENCE_TYPES:
        raise ValueError(
            f"presence type must be one of {', '.join(PRESENCE_TYPES)}, got {presence_type!r}"
        )


def record_presence(
    engine_or_session: Engine | Session,
    entity_id: int,
    country: str,
    presence_type: str,
    evidence_id: int | None = None,
) -> int:
    """Store one presence row (idempotent on its triple) and return its id.

    ``evidence_id`` (MP-02) is a chunk id. On an existing row a NULL evidence_id is filled by a
    non-NULL value; an existing non-NULL value is kept. The chunk is not checked here.

    Validation runs first, so an invalid value writes nothing. Given a ``Session`` the caller owns
    the transaction (this flushes, does not commit); given an ``Engine`` it commits its own.
    """
    check_presence(country, presence_type)
    if isinstance(engine_or_session, Session):
        return _record(engine_or_session, entity_id, country, presence_type, evidence_id)
    with Session(engine_or_session) as session, session.begin():
        return _record(session, entity_id, country, presence_type, evidence_id)


def _record(
    session: Session, entity_id: int, country: str, presence_type: str, evidence_id: int | None
) -> int:
    existing = session.scalar(
        sa.select(MarketPresenceRow).where(
            MarketPresenceRow.entity_id == entity_id,
            MarketPresenceRow.country == country,
            MarketPresenceRow.presence_type == presence_type,
        )
    )
    if existing is not None:
        if existing.evidence_id is None and evidence_id is not None:
            existing.evidence_id = evidence_id
            session.flush()
        return existing.id
    row = MarketPresenceRow(
        entity_id=entity_id, country=country, presence_type=presence_type, evidence_id=evidence_id
    )
    session.add(row)
    session.flush()
    return row.id


def presences_many(
    engine: Engine | Session, entity_ids: Sequence[int]
) -> dict[int, list[tuple[str, str]]]:
    """Return sorted ``(country, presence_type)`` per entity id; one query per call.

    Callers bound the ``IN`` list (<=500 ids, not enforced). Entities without rows are absent;
    empty input returns ``{}`` without querying.
    """
    if not entity_ids:
        return {}
    stmt = (
        sa.select(
            MarketPresenceRow.entity_id, MarketPresenceRow.country, MarketPresenceRow.presence_type
        )
        .where(MarketPresenceRow.entity_id.in_(list(entity_ids)))
        .order_by(
            MarketPresenceRow.entity_id, MarketPresenceRow.country, MarketPresenceRow.presence_type
        )
    )
    found: dict[int, list[tuple[str, str]]] = {}
    if isinstance(engine, Session):
        rows = list(engine.execute(stmt))
    else:
        with engine.connect() as conn:
            rows = list(conn.execute(stmt))
    for entity_id, country, presence_type in rows:
        found.setdefault(entity_id, []).append((country, presence_type))
    return found


def presence_evidence_many(
    engine: Engine | Session, entity_ids: Sequence[int]
) -> dict[int, list[tuple[str, str, int | None]]]:
    """Like :func:`presences_many` but each row carries its ``evidence_id`` (MP-02); one query.

    Same ordering, absence and empty-input rules; callers bound the ``IN`` list (<=500 ids).
    """
    if not entity_ids:
        return {}
    stmt = (
        sa.select(
            MarketPresenceRow.entity_id,
            MarketPresenceRow.country,
            MarketPresenceRow.presence_type,
            MarketPresenceRow.evidence_id,
        )
        .where(MarketPresenceRow.entity_id.in_(list(entity_ids)))
        .order_by(
            MarketPresenceRow.entity_id, MarketPresenceRow.country, MarketPresenceRow.presence_type
        )
    )
    found: dict[int, list[tuple[str, str, int | None]]] = {}
    if isinstance(engine, Session):
        rows = list(engine.execute(stmt))
    else:
        with engine.connect() as conn:
            rows = list(conn.execute(stmt))
    for entity_id, country, presence_type, evidence_id in rows:
        found.setdefault(entity_id, []).append((country, presence_type, evidence_id))
    return found


def layer_ids_for(layers: Iterable[Layer], countries: Iterable[str]) -> list[str]:
    """Return ids of ``layers`` (in order) containing any of ``countries``, or ``"*"``.

    Any presence type counts. Empty ``countries`` yields ``[]`` (a wildcard layer needs at least
    one country).
    """
    wanted = set(countries)
    if not wanted:
        return []
    return [la.id for la in layers if "*" in la.countries or wanted & set(la.countries)]
