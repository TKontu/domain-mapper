"""``assessment`` and ``evidence`` rows for the Assess stage (AS-10).

ARCHITECTURE.md §4 stage 7 and §5: every completed ``landscape assess`` appends one
``assessment`` row (entity × criterion → status, rationale, provenance and route) plus one
``evidence`` row per extractor quote. Assessments are append-only; the "current" view is a later
item (latest per entity, criterion and rubric version). :func:`record_assessment` writes the
assessment and all its evidence in one transaction, so a row is never left without its quotes.

Only the §5 columns that this stage can fill are present now. ``prompt_version``,
``inherited_from_entity_id`` and the evidence ``source_id``/``source_tier`` need later stages
(cache keys, group inheritance and the source catalogue); schema migrations remain a separate
item, as in ``core.screen_store``.

The declarative base is ``core.runs.Base`` (reused, not duplicated), so
``core.runs.create_all`` creates both tables. Consumers (the review queue, current-view queries
and reporting) are later rows, unwired by design; the production writer is ``landscape assess``
(:func:`core.cli.run_assess`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Dialect, Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import TypeDecorator

from core.assess import Extraction
from core.runs import Base
from core.verify import Route

DECIDED_BY_AUTO: Final = "auto"
"""Every row this stage writes is decided automatically (not by cloud LLM or a human)."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """Store an aware datetime as UTC; read it back as aware UTC.

    Same contract as :class:`core.screen_store.UtcDateTime`; SQLite persists wall time only, so
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


class AssessmentRow(Base):
    """One entity × criterion assessment; append-only, one row per completed assess."""

    __tablename__ = "assessment"
    __table_args__ = (
        sa.CheckConstraint(
            "decided_by IN ('auto', 'cloud_llm', 'human')", name="ck_assessment_decided_by"
        ),
        sa.CheckConstraint(
            "route_kind IN ('accept', 'adjudicate', 'silent')", name="ck_assessment_route_kind"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed without a foreign key, like screen_result.entity_id: the row outlives entity deletes.
    entity_id: Mapped[int] = mapped_column(index=True)
    criterion_id: Mapped[str] = mapped_column(sa.String(128))
    status: Mapped[str] = mapped_column(sa.String(64))
    # Reserved for a later stage (cloud adjudication / human review set it); always NULL here.
    confidence: Mapped[float | None] = mapped_column()
    rationale: Mapped[str] = mapped_column(sa.Text)
    rubric_version: Mapped[str] = mapped_column(sa.String(64))
    extractor_model: Mapped[str] = mapped_column(sa.String(128))
    verifier_model: Mapped[str] = mapped_column(sa.String(128))
    decided_by: Mapped[str] = mapped_column(sa.String(16), default=DECIDED_BY_AUTO)
    route_kind: Mapped[str] = mapped_column(sa.String(16))
    run_id: Mapped[int] = mapped_column(sa.ForeignKey("run.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utcnow)


class EvidenceRow(Base):
    """One extractor quote backing an assessment; ``grounded`` records the check outcome."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(sa.ForeignKey("assessment.id"), index=True)
    chunk_id: Mapped[int] = mapped_column(index=True)
    quote: Mapped[str] = mapped_column(sa.Text)
    grounded: Mapped[bool] = mapped_column()


def record_assessment(
    engine: Engine,
    *,
    entity_id: int,
    criterion_id: str,
    extraction: Extraction,
    route: Route,
    rubric_version: str,
    extractor_model: str,
    verifier_model: str,
    run_id: int,
) -> int:
    """Append one assessment and one evidence row per quote in one transaction; return its id.

    ``confidence`` stays NULL and ``decided_by`` is ``auto``: an accept route already records
    the extractor/verifier provenance, while cloud adjudication and human review are later
    stages. Any database error rolls the whole write back, so no assessment is stored without
    its evidence.
    """
    with Session(engine) as session, session.begin():
        row = AssessmentRow(
            entity_id=entity_id,
            criterion_id=criterion_id,
            status=extraction.status,
            confidence=None,
            rationale=extraction.rationale,
            rubric_version=rubric_version,
            extractor_model=extractor_model,
            verifier_model=verifier_model,
            decided_by=DECIDED_BY_AUTO,
            route_kind=route.kind,
            run_id=run_id,
        )
        session.add(row)
        session.flush()
        assessment_id: int = row.id
        for quote in extraction.quotes:
            session.add(
                EvidenceRow(
                    assessment_id=assessment_id,
                    chunk_id=quote.chunk_id,
                    quote=quote.text,
                    grounded=quote.grounded,
                )
            )
        return assessment_id
