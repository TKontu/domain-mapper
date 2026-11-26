"""``assessment`` and ``evidence`` rows for the Assess stage (AS-10).

ARCHITECTURE.md §4 stage 7 and §5: every completed ``landscape assess`` appends one
``assessment`` row (entity × criterion → status, rationale, provenance and route) plus one
``evidence`` row per extractor quote. Assessments are append-only; the "current" view is a later
item (latest per entity, criterion and rubric version). :func:`record_assessment` writes the
assessment and all its evidence in one transaction, so a row is never left without its quotes.
:func:`current_assessments` (AS-11) is the read-only current view: the latest row per entity,
criterion and rubric version with its evidence, read by ``landscape assessments``.

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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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

_IN_CHUNK: Final = 500
"""Evidence ids per ``IN`` list, well under SQLite and PostgreSQL bind limits."""


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


@dataclass(frozen=True)
class CurrentEvidence:
    """One evidence row of a current assessment."""

    chunk_id: int
    quote: str
    grounded: bool


@dataclass(frozen=True)
class CurrentAssessment:
    """The latest assessment for one (entity, criterion, rubric version) with its evidence."""

    id: int
    entity_id: int
    criterion_id: str
    status: str
    confidence: float | None
    rationale: str
    rubric_version: str
    extractor_model: str
    verifier_model: str
    decided_by: str
    route_kind: str
    run_id: int
    created_at: datetime
    evidence: tuple[CurrentEvidence, ...]


def current_assessments(
    engine: Engine,
    *,
    entity_id: int | None = None,
    criterion_id: str | None = None,
    rubric_version: str | None = None,
) -> list[CurrentAssessment]:
    """Return the latest assessment per (entity, criterion, rubric version); read-only.

    The whole view as a list of :func:`iter_current_assessments`; same order and filters.
    """
    return list(
        iter_current_assessments(
            engine,
            entity_id=entity_id,
            criterion_id=criterion_id,
            rubric_version=rubric_version,
        )
    )


def iter_current_assessments(
    engine: Engine,
    *,
    entity_id: int | None = None,
    entity_ids: Sequence[int] | None = None,
    criterion_id: str | None = None,
    rubric_version: str | None = None,
    batch_size: int = 500,
) -> Iterator[CurrentAssessment]:
    """Yield the latest assessment per (entity, criterion, rubric version) in batches; read-only.

    Latest is the highest ``created_at``, ties broken by the highest ``id``. One windowed SQL
    statement (SQLite >= 3.25 and PostgreSQL) selects the rows, filters applied before ranking,
    fetched ``batch_size`` rows at a time; each batch's evidence is fetched with it, so at most
    one batch plus its evidence is held. Ordered by entity id, criterion id and rubric version.
    ``entity_ids`` restricts the view to those entities (an empty sequence yields nothing
    without querying). The session stays open until the generator is exhausted or closed.
    Production callers: ``core.cli.run_export``, ``core.cli.run_scores_export`` and
    :func:`current_assessments`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if entity_ids is not None and not entity_ids:
        return iter(())
    return _iter_current(engine, entity_id, entity_ids, criterion_id, rubric_version, batch_size)


def _iter_current(
    engine: Engine,
    entity_id: int | None,
    entity_ids: Sequence[int] | None,
    criterion_id: str | None,
    rubric_version: str | None,
    batch_size: int,
) -> Iterator[CurrentAssessment]:
    a = AssessmentRow
    rank = (
        sa.func.row_number()
        .over(
            partition_by=(a.entity_id, a.criterion_id, a.rubric_version),
            order_by=(a.created_at.desc(), a.id.desc()),
        )
        .label("rank")
    )
    inner = sa.select(a.id, rank)
    if entity_id is not None:
        inner = inner.where(a.entity_id == entity_id)
    if entity_ids is not None:
        inner = inner.where(a.entity_id.in_(list(entity_ids)))
    if criterion_id is not None:
        inner = inner.where(a.criterion_id == criterion_id)
    if rubric_version is not None:
        inner = inner.where(a.rubric_version == rubric_version)
    ranked = inner.subquery()
    table = AssessmentRow.__table__
    stmt = (
        sa.select(table)
        .join(ranked, ranked.c.id == table.c.id)
        .where(ranked.c.rank == 1)
        .order_by(table.c.entity_id, table.c.criterion_id, table.c.rubric_version)
        .execution_options(yield_per=batch_size)
    )
    with Session(engine) as session:
        for batch in session.execute(stmt).partitions():
            rows = list(batch)
            evidence = _evidence_for(session, [row.id for row in rows])
            for row in rows:
                yield CurrentAssessment(
                    id=row.id,
                    entity_id=row.entity_id,
                    criterion_id=row.criterion_id,
                    status=row.status,
                    confidence=row.confidence,
                    rationale=row.rationale,
                    rubric_version=row.rubric_version,
                    extractor_model=row.extractor_model,
                    verifier_model=row.verifier_model,
                    decided_by=row.decided_by,
                    route_kind=row.route_kind,
                    run_id=row.run_id,
                    created_at=row.created_at,
                    evidence=tuple(evidence.get(row.id, ())),
                )


def _evidence_for(session: Session, ids: Sequence[int]) -> dict[int, list[CurrentEvidence]]:
    found: dict[int, list[CurrentEvidence]] = {}
    for start in range(0, len(ids), _IN_CHUNK):
        chunk = ids[start : start + _IN_CHUNK]
        stmt = (
            sa.select(EvidenceRow)
            .where(EvidenceRow.assessment_id.in_(chunk))
            .order_by(EvidenceRow.assessment_id, EvidenceRow.id)
        )
        for ev in session.scalars(stmt):
            found.setdefault(ev.assessment_id, []).append(
                CurrentEvidence(ev.chunk_id, ev.quote, ev.grounded)
            )
    return found
