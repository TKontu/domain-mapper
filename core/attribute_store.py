"""``attribute_value`` rows for quantitative attributes (AT-01).

ARCHITECTURE.md §2 (an Attribute is a typed int/float/enum/bool/date fact), §4 stage 8
(quantitative attributes) and §5: ``attribute_value(id, entity_id, attribute_id, value_json, unit,
confidence, run_id)``. :func:`record_attribute_value` validates a value against the scope pack's
:class:`core.scope_pack.Attribute` and appends one row; :func:`current_attribute_values` returns
the latest row per attribute for one entity.

The declarative base is ``core.runs.Base`` (reused, not duplicated), so ``core.runs.create_all``
creates the ``attribute_value`` table alongside the other tables. Schema migrations remain a
separate item. AT-05 links evidence: ``record_attribute_value(evidence=...)`` appends
``evidence`` rows with ``attribute_value_id`` set, and :func:`attribute_evidence_many` reads them
back in batches of <=500 ids. Both are unwired by design; attribute extraction is their production
caller.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.assessment_store import CurrentEvidence, EvidenceRow
from core.runs import Base
from core.scope_pack import Attribute


class AttributeValueRow(Base):
    """One observed value for one entity attribute; append-only, one row per observation."""

    __tablename__ = "attribute_value"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed without a foreign key, like screen_result.entity_id: the row outlives entity deletes.
    entity_id: Mapped[int] = mapped_column(index=True)
    attribute_id: Mapped[str] = mapped_column(sa.String(128), index=True)
    value_json: Mapped[Any] = mapped_column(sa.JSON)
    unit: Mapped[str | None] = mapped_column(sa.String(64))
    confidence: Mapped[float | None] = mapped_column()
    run_id: Mapped[int | None] = mapped_column(sa.ForeignKey("run.id"), index=True)


def _checked_value(attribute: Attribute, value: object) -> Any:
    """Return the JSON-storable form of ``value``; raise ``ValueError`` if it is not valid.

    ``bool`` is a subclass of ``int``, so it is rejected explicitly for ``int`` and ``float``.
    A ``float`` attribute accepts a whole number and stores it as a float, so the value keeps its
    type across the JSON round trip. A ``date`` attribute accepts an ISO 8601 string or a
    :class:`datetime.date` (not a ``datetime``) and always stores the ISO date string.
    """
    kind = attribute.type
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"attribute {attribute.id!r} expects an int value, got {type(value).__name__}"
            )
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"attribute {attribute.id!r} expects a float value, got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"attribute {attribute.id!r} expects a finite float value")
        return number
    if kind == "enum":
        if not isinstance(value, str):
            raise ValueError(
                f"attribute {attribute.id!r} expects an enum string, got {type(value).__name__}"
            )
        if attribute.values and value not in attribute.values:
            raise ValueError(
                f"attribute {attribute.id!r} expects one of {attribute.values!r}, got {value!r}"
            )
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(
                f"attribute {attribute.id!r} expects a bool value, got {type(value).__name__}"
            )
        return value
    if kind == "date":
        if isinstance(value, datetime):
            raise ValueError(f"attribute {attribute.id!r} expects an ISO 8601 date, got datetime")
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            raise ValueError(
                f"attribute {attribute.id!r} expects an ISO 8601 date, got {type(value).__name__}"
            )
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"attribute {attribute.id!r} expects an ISO 8601 date, got {value!r}"
            ) from None
        return value
    raise ValueError(  # pragma: no cover - the Attribute Literal excludes other types
        f"attribute {attribute.id!r} has unsupported type {kind!r}"
    )


def record_attribute_value(
    session: Session,
    entity_id: int,
    attribute: Attribute,
    value: object,
    *,
    confidence: float | None = None,
    run_id: int | None = None,
    evidence: Sequence[tuple[int, str, bool]] = (),
) -> int:
    """Validate ``value`` and append one ``attribute_value`` row; return its id.

    The value is checked against ``attribute`` before the session is touched, so an invalid value
    writes nothing. The row stores the validated ``value_json`` and the attribute's ``unit``. The
    caller owns the transaction; this function flushes but does not commit. Each
    ``(chunk_id, quote, grounded)`` in ``evidence`` appends one ``evidence`` row linked by
    ``attribute_value_id`` (``assessment_id`` NULL) in the same session.
    """
    stored = _checked_value(attribute, value)
    row = AttributeValueRow(
        entity_id=entity_id,
        attribute_id=attribute.id,
        value_json=stored,
        unit=attribute.unit,
        confidence=confidence,
        run_id=run_id,
    )
    session.add(row)
    session.flush()
    for chunk_id, quote, grounded in evidence:
        session.add(
            EvidenceRow(
                attribute_value_id=row.id, chunk_id=chunk_id, quote=quote, grounded=grounded
            )
        )
    if evidence:
        session.flush()
    return row.id


ATTRIBUTE_EVIDENCE_BATCH = 500


def attribute_evidence_many(
    session: Session, attribute_value_ids: Sequence[int]
) -> dict[int, list[CurrentEvidence]]:
    """Return evidence per attribute value id, in evidence id order; read-only (AT-05).

    One query per <=500 ids; values without evidence are absent; empty input returns ``{}``
    without querying.
    """
    ids = list(attribute_value_ids)
    found: dict[int, list[CurrentEvidence]] = {}
    for start in range(0, len(ids), ATTRIBUTE_EVIDENCE_BATCH):
        stmt = (
            sa.select(EvidenceRow)
            .where(
                EvidenceRow.attribute_value_id.in_(ids[start : start + ATTRIBUTE_EVIDENCE_BATCH])
            )
            .order_by(EvidenceRow.attribute_value_id, EvidenceRow.id)
        )
        for ev in session.scalars(stmt):
            if ev.attribute_value_id is None:  # pragma: no cover - excluded by the IN filter
                continue
            found.setdefault(ev.attribute_value_id, []).append(
                CurrentEvidence(ev.chunk_id, ev.quote, ev.grounded)
            )
    return found


def current_attribute_values(session: Session, entity_id: int) -> dict[str, Any]:
    """Return ``entity_id``'s latest decoded value per attribute id; read-only.

    Latest is the highest ``id``. Only one entity's rows are read, and the query returns just the
    newest row per attribute rather than every version. An entity with no values yields ``{}``.
    """
    latest_ids = (
        sa.select(sa.func.max(AttributeValueRow.id).label("id"))
        .where(AttributeValueRow.entity_id == entity_id)
        .group_by(AttributeValueRow.attribute_id)
        .subquery()
    )
    stmt = (
        sa.select(AttributeValueRow.attribute_id, AttributeValueRow.value_json)
        .join(latest_ids, AttributeValueRow.id == latest_ids.c.id)
        .order_by(AttributeValueRow.attribute_id)
    )
    return {attribute_id: value for attribute_id, value in session.execute(stmt)}


def current_attribute_value_rows_many(
    session: Session, entity_ids: Sequence[int]
) -> dict[int, dict[str, tuple[int, Any, str | None]]]:
    """Return ``(attribute_value id, decoded value, unit)`` of the latest row per attribute per
    entity.

    Read-only (AT-10, AT-11). One query for all ids (callers bound the ``IN`` list); latest is the
    highest ``id``; ``unit`` is the stored row's unit (``None`` when it has none); entities without
    values are absent; empty input returns ``{}`` without querying.
    Production caller: ``app.cli.run_attributes_export`` (to look up current-value evidence
    and unit).
    """
    if not entity_ids:
        return {}
    latest_ids = (
        sa.select(sa.func.max(AttributeValueRow.id).label("id"))
        .where(AttributeValueRow.entity_id.in_(list(entity_ids)))
        .group_by(AttributeValueRow.entity_id, AttributeValueRow.attribute_id)
        .subquery()
    )
    stmt = (
        sa.select(
            AttributeValueRow.entity_id,
            AttributeValueRow.attribute_id,
            AttributeValueRow.id,
            AttributeValueRow.value_json,
            AttributeValueRow.unit,
        )
        .join(latest_ids, AttributeValueRow.id == latest_ids.c.id)
        .order_by(AttributeValueRow.entity_id, AttributeValueRow.attribute_id)
    )
    result: dict[int, dict[str, tuple[int, Any, str | None]]] = {}
    for entity_id, attribute_id, row_id, value, unit in session.execute(stmt):
        result.setdefault(entity_id, {})[attribute_id] = (row_id, value, unit)
    return result


def current_attribute_values_many(
    session: Session, entity_ids: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Return the latest decoded value per attribute for each of ``entity_ids``; read-only (AT-04).

    One query for all ids (callers bound the ``IN`` list; ``run_attributes_extract`` uses <=500).
    Latest is the highest ``id``; entities without values are absent; empty input returns ``{}``
    without querying. Delegates to :func:`current_attribute_value_rows_many`.
    """
    return {
        entity_id: {attribute_id: value for attribute_id, (_, value, _unit) in values.items()}
        for entity_id, values in current_attribute_value_rows_many(session, entity_ids).items()
    }
