"""``attribute_value`` rows for quantitative attributes (AT-01).

ARCHITECTURE.md §2 (an Attribute is a typed int/float/enum/bool/date fact), §4 stage 8
(quantitative attributes) and §5: ``attribute_value(id, entity_id, attribute_id, value_json, unit,
confidence, run_id)``. :func:`record_attribute_value` validates a value against the scope pack's
:class:`core.scope_pack.Attribute` and appends one row; :func:`current_attribute_values` returns
the latest row per attribute for one entity.

The declarative base is ``core.runs.Base`` (reused, not duplicated), so ``core.runs.create_all``
creates the ``attribute_value`` table alongside the other tables. Schema migrations remain a
separate item. Evidence linkage (``evidence.attribute_value_id``), extraction and export are later
rows: both functions are unwired by design, and AT-02 (attribute extraction) is their production
caller.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, Session, mapped_column

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
) -> int:
    """Validate ``value`` and append one ``attribute_value`` row; return its id.

    The value is checked against ``attribute`` before the session is touched, so an invalid value
    writes nothing. The row stores the validated ``value_json`` and the attribute's ``unit``. The
    caller owns the transaction; this function flushes but does not commit.
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
    return row.id


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
