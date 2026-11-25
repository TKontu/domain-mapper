"""Per-call LLM cost/latency logging (RT-02).

Every model call writes exactly one ``llm_call`` row (ARCHITECTURE.md §5:
``id, run_id, model, purpose, prompt_hash, tokens_in, tokens_out, cost,
latency_ms, cache_hit``, plus ``created_at`` and the RT-02 additions ``status``
and ``error``). Rows share the run-tracking declarative ``Base`` from
:mod:`core.runs`, so ``run_id`` is a foreign key to ``run.id``.

Two entry points:

* :func:`record_llm_call` writes a row for a call already made (or for a usage
  record obtained elsewhere).
* :func:`log_llm_call` wraps the call, measures its latency in milliseconds and
  writes the row on both success and failure; the body's exception is re-raised.

One row is committed per call in its own short transaction; nothing is buffered
in memory. Prompt and response text are never stored or logged: callers hand
over the prompt, only its SHA-256 ``prompt_hash`` is kept. Error rows keep the
exception's type and message (capped), which the caller must not fill with
prompt/response content.

The module is **unwired by design**: the production caller is the ``LLM``
adapter (todo Phase 0 › Adapters).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from core.runs import Base

logger = logging.getLogger(__name__)

# Call outcomes. A call is ``ok`` unless its body raised.
OK: Final = "ok"
ERROR: Final = "error"
STATUSES: Final[tuple[str, ...]] = (OK, ERROR)

# Error rows are for operators; cap the column value so a huge traceback cannot
# bloat the row (matches the run-tracking tables).
MAX_ERROR_CHARS: Final = 2000
_HASH_RE: Final = re.compile(r"[0-9a-f]{64}")
# USD per call at six decimal places is finer than any provider invoices.
_COST_PRECISION: Final = 6


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LlmCall(Base):
    """One logged model call; append-only, one row per call."""

    __tablename__ = "llm_call"
    __table_args__ = (
        sa.CheckConstraint("status IN ('ok', 'error')", name="ck_llm_call_status"),
        sa.CheckConstraint("latency_ms >= 0", name="ck_llm_call_latency"),
        sa.CheckConstraint("tokens_in >= 0 AND tokens_out >= 0", name="ck_llm_call_tokens"),
        sa.Index("ix_llm_call_run_created", "run_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(sa.ForeignKey("run.id"), index=True)
    model: Mapped[str] = mapped_column(sa.String(128))
    purpose: Mapped[str] = mapped_column(sa.String(64))
    # SHA-256 of the prompt; the prompt itself is never persisted.
    prompt_hash: Mapped[str] = mapped_column(sa.String(64), index=True)
    tokens_in: Mapped[int] = mapped_column(default=0)
    tokens_out: Mapped[int] = mapped_column(default=0)
    cost: Mapped[float] = mapped_column(
        sa.Numeric(12, _COST_PRECISION, asdecimal=False), default=0.0
    )
    latency_ms: Mapped[int] = mapped_column(default=0)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(sa.String(8), default=OK)
    error: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


def hash_prompt(prompt: str) -> str:
    """Return the SHA-256 hex digest stored in ``llm_call.prompt_hash``."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _resolve_prompt_hash(prompt: str | None, prompt_hash: str | None) -> str:
    if prompt_hash is not None:
        if _HASH_RE.fullmatch(prompt_hash) is None:
            raise ValueError("prompt_hash must be a lowercase SHA-256 hex digest")
        return prompt_hash
    if prompt is None:
        raise ValueError("record_llm_call needs a prompt or a precomputed prompt_hash")
    return hash_prompt(prompt)


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return text[: MAX_ERROR_CHARS - 1] + "…"


def _elapsed_ms(started: float) -> int:
    # Rounding can only shave a fraction of a millisecond; never negative.
    return max(0, round((time.perf_counter() - started) * 1000))


def _log_call(values: dict[str, object]) -> None:
    """Emit one structured record; prompt/response text is never passed here."""
    logger.info(
        "llm_call",
        extra={
            "call_id": values["call_id"],
            "run_id": values["run_id"],
            "model": values["model"],
            "purpose": values["purpose"],
            "prompt_hash": values["prompt_hash"],
            "tokens_in": values["tokens_in"],
            "tokens_out": values["tokens_out"],
            "cost": values["cost"],
            "latency_ms": values["latency_ms"],
            "cache_hit": values["cache_hit"],
            "status": values["status"],
        },
    )


def _write_row(session_or_engine: Session | Engine, row: LlmCall) -> int:
    """Persist one row in a short transaction and return its id."""
    if isinstance(session_or_engine, Session):
        session = session_or_engine
        if session.in_transaction():
            session.add(row)
            session.flush()
            return row.id
        with session.begin():
            session.add(row)
            session.flush()
            return row.id

    # expire_on_commit=False keeps the row readable for the log record after the
    # short transaction commits and the session closes.
    with Session(session_or_engine, expire_on_commit=False) as session, session.begin():
        session.add(row)
        session.flush()
        return row.id


def record_llm_call(
    session_or_engine: Session | Engine,
    *,
    run_id: int,
    model: str,
    purpose: str,
    prompt: str | None = None,
    prompt_hash: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost: float = 0.0,
    latency_ms: int = 0,
    cache_hit: bool = False,
    status: str = OK,
    error: str | None = None,
) -> int:
    """Write exactly one ``llm_call`` row and return its id.

    Pass ``prompt`` (hashed here) or an already computed ``prompt_hash``; the
    prompt text is never stored or logged. With an :class:`~sqlalchemy.Engine`
    the row gets its own short transaction; with a :class:`Session` the row joins
    the session's open transaction, or gets a short one when none is open.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    if error is not None:
        error = error if len(error) <= MAX_ERROR_CHARS else error[: MAX_ERROR_CHARS - 1] + "…"

    row = LlmCall(
        run_id=run_id,
        model=model,
        purpose=purpose,
        prompt_hash=_resolve_prompt_hash(prompt, prompt_hash),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        status=status,
        error=error,
    )
    call_id = _write_row(session_or_engine, row)
    _log_call(
        {
            "call_id": call_id,
            "run_id": row.run_id,
            "model": row.model,
            "purpose": row.purpose,
            "prompt_hash": row.prompt_hash,
            "tokens_in": row.tokens_in,
            "tokens_out": row.tokens_out,
            "cost": row.cost,
            "latency_ms": row.latency_ms,
            "cache_hit": row.cache_hit,
            "status": row.status,
        }
    )
    return call_id


@dataclass
class LlmCallUsage:
    """Usage counters the wrapped call fills in before the row is written."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    cache_hit: bool = False


@contextmanager
def log_llm_call(
    session_or_engine: Session | Engine,
    *,
    run_id: int,
    model: str,
    purpose: str,
    prompt: str | None = None,
    prompt_hash: str | None = None,
) -> Iterator[LlmCallUsage]:
    """Wrap one model call: measure latency and write exactly one row.

    The body sets usage on the yielded :class:`LlmCallUsage`; on exit one row is
    written with the measured ``latency_ms``. If the body raises, the row is
    written with ``status='error'`` and the exception's text, then the exception
    is re-raised unchanged. A failure to write the row is logged and never masks
    the body's exception.
    """
    usage = LlmCallUsage()
    started = time.perf_counter()
    try:
        yield usage
    except BaseException as exc:
        try:
            record_llm_call(
                session_or_engine,
                run_id=run_id,
                model=model,
                purpose=purpose,
                prompt=prompt,
                prompt_hash=prompt_hash,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cost=usage.cost,
                latency_ms=_elapsed_ms(started),
                cache_hit=usage.cache_hit,
                status=ERROR,
                error=_error_text(exc),
            )
        except Exception:
            logger.exception("llm_call row could not be written after a failed call")
        raise
    else:
        record_llm_call(
            session_or_engine,
            run_id=run_id,
            model=model,
            purpose=purpose,
            prompt=prompt,
            prompt_hash=prompt_hash,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost=usage.cost,
            latency_ms=_elapsed_ms(started),
            cache_hit=usage.cache_hit,
        )
