"""Idempotent, resumable stage runs over a ``work_unit`` table (RT-01).

Every stage body runs through :class:`StageRunner`: it opens a ``run`` row, then
processes units one at a time, committing each unit's status in its own short
transaction. A unit that is already ``done`` for the same input hash is skipped,
so re-running a stage (including after a crash) only touches unfinished work.

Its first production caller is ``landscape screen`` (app/cli.py ``run_screen``),
which attaches it to the invocation's own run row (``run_id=``) and stops the
whole run on a budget error (``catch=``). The runner does not own the handler's
side effects; handlers must be idempotent because a unit can be retried after a
partial failure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Run statuses. A run starts ``running`` and ends ``finished`` or ``failed``.
RUNNING: Final = "running"
FINISHED: Final = "finished"
FAILED: Final = "failed"

# Work-unit statuses (ARCHITECTURE.md §1). The runner writes pending/done/failed;
# needs_review is reserved for stage bodies that hand a unit to human review.
PENDING: Final = "pending"
DONE: Final = "done"
NEEDS_REVIEW: Final = "needs_review"

WORK_UNIT_STATUSES: Final[tuple[str, ...]] = (PENDING, DONE, FAILED, NEEDS_REVIEW)

# Handler errors are stored for operators; cap the column value so a huge
# traceback cannot bloat the row (the full traceback still reaches the logs).
MAX_ERROR_CHARS: Final = 2000


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the run-tracking tables."""


class Run(Base):
    __tablename__ = "run"
    __table_args__ = (
        sa.CheckConstraint("status IN ('running', 'finished', 'failed')", name="ck_run_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    stage: Mapped[str] = mapped_column(sa.String(64), index=True)
    status: Mapped[str] = mapped_column(sa.String(16), default=RUNNING)
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))


class WorkUnit(Base):
    __tablename__ = "work_unit"
    __table_args__ = (
        sa.UniqueConstraint("stage", "unit_key", "input_hash", name="uq_work_unit_stage_key_hash"),
        sa.Index("ix_work_unit_stage_status", "stage", "status"),
        sa.CheckConstraint(
            "status IN ('pending', 'done', 'failed', 'needs_review')",
            name="ck_work_unit_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Latest run that touched the unit; the unit row survives across runs so a
    # retry keeps its attempt count.
    run_id: Mapped[int] = mapped_column(sa.ForeignKey("run.id"), index=True)
    stage: Mapped[str] = mapped_column(sa.String(64))
    unit_key: Mapped[str] = mapped_column(sa.String(512))
    status: Mapped[str] = mapped_column(sa.String(16), default=PENDING)
    attempts: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(sa.Text)
    input_hash: Mapped[str] = mapped_column(sa.String(64))
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


@dataclass(frozen=True)
class RunSummary:
    """Counts from one runner call; the units themselves stay in the database."""

    run_id: int
    status: str
    executed: int = 0
    skipped: int = 0
    failed: int = 0


def create_db_engine(database_url: str) -> Engine:
    """Create an engine for ``database_url`` (PostgreSQL in production)."""
    return create_engine(database_url)


def create_all(engine: Engine) -> None:
    """Create the run-tracking tables if they are absent.

    Schema migrations are a separate item; this helper exists so tests and a
    fresh database can start without one (recorded residual in RT-01).
    """
    Base.metadata.create_all(engine)


class StageRunner:
    """Run a stage idempotently and resumably over an iterable of work units.

    ``units`` yields ``(unit_key, input_hash)`` pairs. They are consumed lazily,
    one at a time, so a stage never materialises the whole unit set in memory.
    ``handler`` is called with the unit key; an exception matching ``catch``
    (default: any ``Exception``) is recorded on the unit and the run continues.
    Any other ``Exception`` is recorded ``failed`` on the unit, the run is closed
    ``failed`` (when the runner owns it) and the exception propagates.

    With ``run_id`` the runner attaches to a run row the caller opened and owns:
    it neither opens nor closes the run, and the summary status only reflects
    unit failures. Each unit's claim (attempt increment) and outcome are
    committed in two short transactions; the handler runs between them, outside
    any runner transaction, so it may write through its own connections.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def run(
        self,
        stage: str,
        units: Iterable[tuple[str, str]],
        handler: Callable[[str], None],
        *,
        run_id: int | None = None,
        catch: tuple[type[Exception], ...] = (Exception,),
    ) -> RunSummary:
        owns_run = run_id is None
        if run_id is None:
            run_id = self._open_run(stage)
        executed = skipped = failed = 0
        try:
            for unit_key, input_hash in units:
                outcome = self._process_unit(run_id, stage, unit_key, input_hash, handler, catch)
                if outcome is None:
                    skipped += 1
                    continue
                executed += 1
                if outcome == FAILED:
                    failed += 1
        except BaseException:
            # The iterator or an uncaught handler error stopped the run; do not
            # leave an owned run ``running``.
            if owns_run:
                self._close_run(run_id, FAILED)
            raise

        status = FAILED if failed else FINISHED
        if owns_run:
            self._close_run(run_id, status)
        return RunSummary(
            run_id=run_id,
            status=status,
            executed=executed,
            skipped=skipped,
            failed=failed,
        )

    def _open_run(self, stage: str) -> int:
        with self._session_factory.begin() as session:
            run = Run(stage=stage, status=RUNNING, started_at=_utcnow())
            session.add(run)
            session.flush()
            return run.id

    def _close_run(self, run_id: int, status: str) -> None:
        with self._session_factory.begin() as session:
            run = session.get(Run, run_id)
            if run is None:  # pragma: no cover - the run row was just created
                raise RuntimeError(f"run {run_id} disappeared")
            run.status = status
            run.finished_at = _utcnow()

    def _process_unit(
        self,
        run_id: int,
        stage: str,
        unit_key: str,
        input_hash: str,
        handler: Callable[[str], None],
        catch: tuple[type[Exception], ...] = (Exception,),
    ) -> str | None:
        """Claim, run and record one unit; ``None`` means already done."""
        with self._session_factory.begin() as session:
            unit = session.scalar(
                sa.select(WorkUnit).where(
                    WorkUnit.stage == stage,
                    WorkUnit.unit_key == unit_key,
                    WorkUnit.input_hash == input_hash,
                )
            )
            if unit is not None and unit.status == DONE:
                return None

            if unit is None:
                unit = WorkUnit(
                    run_id=run_id,
                    stage=stage,
                    unit_key=unit_key,
                    status=PENDING,
                    attempts=0,
                    input_hash=input_hash,
                )
                session.add(unit)
            unit.run_id = run_id
            unit.attempts += 1
            unit.status = PENDING
            unit.error = None
            unit.updated_at = _utcnow()
            session.flush()
            unit_id = unit.id

        status, error, stop = DONE, None, None
        try:
            handler(unit_key)
        except catch as exc:
            status, error = FAILED, _error_text(exc)
        except Exception as exc:
            status, error, stop = FAILED, _error_text(exc), exc

        with self._session_factory.begin() as session:
            row = session.get(WorkUnit, unit_id)
            if row is None:  # pragma: no cover - the row was just committed
                raise RuntimeError(f"work unit {unit_id} disappeared")
            row.status = status
            row.error = error
            row.updated_at = _utcnow()
        if stop is not None:
            raise stop
        return status


def _error_text(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return text[: MAX_ERROR_CHARS - 1] + "…"
