"""Tests for the resumable stage runner (RT-01).

The suite runs offline against in-memory SQLite. The optional PostgreSQL test is
skipped unless ``DM_TEST_DATABASE_URL`` points at a disposable database.
"""

from __future__ import annotations

import os
import unittest
from collections.abc import Callable, Iterable, Iterator

import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from core.runs import (
    DONE,
    FAILED,
    FINISHED,
    NEEDS_REVIEW,
    PENDING,
    RUNNING,
    WORK_UNIT_STATUSES,
    Base,
    Run,
    RunSummary,
    StageRunner,
    WorkUnit,
    create_all,
)

POSTGRES_URL = os.environ.get("DM_TEST_DATABASE_URL")


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def unit_rows(engine: Engine, stage: str = "discover") -> dict[str, WorkUnit]:
    """Return the latest row per unit key (a key can have several input hashes)."""
    with sa.orm.Session(engine) as session:
        rows = session.scalars(
            sa.select(WorkUnit).where(WorkUnit.stage == stage).order_by(WorkUnit.id)
        ).all()
    latest: dict[str, WorkUnit] = {}
    for row in rows:
        latest[row.unit_key] = row
    return latest


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = memory_engine()
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)
        self.runner = StageRunner(self.engine)

    def run_stage(
        self,
        units: Iterable[tuple[str, str]],
        handler: Callable[[str], None] | None = None,
    ) -> RunSummary:
        if handler is None:
            handler = _noop
        return self.runner.run("discover", units, handler)

    def run_row(self, run_id: int) -> Run:
        with sa.orm.Session(self.engine) as session:
            run = session.get(Run, run_id)
        assert run is not None
        return run


def _noop(key: str) -> None:
    return None


class FreshRunTests(RunnerTestCase):
    def test_fresh_run_executes_every_unit_and_records_done(self) -> None:
        seen: list[str] = []

        summary = self.run_stage([("a", "h1"), ("b", "h1"), ("c", "h1")], seen.append)

        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(summary.status, FINISHED)
        self.assertEqual((summary.executed, summary.skipped, summary.failed), (3, 0, 0))
        rows = unit_rows(self.engine)
        self.assertEqual({row.status for row in rows.values()}, {DONE})
        run = self.run_row(summary.run_id)
        self.assertEqual(run.status, FINISHED)
        self.assertIsNotNone(run.finished_at)

    def test_empty_unit_set_finishes_the_run(self) -> None:
        summary = self.run_stage([])

        self.assertEqual(summary.status, FINISHED)
        self.assertEqual(summary.executed, 0)

    def test_units_are_consumed_lazily_not_materialised(self) -> None:
        handled: list[str] = []

        def units() -> Iterator[tuple[str, str]]:
            yield ("a", "h1")
            # If the runner built a list first, "a" would not be handled yet.
            self.assertEqual(handled, ["a"])
            yield ("b", "h1")

        summary = self.runner.run("discover", units(), handled.append)

        self.assertEqual(summary.executed, 2)
        self.assertEqual(handled, ["a", "b"])

    def test_each_unit_row_starts_with_one_attempt_and_a_timestamp(self) -> None:
        self.run_stage([("a", "h1"), ("b", "h1")])

        for row in unit_rows(self.engine).values():
            self.assertEqual(row.attempts, 1)
            self.assertIsNone(row.error)
            self.assertIsNotNone(row.updated_at)


class FailureTests(RunnerTestCase):
    def test_handler_failure_records_error_and_continues_with_other_units(self) -> None:
        def handler(key: str) -> None:
            if key == "b":
                raise ValueError("upstream exploded")

        summary = self.run_stage([("a", "h1"), ("b", "h1"), ("c", "h1")], handler)

        self.assertEqual(summary.status, FAILED)
        self.assertEqual((summary.executed, summary.skipped, summary.failed), (3, 0, 1))
        rows = unit_rows(self.engine)
        self.assertEqual(rows["a"].status, DONE)
        self.assertEqual(rows["b"].status, FAILED)
        self.assertIn("ValueError", rows["b"].error or "")
        self.assertIn("upstream exploded", rows["b"].error or "")
        self.assertEqual(rows["c"].status, DONE)
        self.assertEqual(self.run_row(summary.run_id).status, FAILED)

    def test_error_text_is_bounded(self) -> None:
        def handler(key: str) -> None:
            raise RuntimeError("x" * 10_000)

        self.run_stage([("a", "h1")], handler)

        error = unit_rows(self.engine)["a"].error or ""
        self.assertLessEqual(len(error), 2000)


class ResumeTests(RunnerTestCase):
    def test_second_run_with_same_inputs_executes_nothing(self) -> None:
        first = self.run_stage([("a", "h1"), ("b", "h1")])
        handled: list[str] = []

        second = self.run_stage([("a", "h1"), ("b", "h1")], handled.append)

        self.assertEqual(first.executed, 2)
        self.assertEqual(handled, [])
        self.assertEqual(second.status, FINISHED)
        self.assertEqual((second.executed, second.skipped), (0, 2))
        self.assertEqual(len(unit_rows(self.engine)), 2)

    def test_changed_input_hash_reexecutes_only_that_unit(self) -> None:
        self.run_stage([("a", "h1"), ("b", "h1")])
        handled: list[str] = []

        summary = self.run_stage([("a", "h1"), ("b", "h2")], handled.append)

        self.assertEqual(handled, ["b"])
        self.assertEqual((summary.executed, summary.skipped), (1, 1))
        with sa.orm.Session(self.engine) as session:
            hashes = session.scalars(
                sa.select(WorkUnit.input_hash).where(WorkUnit.unit_key == "b")
            ).all()
        self.assertEqual(sorted(hashes), ["h1", "h2"])

    def test_failed_units_are_retried_with_attempts_incremented(self) -> None:
        calls: list[str] = []

        def flaky(key: str) -> None:
            calls.append(key)
            if key == "b" and calls.count("b") == 1:
                raise TimeoutError("try again")

        first = self.run_stage([("a", "h1"), ("b", "h1")], flaky)
        second = self.run_stage([("a", "h1"), ("b", "h1")], flaky)

        self.assertEqual(first.status, FAILED)
        self.assertEqual(second.status, FINISHED)
        self.assertEqual((second.executed, second.skipped, second.failed), (1, 1, 0))
        rows = unit_rows(self.engine)
        self.assertEqual(rows["a"].attempts, 1)
        self.assertEqual(rows["b"].attempts, 2)
        self.assertEqual(rows["b"].status, DONE)
        self.assertIsNone(rows["b"].error)
        self.assertEqual(rows["b"].run_id, second.run_id)

    def test_needs_review_units_are_retried(self) -> None:
        # needs_review is a stage-body decision; the runner treats it as unfinished
        # work, so stage bodies must be idempotent.
        self.run_stage([("a", "h1")])
        with sa.orm.Session(self.engine) as session:
            unit = session.scalar(sa.select(WorkUnit).where(WorkUnit.unit_key == "a"))
            assert unit is not None
            unit.status = NEEDS_REVIEW
            session.commit()

        handled: list[str] = []
        self.run_stage([("a", "h1")], handled.append)

        self.assertEqual(handled, ["a"])
        self.assertEqual(unit_rows(self.engine)["a"].status, DONE)

    def test_unit_keys_are_scoped_by_stage(self) -> None:
        self.run_stage([("a", "h1")])
        handled: list[str] = []

        self.runner.run("index", [("a", "h1")], handled.append)

        self.assertEqual(handled, ["a"])
        self.assertEqual(len(unit_rows(self.engine)), 1)
        self.assertEqual(len(unit_rows(self.engine, "index")), 1)


class SchemaTests(RunnerTestCase):
    def test_tables_and_unique_constraint_exist(self) -> None:
        inspector = sa.inspect(self.engine)
        self.assertIn("run", inspector.get_table_names())
        self.assertIn("work_unit", inspector.get_table_names())
        uniques = {
            tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints("work_unit")
        }
        self.assertIn(("stage", "unit_key", "input_hash"), uniques)

    def test_status_sets_match_the_architecture(self) -> None:
        self.assertEqual(set(WORK_UNIT_STATUSES), {PENDING, DONE, FAILED, NEEDS_REVIEW})
        self.assertEqual(RUNNING, "running")


@unittest.skipUnless(POSTGRES_URL, "DM_TEST_DATABASE_URL is not set; PostgreSQL path not exercised")
class PostgresRunnerTests(unittest.TestCase):
    """Optional: proves the same schema and runner work on PostgreSQL 16.

    ``DM_TEST_DATABASE_URL`` must name a disposable database: the test drops and
    recreates the ``run`` and ``work_unit`` tables in it.
    """

    def setUp(self) -> None:
        assert POSTGRES_URL is not None
        self.engine = create_engine(POSTGRES_URL)
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as connection:
            Base.metadata.drop_all(connection)
        create_all(self.engine)
        self.addCleanup(self._drop_tables)

    def _drop_tables(self) -> None:
        with self.engine.begin() as connection:
            Base.metadata.drop_all(connection)

    def test_run_resume_round_trip(self) -> None:
        runner = StageRunner(self.engine)
        first_handled: list[str] = []
        units = [("a", "h1"), ("b", "h1")]

        first = runner.run("round-trip", units, first_handled.append)
        second_handled: list[str] = []
        second = runner.run("round-trip", units, second_handled.append)
        third_handled: list[str] = []
        third = runner.run("round-trip", [("a", "h1"), ("b", "h2")], third_handled.append)

        self.assertEqual(first_handled, ["a", "b"])
        self.assertEqual((first.executed, first.skipped), (2, 0))
        self.assertEqual(second_handled, [])
        self.assertEqual((second.executed, second.skipped), (0, 2))
        self.assertEqual(third_handled, ["b"])
        self.assertEqual((third.executed, third.skipped), (1, 1))
        self.assertEqual(first.status, FINISHED)

    def test_failure_is_recorded_with_error(self) -> None:
        def handler(key: str) -> None:
            raise RuntimeError("postgres boom")

        summary = StageRunner(self.engine).run("round-trip", [("a", "h1")], handler)

        self.assertEqual(summary.status, FAILED)
        row = unit_rows(self.engine, "round-trip")["a"]
        self.assertEqual(row.status, FAILED)
        self.assertIn("postgres boom", row.error or "")


if __name__ == "__main__":
    unittest.main()
