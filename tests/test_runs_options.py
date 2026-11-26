"""Direct tests for StageRunner's ``run_id=`` and ``catch=`` options (RT-01, 99/A1).

These exercise the runner without any CLI caller, on in-memory SQLite.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterator

import sqlalchemy as sa

from core.runs import DONE, FAILED, FINISHED, RUNNING, Run, StageRunner, create_all
from tests.test_runs import memory_engine, unit_rows

STAGE = "discover"


class BudgetError(Exception):
    """Neutral stand-in for an error a caller wants to stop the run on."""


class RunnerOptionsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = memory_engine()
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)
        self.runner = StageRunner(self.engine)

    def open_caller_run(self) -> int:
        with sa.orm.Session(self.engine) as session, session.begin():
            run = Run(stage=STAGE, status=RUNNING)
            session.add(run)
            session.flush()
            return run.id

    def run_row(self, run_id: int) -> Run:
        with sa.orm.Session(self.engine) as session:
            run = session.get(Run, run_id)
        assert run is not None
        return run

    def only_run(self) -> Run:
        with sa.orm.Session(self.engine) as session:
            runs = session.scalars(sa.select(Run)).all()
        self.assertEqual(len(runs), 1)
        return runs[0]

    # (a) attach mode
    def test_attach_mode_leaves_caller_run_open_and_tags_units(self) -> None:
        run_id = self.open_caller_run()
        summary = self.runner.run(STAGE, [("a", "h"), ("b", "h")], lambda k: None, run_id=run_id)
        self.assertEqual(summary.run_id, run_id)
        self.assertEqual(summary.status, FINISHED)
        run = self.run_row(run_id)
        self.assertEqual(run.status, RUNNING)
        self.assertIsNone(run.finished_at)
        rows = unit_rows(self.engine)
        self.assertEqual({r.run_id for r in rows.values()}, {run_id})
        self.assertEqual({r.status for r in rows.values()}, {DONE})
        with sa.orm.Session(self.engine) as session:
            self.assertEqual(session.scalar(sa.select(sa.func.count()).select_from(Run)), 1)

    def test_attach_mode_summary_failed_when_a_unit_fails(self) -> None:
        run_id = self.open_caller_run()

        def handler(key: str) -> None:
            if key == "b":
                raise ValueError("boom")

        summary = self.runner.run(STAGE, [("a", "h"), ("b", "h")], handler, run_id=run_id)
        self.assertEqual(summary.status, FAILED)
        self.assertEqual(summary.failed, 1)
        run = self.run_row(run_id)
        self.assertEqual(run.status, RUNNING)
        self.assertIsNone(run.finished_at)

    # (b) catch narrowing
    def test_caught_exception_is_recorded_and_run_continues(self) -> None:
        seen: list[str] = []

        def handler(key: str) -> None:
            seen.append(key)
            if key == "a":
                raise ValueError("bad input")

        summary = self.runner.run(STAGE, [("a", "h"), ("b", "h")], handler, catch=(ValueError,))
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual((summary.executed, summary.failed, summary.status), (2, 1, FAILED))
        rows = unit_rows(self.engine)
        self.assertEqual(rows["a"].status, FAILED)
        self.assertEqual(rows["a"].attempts, 1)
        self.assertIn("ValueError: bad input", rows["a"].error or "")
        self.assertEqual(rows["b"].status, DONE)

    def test_uncaught_exception_fails_unit_closes_owned_run_and_propagates(self) -> None:
        seen: list[str] = []

        def handler(key: str) -> None:
            seen.append(key)
            raise BudgetError("over budget")

        with self.assertRaises(BudgetError):
            self.runner.run(STAGE, [("a", "h"), ("b", "h")], handler, catch=(ValueError,))
        self.assertEqual(seen, ["a"])
        rows = unit_rows(self.engine)
        self.assertEqual(rows["a"].status, FAILED)
        self.assertIn("BudgetError", rows["a"].error or "")
        self.assertNotIn("b", rows)
        run = self.only_run()
        self.assertEqual(run.status, FAILED)
        self.assertIsNotNone(run.finished_at)

    def test_uncaught_exception_in_attach_mode_leaves_caller_run_open(self) -> None:
        run_id = self.open_caller_run()

        def handler(key: str) -> None:
            raise BudgetError("over budget")

        with self.assertRaises(BudgetError):
            self.runner.run(STAGE, [("a", "h")], handler, run_id=run_id, catch=(ValueError,))
        self.assertEqual(unit_rows(self.engine)["a"].status, FAILED)
        run = self.run_row(run_id)
        self.assertEqual(run.status, RUNNING)
        self.assertIsNone(run.finished_at)

    # (c) crash paths
    def test_keyboard_interrupt_closes_owned_run_failed_and_propagates(self) -> None:
        def handler(key: str) -> None:
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.runner.run(STAGE, [("a", "h")], handler)
        run = self.only_run()
        self.assertEqual(run.status, FAILED)
        self.assertIsNotNone(run.finished_at)

    def test_iterator_error_closes_owned_run_failed_and_propagates(self) -> None:
        def units() -> Iterator[tuple[str, str]]:
            yield ("a", "h")
            raise RuntimeError("source broke")

        with self.assertRaises(RuntimeError):
            self.runner.run(STAGE, units(), lambda k: None)
        self.assertEqual(unit_rows(self.engine)["a"].status, DONE)
        run = self.only_run()
        self.assertEqual(run.status, FAILED)
        self.assertIsNotNone(run.finished_at)

    # (d) retry and skip
    def test_retry_increments_attempts_and_skips_unchanged_done_unit(self) -> None:
        fail = {"a"}

        def handler(key: str) -> None:
            if key in fail:
                raise ValueError("transient")

        first = self.runner.run(STAGE, [("a", "h"), ("b", "h")], handler)
        self.assertEqual(first.failed, 1)
        fail.clear()
        second = self.runner.run(STAGE, [("a", "h"), ("b", "h")], handler)
        self.assertEqual((second.executed, second.skipped, second.status), (1, 1, FINISHED))
        rows = unit_rows(self.engine)
        self.assertEqual((rows["a"].status, rows["a"].attempts), (DONE, 2))
        self.assertIsNone(rows["a"].error)
        self.assertEqual((rows["b"].status, rows["b"].attempts), (DONE, 1))
        self.assertEqual(rows["a"].run_id, second.run_id)
        self.assertEqual(rows["b"].run_id, first.run_id)


if __name__ == "__main__":
    unittest.main()
