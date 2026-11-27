"""RT-01: ``landscape resolve --apply`` runs through StageRunner (assignment 134/A1); SQLite."""

from __future__ import annotations

from unittest import mock

import sqlalchemy as sa

from core.runs import Run, WorkUnit
from tests.test_cli_resolve import _CliFixture

ROWS: list[tuple[str, str, int | None]] = [
    ("Acme", "https://acme.example/", None),
    ("Beta", "https://beta.example/", None),
]


class ResolveRunsTest(_CliFixture):
    def runs(self) -> list[tuple[str, str]]:
        engine = sa.create_engine(self.db_url)
        try:
            with engine.connect() as conn:
                return [tuple(r) for r in conn.execute(sa.select(Run.stage, Run.status))]
        finally:
            engine.dispose()

    def units(self) -> list[tuple[str, str, str, int]]:
        engine = sa.create_engine(self.db_url)
        try:
            with engine.connect() as conn:
                return [
                    tuple(r)
                    for r in conn.execute(
                        sa.select(
                            WorkUnit.stage, WorkUnit.unit_key, WorkUnit.status, WorkUnit.attempts
                        )
                    )
                ]
        finally:
            engine.dispose()

    def test_first_apply_one_finished_run_and_done_unit(self) -> None:
        self.seed(ROWS)
        result = self.invoke("--apply")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.splitlines(), ["created 2 linked 2 skipped 0"])
        self.assertEqual(self.runs(), [("resolve", "finished")])
        self.assertEqual(self.units(), [("resolve", "example", "done", 1)])

    def test_rerun_unchanged_skips_unit(self) -> None:
        self.seed(ROWS)
        self.invoke("--apply")
        with mock.patch("app.cli.apply_resolution") as applied:
            result = self.invoke("--apply")
        applied.assert_not_called()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.splitlines(), ["created 0 linked 0 skipped 0", "skipped 1"])
        self.assertEqual(self.runs(), [("resolve", "finished"), ("resolve", "finished")])
        self.assertEqual(self.units(), [("resolve", "example", "done", 1)])

    def test_new_candidate_reruns_unit(self) -> None:
        self.seed(ROWS)
        self.invoke("--apply")
        self.seed([("Gamma", "https://gamma.example/", None)])
        result = self.invoke("--apply")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.splitlines(), ["created 1 linked 1 skipped 0"])
        done = [u for u in self.units() if u[2] == "done"]
        self.assertEqual(len(done), 2)

    def test_handler_error_fails_unit_and_run_without_url(self) -> None:
        self.seed(ROWS)
        with mock.patch("app.cli.apply_resolution", side_effect=RuntimeError(self.db_url)):
            result = self.invoke("--apply")
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn(self.db_url, result.output)
        self.assertIn("error: database unavailable (RuntimeError)", result.output)
        self.assertEqual(self.runs(), [("resolve", "failed")])
        self.assertEqual(self.units(), [("resolve", "example", "failed", 1)])

    def test_plain_resolve_creates_no_run_rows(self) -> None:
        self.seed(ROWS)
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.runs(), [])
        self.assertEqual(self.units(), [])
