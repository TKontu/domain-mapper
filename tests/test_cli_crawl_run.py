"""Tests that ``landscape crawl`` records one ``crawl`` run row per invocation (CR-14).

Offline: reuses the loopback fake Firecrawl servers and SQLite setup of the URL and entity crawl
CLI tests; neutral fixtures only (D-003).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core import cli
from core.runs import Run
from tests import test_cli_crawl as url_tests  # module import: its tests are not re-collected
from tests import test_cli_crawl_entities as entity_tests

SCOPE_ID = entity_tests.SCOPE_ID


def _assert_db_unavailable_line(test: unittest.TestCase, result: object) -> None:
    """One stderr line names the close failure; the database URL never appears in the output."""
    test.assertEqual(result.stderr.count("error: database unavailable"), 1)
    test.assertIn("error: database unavailable (OperationalError)", result.stderr)
    test.assertNotIn(str(test.db), result.output)


def _runs(db: object) -> list[tuple[str, str, bool]]:
    engine = sa.create_engine(f"sqlite+pysqlite:///{db}")
    try:
        with Session(engine) as session:
            return [
                (r.stage, r.status, r.finished_at is not None)
                for r in session.scalars(sa.select(Run).order_by(Run.id))
            ]
    except sa.exc.OperationalError:  # no schema: nothing was opened
        return []
    finally:
        engine.dispose()


class _Boom:
    """A fetcher whose every fetch raises ``exc``."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def fetch(self, url: str, *, render_js: bool = False) -> object:
        raise self.exc


class UrlCrawlRunTest(unittest.TestCase):
    setUp = url_tests.CrawlStageTest.setUp
    crawl = url_tests.CrawlStageTest.crawl

    def test_all_ok_closes_one_finished_crawl_run(self) -> None:
        result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "finished", True)])

    def test_failed_url_closes_run_failed_and_exits_1(self) -> None:
        result = self.crawl(self.base + "/broken", self.base + "/page")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_fetcher_runtime_error_is_a_target_failure(self) -> None:
        # _Crawler reports any Exception per target, so it fails the run with exit 1.
        with mock.patch.object(cli, "_make_fetcher", return_value=_Boom(RuntimeError("x"))):
            result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_unexpected_runtime_error_closes_run_failed_and_propagates(self) -> None:
        with mock.patch.object(cli._Crawler, "crawl", side_effect=RuntimeError("boom")):
            result = self.crawl(self.base + "/page")
        self.assertIsInstance(result.exception, RuntimeError)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_interrupt_from_fetcher_closes_run_failed_and_propagates(self) -> None:
        with mock.patch.object(cli, "_make_fetcher", return_value=_Boom(KeyboardInterrupt())):
            result = self.crawl(self.base + "/page")
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_interrupt_with_close_db_error_propagates_original(self) -> None:  # CR-17
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        with mock.patch.object(cli, "_make_fetcher", return_value=_Boom(KeyboardInterrupt())):
            with mock.patch.object(cli, "_close_run", side_effect=err):
                result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 130, result.output)
        _assert_db_unavailable_line(self, result)

    def test_runtime_error_with_close_db_error_propagates_original(self) -> None:  # CR-17
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        with mock.patch.object(cli._Crawler, "crawl", side_effect=RuntimeError("boom")):
            with mock.patch.object(cli, "_close_run", side_effect=err):
                result = self.crawl(self.base + "/page")
        self.assertIsInstance(result.exception, RuntimeError)
        _assert_db_unavailable_line(self, result)

    def test_missing_fetcher_config_exits_2_without_run_row(self) -> None:
        del os.environ["FIRECRAWL_URL"]
        result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertEqual(_runs(self.db), [])

    def _assert_db_exit_2(self, result: object) -> None:
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (OperationalError)", result.stderr)
        self.assertNotIn(str(self.db), result.output)
        self.assertNotIn("sqlite", result.output)

    def test_open_run_db_error_exits_2(self) -> None:  # CR-16
        err = sa.exc.OperationalError("INSERT", {}, Exception("gone"))
        with mock.patch.object(cli, "_open_run", side_effect=err):
            result = self.crawl(self.base + "/page")
        self._assert_db_exit_2(result)
        self.assertEqual(_runs(self.db), [])

    def test_close_run_db_error_exits_2_and_retries_close_failed(self) -> None:  # CR-16
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        real = cli._close_run
        calls: list[str] = []

        def close(engine: object, run_id: int, status: str) -> None:
            calls.append(status)
            if len(calls) == 1:  # the final close fails; the best-effort close succeeds
                raise err
            real(engine, run_id, status)

        with mock.patch.object(cli, "_close_run", side_effect=close):
            result = self.crawl(self.base + "/page")
        self._assert_db_exit_2(result)
        self.assertEqual(calls, ["finished", "failed"])
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_close_run_db_error_when_close_impossible_still_exits_2(self) -> None:  # CR-16
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        with mock.patch.object(cli, "_close_run", side_effect=err):
            result = self.crawl(self.base + "/page")
        self._assert_db_exit_2(result)
        self.assertEqual(_runs(self.db), [("crawl", "running", False)])

    def test_target_write_db_error_closes_run_failed_and_exits_2(self) -> None:  # CR-16
        err = sa.exc.OperationalError("INSERT", {}, Exception("gone"))
        with mock.patch.object(cli._Crawler, "crawl", side_effect=err):
            result = self.crawl(self.base + "/page")
        self._assert_db_exit_2(result)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_dry_run_opens_no_run_row(self) -> None:
        result = self.runner.invoke(
            cli.app,
            ["crawl", "--scope", url_tests.EXAMPLE_SCOPE, "--dry-run", "--url", self.base + "/"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_runs(self.db), [])


class EntityCrawlRunTest(unittest.TestCase):
    setUp = entity_tests.EntityCrawlCliTest.setUp
    seed = entity_tests.EntityCrawlCliTest.seed
    crawl = entity_tests.EntityCrawlCliTest.crawl

    def test_all_ok_closes_one_finished_crawl_run(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "finished", True)])

    def test_failed_entity_closes_run_failed_and_exits_1(self) -> None:
        self.seed([(SCOPE_ID, "broken.example.com"), (SCOPE_ID, "a.example.com")])
        result = self.crawl()
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_unexpected_runtime_error_closes_run_failed_and_propagates(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        with mock.patch.object(cli._Crawler, "crawl", side_effect=RuntimeError("boom")):
            result = self.crawl()
        self.assertIsInstance(result.exception, RuntimeError)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_interrupt_from_fetcher_closes_run_failed(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        with mock.patch.object(cli, "_make_fetcher", return_value=_Boom(KeyboardInterrupt())):
            result = self.crawl()
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_interrupt_with_close_db_error_propagates_original(self) -> None:  # CR-17
        self.seed([(SCOPE_ID, "a.example.com")])
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        with mock.patch.object(cli, "_make_fetcher", return_value=_Boom(KeyboardInterrupt())):
            with mock.patch.object(cli, "_close_run", side_effect=err):
                result = self.crawl()
        self.assertEqual(result.exit_code, 130, result.output)
        _assert_db_unavailable_line(self, result)

    def test_runtime_error_with_close_db_error_propagates_original(self) -> None:  # CR-17
        self.seed([(SCOPE_ID, "a.example.com")])
        err = sa.exc.OperationalError("UPDATE", {}, Exception("gone"))
        with mock.patch.object(cli._Crawler, "crawl", side_effect=RuntimeError("boom")):
            with mock.patch.object(cli, "_close_run", side_effect=err):
                result = self.crawl()
        self.assertIsInstance(result.exception, RuntimeError)
        _assert_db_unavailable_line(self, result)

    def test_page_read_failure_closes_run_failed_and_exits_2(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        err = sa.exc.OperationalError("SELECT", {}, Exception("gone"))
        with mock.patch.object(cli._Crawler, "crawl", side_effect=err):
            result = self.crawl()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertEqual(_runs(self.db), [("crawl", "failed", True)])

    def test_missing_fetcher_config_exits_2_without_run_row(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        del os.environ["FIRECRAWL_URL"]
        result = self.crawl()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertEqual(_runs(self.db), [])

    def test_zero_entities_opens_no_run_row(self) -> None:
        self.seed([])
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_runs(self.db), [])
