"""Tests for ``landscape web-search`` (DS-01); offline, urlopen is faked."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from typer.testing import CliRunner

from adapters.searxng import SearxngSearch
from app.cli import app

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
BASE = "http://search.invalid"


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _body(n: int) -> bytes:
    return json.dumps(
        {
            "results": [
                {"url": f"https://example.org/{i}", "title": f"Title {i}", "content": "x"}
                for i in range(1, n + 1)
            ]
        }
    ).encode()


class _CliFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clean = {
            k: v for k, v in os.environ.items() if not k.startswith("DM_") and k != "SEARXNG_URL"
        }
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.urls: list[str] = []

    def fake_urlopen(self, n: int = 3):
        def opener(request, timeout):  # noqa: ANN001
            self.urls.append(request.full_url)
            return _Resp(_body(n))

        return mock.patch("urllib.request.urlopen", side_effect=opener)

    def invoke(self, *args: str):
        return self.runner.invoke(app, ["web-search", "--scope", EXAMPLE_SCOPE, *args])


class WebSearchCliTest(_CliFixture):
    def test_prints_rank_url_title(self) -> None:
        os.environ["SEARXNG_URL"] = BASE
        with self.fake_urlopen(3):
            result = self.invoke("--query", "web crawler")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            result.stdout.splitlines(),
            [f"{i}\thttps://example.org/{i}\tTitle {i}" for i in (1, 2, 3)],
        )

    def test_limit_language_site_passed_through(self) -> None:
        os.environ["SEARXNG_URL"] = BASE
        with self.fake_urlopen(10):
            result = self.invoke(
                "--query", "crawler", "--limit", "2", "--language", "fi", "--site", "example.org"
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        params = parse_qs(urlsplit(self.urls[0]).query)
        self.assertEqual(params["q"], ["crawler site:example.org"])
        self.assertEqual(params["language"], ["fi"])

    def test_default_limit_is_ten(self) -> None:
        os.environ["SEARXNG_URL"] = BASE
        with self.fake_urlopen(20):
            result = self.invoke("--query", "crawler")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(result.stdout.splitlines()), 10)

    def test_unset_env_exits_2_without_network(self) -> None:
        for value in (None, "  "):
            if value is not None:
                os.environ["SEARXNG_URL"] = value
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
                result = self.invoke("--query", "crawler")
            self.assertEqual(result.exit_code, 2, result.output)
            self.assertIn("SEARXNG_URL", result.output)

    def test_search_error_exits_1_without_url_leak(self) -> None:
        os.environ["SEARXNG_URL"] = BASE
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom " + BASE)):
            result = self.invoke("--query", "secret query")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("error SearchError: transport failed: OSError", result.stderr)
        self.assertNotIn("search.invalid", result.output)
        self.assertNotIn("secret query", result.output)

    def test_bad_query_and_limit_exit_2(self) -> None:
        os.environ["SEARXNG_URL"] = BASE
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
            for args in (
                ("--query", "  "),
                ("--query", "x", "--limit", "0"),
                ("--query", "x", "--limit", "101"),
            ):
                result = self.invoke(*args)
                self.assertEqual(result.exit_code, 2, (args, result.output))


class FromEnvTest(unittest.TestCase):
    def test_from_env_reads_variable(self) -> None:
        with mock.patch.dict(os.environ, {"SEARXNG_URL": BASE + "/"}):
            self.assertIsInstance(SearxngSearch.from_env(), SearxngSearch)

    def test_from_env_missing_names_variable(self) -> None:
        with mock.patch.dict(os.environ, {"SEARXNG_URL": ""}):
            with self.assertRaisesRegex(ValueError, "SEARXNG_URL"):
                SearxngSearch.from_env()

    def test_from_env_invalid_does_not_echo_value(self) -> None:
        with mock.patch.dict(os.environ, {"SEARXNG_URL": "ftp://hidden.invalid"}):
            with self.assertRaises(ValueError) as ctx:
                SearxngSearch.from_env()
        self.assertNotIn("hidden", str(ctx.exception))
        self.assertIn("SEARXNG_URL", str(ctx.exception))


class WebSearchSaveTest(_CliFixture):
    """``--save`` persists results as ``candidate`` rows (DS-02); SQLite under a temp dir."""

    def setUp(self) -> None:
        super().setUp()
        os.environ["SEARXNG_URL"] = BASE
        self.db_url = f"sqlite:///{os.getcwd()}/dm.sqlite"
        os.environ["DM_DATABASE_URL"] = self.db_url

    def rows(self):  # noqa: ANN201
        import sqlalchemy as sa
        from sqlalchemy.orm import Session

        from core.candidate_store import Candidate

        engine = sa.create_engine(self.db_url)
        try:
            if not sa.inspect(engine).has_table("candidate"):
                return []
            with Session(engine) as s:
                return list(s.scalars(sa.select(Candidate).order_by(Candidate.id)))
        finally:
            engine.dispose()

    def test_save_writes_one_row_per_result(self) -> None:
        from datetime import UTC

        with self.fake_urlopen(3):
            result = self.invoke("--query", "web crawler", "--save")
        self.assertEqual(result.exit_code, 0, result.output)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:3], [f"{i}\thttps://example.org/{i}\tTitle {i}" for i in (1, 2, 3)])
        self.assertEqual(lines[3:], ["saved 3 candidates"])
        rows = self.rows()
        self.assertEqual([r.url for r in rows], [f"https://example.org/{i}" for i in (1, 2, 3)])
        self.assertEqual([r.name for r in rows], ["Title 1", "Title 2", "Title 3"])
        for r in rows:
            self.assertEqual((r.channel, r.query), ("search", "web crawler"))
            self.assertEqual(r.found_at.tzinfo, UTC)
            self.assertIsNone(r.country_hint)
            self.assertIsNone(r.source_id)

    def test_save_zero_results(self) -> None:
        with self.fake_urlopen(0):
            result = self.invoke("--query", "nothing", "--save")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines(), ["saved 0 candidates"])

    def test_save_db_failure_exits_2_without_url(self) -> None:
        os.environ["DM_DATABASE_URL"] = "postgresql+psycopg://user:hunter2@db.invalid:1/x"
        with (
            self.fake_urlopen(2),
            mock.patch("app.cli.create_db_engine", side_effect=RuntimeError("db.invalid hunter2")),
        ):
            result = self.invoke("--query", "crawler", "--save")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (RuntimeError)", result.stderr)
        self.assertNotIn("db.invalid", result.output)
        self.assertNotIn("hunter2", result.output)

    def test_search_error_with_save_writes_nothing(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            result = self.invoke("--query", "crawler", "--save")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(self.rows(), [])

    def test_without_save_no_engine_created(self) -> None:
        with (
            self.fake_urlopen(2),
            mock.patch("app.cli.create_db_engine", side_effect=AssertionError("no engine")),
        ):
            result = self.invoke("--query", "crawler")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("saved", result.stdout)
