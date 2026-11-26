"""Tests for ``landscape resolve`` (RV-01); SQLite under a temp dir, offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core.candidate_store import Candidate
from app.cli import app
from core.runs import create_all

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
FOUND_AT = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


class _CliFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(tmp.name)  # no stray .env
        self.addCleanup(os.chdir, cwd)
        self.db_url = f"sqlite:///{os.getcwd()}/dm.sqlite"
        os.environ["DM_DATABASE_URL"] = self.db_url

    def invoke(self, *args: str):
        return self.runner.invoke(app, ["resolve", "--scope", EXAMPLE_SCOPE, *args])

    def seed(self, rows: list[tuple[str, str, int | None]]) -> None:
        engine = sa.create_engine(self.db_url)
        try:
            create_all(engine)
            with Session(engine) as session, session.begin():
                for name, url, resolved in rows:
                    session.add(
                        Candidate(
                            name=name,
                            url=url,
                            channel="search",
                            query=None,
                            found_at=FOUND_AT,
                            resolved_entity_id=resolved,
                        )
                    )
        finally:
            engine.dispose()

    def rows(self) -> list[tuple[int, str, str, str, str | None, int | None]]:
        engine = sa.create_engine(self.db_url)
        try:
            with Session(engine) as session:
                return [
                    (r.id, r.name, r.url, r.channel, r.query, r.resolved_entity_id)
                    for r in session.scalars(sa.select(Candidate).order_by(Candidate.id))
                ]
        finally:
            engine.dispose()


class ResolveCliTest(_CliFixture):
    def test_prints_groups_sorted_then_summary(self) -> None:
        self.seed(
            [
                ("A first", "https://a.example.com/1", None),
                ("B", "https://b.example.com/", None),
                ("A second", "https://www.a.example.com/2", None),
                ("C", "https://c.example.com/", None),
            ]
        )
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "a.example.com\t2\tA first",
                "b.example.com\t1\tB",
                "c.example.com\t1\tC",
                "domains 3 candidates 4",
            ],
        )

    def test_resolved_rows_excluded(self) -> None:
        self.seed(
            [
                ("kept", "https://a.example.com/", None),
                ("resolved", "https://b.example.com/", 9),
            ]
        )
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            result.stdout.splitlines(),
            ["a.example.com\t1\tkept", "domains 1 candidates 1"],
        )

    def test_no_skipped_line_when_every_url_normalises(self) -> None:
        self.seed([("a", "https://a.example.com/", None)])
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("skipped", result.stdout)

    def test_skipped_line_trails_output(self) -> None:
        self.seed(
            [
                ("a", "https://a.example.com/", None),
                ("bad", "not a url", None),
                ("bad2", "ftp://files.example.com/x", None),
            ]
        )
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            result.stdout.splitlines(),
            ["a.example.com\t1\ta", "domains 1 candidates 1", "skipped 2"],
        )

    def test_empty_table_prints_zero_counts(self) -> None:
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines(), ["domains 0 candidates 0"])

    def test_missing_scope_exits_2(self) -> None:
        result = self.runner.invoke(app, ["resolve", "--scope", "no-such-scope"])
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("scope directory not found", result.stderr)

    def test_db_failure_exits_2_without_url(self) -> None:
        os.environ["DM_DATABASE_URL"] = "postgresql+psycopg://user:hunter2@db.invalid:1/x"
        with mock.patch("app.cli.create_db_engine", side_effect=RuntimeError("db.invalid hunter2")):
            result = self.invoke()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (RuntimeError)", result.stderr)
        self.assertNotIn("db.invalid", result.output)
        self.assertNotIn("hunter2", result.output)

    def test_rows_unchanged(self) -> None:
        self.seed(
            [
                ("a", "https://a.example.com/", None),
                ("b", "https://b.example.com/", 4),
                ("bad", "not a url", None),
            ]
        )
        before = self.rows()
        for _ in range(2):
            result = self.invoke()
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(self.rows(), before)
