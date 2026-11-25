"""Tests for entity-driven ``landscape crawl`` (CR-04); offline fakes only.

A local ``http.server`` plays Firecrawl's ``/v1/scrape`` API only; ``robots.txt`` is a patched
``core.cli.get_robots`` fake (no page URL is fetched directly: the fetcher is Firecrawl). The
database is a temp SQLite file and the raw store a temp directory. Entity hosts use
``example.com``/``.invalid`` only (D-003).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.cli import app
from core.entity_store import STATUS_UNSCREENED, Entity
from core.runs import create_all
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
SCOPE_ID = "example"
HTML = "<html><head><title>T</title></head><body><h1>Guide</h1><p>Widgets here.</p></body></html>"
ROBOTS_DISALLOW_ALL = b"User-agent: *\nDisallow: /\n"


class _State:
    def __init__(self) -> None:
        self.scrapes: list[str] = []


def _handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silence
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", "0"))
            url = json.loads(self.rfile.read(length))["url"]
            state.scrapes.append(url)
            host = urlsplit(url).hostname or ""
            if host.startswith("broken."):
                payload = {"success": False, "error": "scrape failed"}
            else:
                payload = {
                    "success": True,
                    "data": {"rawHtml": HTML, "metadata": {"statusCode": 200, "url": url}},
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def fake_robots(url: str, *, user_agent: str = "", timeout_s: float = 0.0) -> tuple[int, bytes]:
    """404 (allow all) for most hosts; hosts named ``blocked.*`` disallow everything."""
    host = urlsplit(url).hostname or ""
    if host.startswith("blocked."):
        return 200, ROBOTS_DISALLOW_ALL
    return 404, b""


class EntityCrawlCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.state = _State()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self.state))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.db = self.tmp / "crawl.sqlite"
        self.raw = self.tmp / "raw"
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        clean.update(
            {
                "FIRECRAWL_URL": self.base + "/",
                "DM_DATABASE_URL": f"sqlite+pysqlite:///{self.db}",
                "RAW_STORE_DIR": str(self.raw),
            }
        )
        env = mock.patch.dict(os.environ, clean, clear=True)
        env.start()
        self.addCleanup(env.stop)
        interval = mock.patch.object(cli, "CRAWL_MIN_INTERVAL_S", 0.0)
        interval.start()
        self.addCleanup(interval.stop)
        robots = mock.patch.object(cli, "get_robots", fake_robots)
        robots.start()
        self.addCleanup(robots.stop)
        cwd = os.getcwd()
        os.chdir(self.tmp)  # no stray .env
        self.addCleanup(os.chdir, cwd)

    def seed(self, rows: list[tuple[str, str]]) -> list[int]:
        """Insert ``(scope_id, primary_domain)`` entities in order; return their ids."""
        engine = sa.create_engine(f"sqlite+pysqlite:///{self.db}")
        ids: list[int] = []
        try:
            create_all(engine)
            with Session(engine) as session, session.begin():
                for scope_id, domain in rows:
                    entity = Entity(
                        scope_id=scope_id,
                        name=domain,
                        primary_domain=domain,
                        status=STATUS_UNSCREENED,
                    )
                    session.add(entity)
                    session.flush()
                    ids.append(entity.id)
        finally:
            engine.dispose()
        return ids

    def crawl(self, *extra: str, scope: str = EXAMPLE_SCOPE):
        return self.runner.invoke(app, ["crawl", "--scope", scope, *extra])

    def snapshots(self) -> list[tuple[int | None, str]]:
        engine = sa.create_engine(f"sqlite+pysqlite:///{self.db}")
        try:
            with Session(engine) as session:
                return [
                    (row.entity_id, row.url)
                    for row in session.scalars(sa.select(SnapshotRow).order_by(SnapshotRow.id))
                ]
        finally:
            engine.dispose()

    def test_crawls_scope_entities_in_id_order_with_entity_ids(self) -> None:
        a_id, t_id, b_id = self.seed(
            [
                (SCOPE_ID, "a.example.com"),
                ("other", "t.example.com"),
                (SCOPE_ID, "b.example.com"),
            ]
        )
        a_url, b_url = "https://a.example.com/", "https://b.example.com/"
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [a_url, b_url])
        self.assertEqual(self.snapshots(), [(a_id, a_url), (b_id, b_url)])
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2, result.stdout)
        self.assertTrue(lines[0].startswith(f"ok {a_url} snapshot="), lines[0])
        self.assertTrue(lines[1].startswith(f"ok {b_url} snapshot="), lines[1])
        self.assertNotIn("t.example.com", result.stdout)

    def test_limit_fetches_at_most_n(self) -> None:
        a_id, b_id = self.seed([(SCOPE_ID, "a.example.com"), (SCOPE_ID, "b.example.com")])
        a_url = "https://a.example.com/"
        result = self.crawl("--limit", "1")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [a_url])
        self.assertEqual(self.snapshots(), [(a_id, a_url)])
        self.assertNotEqual(a_id, b_id)

    def test_robots_disallow_prints_skipped_and_fetches_nothing(self) -> None:
        self.seed([(SCOPE_ID, "blocked.example.invalid")])
        url = "https://blocked.example.invalid/"
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(f"skipped {url} robots", result.output)
        self.assertEqual(self.state.scrapes, [])
        self.assertEqual(self.snapshots(), [])

    def test_fetch_error_on_one_entity_continues_and_exits_1(self) -> None:
        broken_id, ok_id = self.seed(
            [(SCOPE_ID, "broken.example.com"), (SCOPE_ID, "a.example.com")]
        )
        broken_url, ok_url = "https://broken.example.com/", "https://a.example.com/"
        result = self.crawl()
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertTrue(
            any(
                line.startswith(f"error {broken_url} FetchError")
                for line in result.output.splitlines()
            ),
            result.output,
        )
        self.assertIn(f"ok {ok_url}", result.output)
        self.assertEqual(self.state.scrapes, [broken_url, ok_url])
        self.assertEqual(self.snapshots(), [(ok_id, ok_url)])
        self.assertNotEqual(broken_id, ok_id)

    def test_zero_entities_prints_entities_0_exits_0_without_setup(self) -> None:
        del os.environ["FIRECRAWL_URL"]  # nothing to crawl needs no fetcher
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines(), ["entities 0"])
        self.assertEqual(self.state.scrapes, [])

    def test_scope_pack_error_exits_1_like_resolve_apply(self) -> None:
        broken = self.tmp / "broken-scope"
        broken.mkdir()
        for name in ("scope.yaml", "rubric.yaml"):
            (broken / name).write_text(
                (Path(EXAMPLE_SCOPE) / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        (broken / "scope.yaml").write_text(
            (broken / "scope.yaml").read_text(encoding="utf-8") + "\nsurprise: 1\n",
            encoding="utf-8",
        )
        result = self.crawl(scope=str(broken))
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("scope.yaml", result.stderr)
        self.assertEqual(self.state.scrapes, [])

    def test_db_setup_failure_exits_2_without_echoing_url(self) -> None:
        os.environ["DM_DATABASE_URL"] = "postgresql+psycopg://user:hunter2@db.invalid:1/x"
        with mock.patch("core.cli.create_engine", side_effect=RuntimeError("db.invalid hunter2")):
            result = self.crawl()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (RuntimeError)", result.stderr)
        self.assertNotIn("db.invalid", result.output)
        self.assertNotIn("hunter2", result.output)

    def test_entity_read_failure_exits_2_without_echoing_url(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        error = sa.exc.OperationalError("select 1", {}, Exception("db.invalid hunter2"))
        with mock.patch.object(cli, "iter_entities", side_effect=error):
            result = self.crawl()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (OperationalError)", result.stderr)
        self.assertNotIn("db.invalid", result.output)
        self.assertNotIn("hunter2", result.output)

    def test_url_mode_ignores_limit_and_writes_null_entity_id(self) -> None:
        self.seed([(SCOPE_ID, "a.example.com")])
        url = "https://c.example.com/"
        result = self.crawl("--url", url, "--limit", "1")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [url])
        self.assertEqual(self.snapshots(), [(None, url)])

    def test_crawl_help_mentions_entity_crawl(self) -> None:
        result = self.runner.invoke(app, ["crawl", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("entity", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
