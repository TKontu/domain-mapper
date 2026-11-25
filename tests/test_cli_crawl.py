"""Tests for the CLI ``crawl`` stage over explicit URLs (CL-01).

Offline: a local ``http.server`` on 127.0.0.1 plays both the Firecrawl ``/v1/scrape`` API and the
crawled site's ``robots.txt``; the database is a SQLite file and the raw store a temp directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import cli
from core.chunk_store import ChunkRow
from core.cli import app
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
HTML = "<html><head><title>T</title></head><body><h1>Guide</h1><p>Widgets here.</p></body></html>"
ROBOTS = b"User-agent: *\nDisallow: /private\n"


class _State:
    scrapes: list[str]
    robots_hits: int

    def __init__(self) -> None:
        self.scrapes = []
        self.robots_hits = 0


def _handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silence
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/robots.txt":
                state.robots_hits += 1
                self._send(200, ROBOTS, "text/plain")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            url = request["url"]
            state.scrapes.append(url)
            if url.endswith("/broken"):
                payload = {"success": False, "error": "scrape failed"}
            else:
                payload = {
                    "success": True,
                    "data": {"rawHtml": HTML, "metadata": {"statusCode": 200, "url": url}},
                }
            self._send(200, json.dumps(payload).encode(), "application/json")

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class CrawlStageTest(unittest.TestCase):
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
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        interval = mock.patch.object(cli, "CRAWL_MIN_INTERVAL_S", 0.0)
        interval.start()
        self.addCleanup(interval.stop)
        cwd = os.getcwd()
        os.chdir(self.tmp)  # no stray .env
        self.addCleanup(os.chdir, cwd)

    def crawl(self, *urls: str):
        args = ["crawl", "--scope", EXAMPLE_SCOPE]
        for url in urls:
            args += ["--url", url]
        return self.runner.invoke(app, args)

    def rows(self) -> tuple[list[SnapshotRow], list[ChunkRow]]:
        engine = sa.create_engine(f"sqlite+pysqlite:///{self.db}")
        with Session(engine) as s:
            snaps = list(s.scalars(sa.select(SnapshotRow).order_by(SnapshotRow.id)))
            chunks = list(s.scalars(sa.select(ChunkRow)))
        engine.dispose()
        return snaps, chunks

    def test_success_writes_snapshot_chunks_and_blob(self) -> None:
        url = self.base + "/page"
        result = self.crawl(url)
        self.assertEqual(result.exit_code, 0, result.output)
        snaps, chunks = self.rows()
        self.assertEqual(len(snaps), 1)
        self.assertGreater(len(chunks), 0)
        digest = hashlib.sha256(HTML.encode()).hexdigest()
        self.assertEqual(snaps[0].sha256, digest)
        self.assertTrue((self.raw / digest[:2] / digest[2:4] / digest).is_file())
        self.assertIn(
            f"ok {url} snapshot={snaps[0].id} chunks={len(chunks)} raw={digest}", result.output
        )
        self.assertEqual(self.state.robots_hits, 1)

    def test_robots_disallow_skips_without_fetch(self) -> None:
        url = self.base + "/private/x"
        result = self.crawl(url)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(f"skipped {url} robots", result.output)
        self.assertEqual(self.state.scrapes, [])
        self.assertFalse(self.db.exists() and self.rows()[0])

    def test_fetch_error_on_one_url_others_proceed(self) -> None:
        bad, good = self.base + "/broken", self.base + "/page"
        result = self.crawl(bad, good)
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn(f"error {bad} FetchError", result.output)
        self.assertIn(f"ok {good}", result.output)
        self.assertEqual(len(self.rows()[0]), 1)
        self.assertNotIn(self.base + "/v1", result.output)

    def test_missing_firecrawl_url_exits_2(self) -> None:
        del os.environ["FIRECRAWL_URL"]
        result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("FIRECRAWL_URL", result.output)
        self.assertEqual(self.state.scrapes, [])

    def test_no_url_exits_2(self) -> None:
        result = self.crawl()
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("--url", result.output)

    def test_env_urls_never_echoed(self) -> None:
        secret_base = "http://firecrawl-secret.invalid:9"
        os.environ["FIRECRAWL_URL"] = secret_base
        os.environ["DM_DATABASE_URL"] = f"sqlite+pysqlite:///{self.db}"
        result = self.crawl(self.base + "/page")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertNotIn("firecrawl-secret", result.output)
        self.assertNotIn(str(self.db), result.output)

    def _count_waits(self) -> list[str]:
        waits: list[str] = []
        real = cli.HostRateLimiter.wait

        def counting(limiter: object, url: str) -> None:
            waits.append(url)
            real(limiter, url)  # type: ignore[arg-type]

        patcher = mock.patch.object(cli.HostRateLimiter, "wait", counting)
        patcher.start()
        self.addCleanup(patcher.stop)
        return waits

    def test_limiter_waits_once_per_real_request(self) -> None:
        waits = self._count_waits()
        first, second = self.base + "/page", self.base + "/other"
        result = self.crawl(first, second)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.robots_hits, 1)
        # first URL: robots GET + page fetch; second (robots cached): page fetch only
        self.assertEqual(len(waits), 3)

    def test_robots_disallowed_cached_url_never_waits(self) -> None:
        waits = self._count_waits()
        result = self.crawl(self.base + "/page", self.base + "/private/x")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(waits), 2)

    def test_dry_run_unchanged(self) -> None:
        result = self.runner.invoke(app, ["crawl", "--scope", EXAMPLE_SCOPE, "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("stage: crawl", result.output)
        self.assertEqual(self.state.scrapes, [])


if __name__ == "__main__":
    unittest.main()
