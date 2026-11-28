"""CR-19: ``landscape crawl`` follows a robots.txt redirect through the fake Firecrawl backend.

Offline: :mod:`adapters.fake_firecrawl_server` on 127.0.0.1 plays both the Firecrawl API and the
crawled site, whose ``/robots.txt`` answers 301/302 (or a redirect loop).
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from adapters.fake_firecrawl_server import make_server
from app import cli
from app.cli import app

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


class CrawlRobotsRedirectTest(unittest.TestCase):
    def start(self, mode: str) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        server = make_server(port=0, robots_redirect=mode)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        clean.update(
            {
                "FIRECRAWL_URL": base + "/",
                "DM_DATABASE_URL": f"sqlite+pysqlite:///{root / 'crawl.sqlite'}",
                "RAW_STORE_DIR": str(root / "raw"),
            }
        )
        for patcher in (
            mock.patch.dict(os.environ, clean, clear=True),
            mock.patch.object(cli, "CRAWL_MIN_INTERVAL_S", 0.0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(root)  # no stray .env
        self.addCleanup(os.chdir, cwd)
        return base

    def crawl(self, *urls: str):
        args = ["crawl", "--scope", EXAMPLE_SCOPE]
        for url in urls:
            args += ["--url", url]
        return CliRunner().invoke(app, args)

    def test_redirected_robots_allows_pages(self) -> None:
        for mode in ("301", "302"):
            with self.subTest(mode=mode):
                base = self.start(mode)
                result = self.crawl(base + "/a", base + "/b")
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(f"ok {base}/a snapshot=", result.output)
                self.assertIn(f"ok {base}/b snapshot=", result.output)
                self.assertNotIn("robots", result.output)

    def test_robots_redirect_loop_skips_pages(self) -> None:
        base = self.start("loop")
        result = self.crawl(base + "/a")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(f"skipped {base}/a robots", result.output)
        self.assertNotIn("ok ", result.output)
