"""Tests for entity-driven ``landscape crawl`` (CR-04); offline fakes only.

A local ``http.server`` plays Firecrawl's ``/v1/scrape`` API only; ``robots.txt`` is a patched
``core.cli.get_robots`` fake (no page URL is fetched directly: the fetcher is Firecrawl). The
database is a temp SQLite file and the raw store a temp directory. Entity hosts use
``example.com``/``.invalid`` only (D-003).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import sqlalchemy as sa
import yaml
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
ROBOTS_DISALLOW_BLOCKED = b"User-agent: *\nDisallow: /blocked\n"


class _State:
    def __init__(self) -> None:
        self.scrapes: list[str] = []
        self.pages: dict[str, str] = {}  # URL -> rawHtml; others serve HTML
        self.statuses: dict[str, int] = {}  # URL -> metadata.statusCode; default 200


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
                    "data": {
                        "rawHtml": state.pages.get(url, HTML),
                        "metadata": {
                            "statusCode": state.statuses.get(url, 200),
                            "url": url,
                        },
                    },
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def fake_robots(url: str, *, user_agent: str = "", timeout_s: float = 0.0) -> tuple[int, bytes]:
    """Hosts named ``blocked.*`` disallow everything; other hosts disallow only ``/blocked``."""
    host = urlsplit(url).hostname or ""
    if host.startswith("blocked."):
        return 200, ROBOTS_DISALLOW_ALL
    return 200, ROBOTS_DISALLOW_BLOCKED


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
        # CR-04 behaviour: a pack without a `crawl` section crawls the homepage only.
        self.plain_scope = self.tmp / "plain-scope"
        self.plain_scope.mkdir()
        for name in ("scope.yaml", "rubric.yaml"):
            shutil.copy(Path(EXAMPLE_SCOPE) / name, self.plain_scope / name)
        scope_file = self.plain_scope / "scope.yaml"
        data = yaml.safe_load(scope_file.read_text(encoding="utf-8"))
        data.pop("crawl", None)
        scope_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

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

    def crawl(self, *extra: str, scope: str | None = None):
        return self.runner.invoke(app, ["crawl", "--scope", scope or str(self.plain_scope), *extra])

    def scope_with_crawl(self, body: str) -> str:
        """Copy the no-crawl pack into a fresh directory and append a ``crawl`` section."""
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        for name in ("scope.yaml", "rubric.yaml"):
            shutil.copy(self.plain_scope / name, root / name)
        path = root / "scope.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\n" + body, encoding="utf-8")
        return str(root)

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

    def test_no_crawl_section_fetches_homepage_only(self) -> None:
        a_id = self.seed([(SCOPE_ID, "a.example.com")])[0]
        url = "https://a.example.com/"
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [url])
        self.assertEqual(self.snapshots(), [(a_id, url)])

    def test_example_policy_fetches_homepage_then_priority_paths(self) -> None:
        a_id, b_id = self.seed([(SCOPE_ID, "a.example.com"), (SCOPE_ID, "b.example.com")])
        result = self.crawl(scope=EXAMPLE_SCOPE)
        self.assertEqual(result.exit_code, 0, result.output)
        expected = [
            "https://a.example.com/",
            "https://a.example.com/about",
            "https://a.example.com/services",
            "https://b.example.com/",
            "https://b.example.com/about",
            "https://b.example.com/services",
        ]
        self.assertEqual(self.state.scrapes, expected)
        self.assertEqual(
            self.snapshots(),
            [
                (a_id, expected[0]),
                (a_id, expected[1]),
                (a_id, expected[2]),
                (b_id, expected[3]),
                (b_id, expected[4]),
                (b_id, expected[5]),
            ],
        )
        self.assertEqual(len(result.stdout.splitlines()), len(expected), result.stdout)

    def test_policy_is_truncated_at_max_pages(self) -> None:
        scope = self.scope_with_crawl("crawl:\n  max_pages: 2\n  priority_paths: [/a, /b]\n")
        a_id = self.seed([(SCOPE_ID, "a.example.com")])[0]
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        expected = ["https://a.example.com/", "https://a.example.com/a"]
        self.assertEqual(self.state.scrapes, expected)
        self.assertEqual(self.snapshots(), [(a_id, expected[0]), (a_id, expected[1])])

    def test_limit_counts_entities_not_pages(self) -> None:
        a_id, b_id = self.seed([(SCOPE_ID, "a.example.com"), (SCOPE_ID, "b.example.com")])
        result = self.crawl("--limit", "1", scope=EXAMPLE_SCOPE)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            self.state.scrapes,
            [
                "https://a.example.com/",
                "https://a.example.com/about",
                "https://a.example.com/services",
            ],
        )
        self.assertTrue(all(entity_id == a_id for entity_id, _ in self.snapshots()))
        self.assertNotEqual(a_id, b_id)

    def test_robots_disallow_on_one_priority_path_continues(self) -> None:
        scope = self.scope_with_crawl(
            "crawl:\n  max_pages: 3\n  priority_paths: [/blocked, /after]\n"
        )
        a_id = self.seed([(SCOPE_ID, "a.example.com")])[0]
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("skipped https://a.example.com/blocked robots", result.output)
        expected = ["https://a.example.com/", "https://a.example.com/after"]
        self.assertEqual(self.state.scrapes, expected)
        self.assertEqual(self.snapshots(), [(a_id, expected[0]), (a_id, expected[1])])

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

    # --- CR-06: discover_links fills the remaining per-entity budget with same-host links ---

    HOME_LINKS = (
        '<a href="/b">b</a><a href="https://other.invalid/x">x</a><a href="/a">a</a>'
        '<a href="/c#frag">c</a><a href="/d.pdf">d</a><a href="/b">b again</a>'
    )

    def page(self, url: str, body: str) -> None:
        self.state.pages[url] = f"<html><body><h1>P</h1><p>Text.</p>{body}</body></html>"

    def test_discovered_links_fill_budget_in_document_order(self) -> None:
        (eid,) = self.seed([(SCOPE_ID, "alpha.example.com")])
        self.page("https://alpha.example.com/", self.HOME_LINKS)
        scope = self.scope_with_crawl(
            "crawl:\n  max_pages: 4\n  priority_paths: [/a]\n  discover_links: true\n"
        )
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        expected = [f"https://alpha.example.com{p}" for p in ("/", "/a", "/b", "/c")]
        self.assertEqual(self.state.scrapes, expected)
        self.assertEqual(self.snapshots(), [(eid, url) for url in expected])

    def test_scheme_and_trailing_slash_variants_fetched_once(self) -> None:
        self.seed([(SCOPE_ID, "alpha.example.com")])
        self.page(
            "https://alpha.example.com/",
            '<a href="http://alpha.example.com/a">a</a><a href="https://alpha.example.com/a/">a</a>'
            '<a href="/a">a</a><a href="/b">b</a>',
        )
        scope = self.scope_with_crawl("crawl:\n  max_pages: 4\n  discover_links: true\n")
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            self.state.scrapes,
            [
                "https://alpha.example.com/",
                "http://alpha.example.com/a",
                "https://alpha.example.com/b",
            ],
        )

    def test_discover_false_ignores_links(self) -> None:
        self.seed([(SCOPE_ID, "alpha.example.com")])
        self.page("https://alpha.example.com/", self.HOME_LINKS)
        scope = self.scope_with_crawl("crawl:\n  max_pages: 4\n  discover_links: false\n")
        self.crawl(scope=scope)
        self.assertEqual(self.state.scrapes, ["https://alpha.example.com/"])

    def test_links_from_blocked_or_failed_pages_are_not_used(self) -> None:
        self.seed([(SCOPE_ID, "alpha.example.com"), (SCOPE_ID, "broken.example.com")])
        self.page("https://alpha.example.com/", '<a href="/blocked">x</a><a href="/next">n</a>')
        self.page("https://alpha.example.com/blocked", '<a href="/hidden">h</a>')
        self.page("https://broken.example.com/", '<a href="/never">n</a>')
        scope = self.scope_with_crawl("crawl:\n  max_pages: 5\n  discover_links: true\n")
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("skipped https://alpha.example.com/blocked robots", result.output)
        self.assertEqual(
            self.state.scrapes,
            [
                "https://alpha.example.com/",
                "https://alpha.example.com/next",
                "https://broken.example.com/",
            ],
        )

    def test_budget_is_per_entity_and_limit_counts_entities(self) -> None:
        self.seed([(SCOPE_ID, f"{n}.example.com") for n in ("a1", "a2", "a3")])
        for n in ("a1", "a2", "a3"):
            self.page(f"https://{n}.example.com/", '<a href="/x">x</a><a href="/y">y</a>')
        scope = self.scope_with_crawl("crawl:\n  max_pages: 2\n  discover_links: true\n")
        result = self.crawl("--limit", "2", scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            self.state.scrapes,
            [
                "https://a1.example.com/",
                "https://a1.example.com/x",
                "https://a2.example.com/",
                "https://a2.example.com/x",
            ],
        )

    def test_never_fetches_a_url_twice_and_follows_breadth_first(self) -> None:
        self.seed([(SCOPE_ID, "alpha.example.com")])
        self.page("https://alpha.example.com/", '<a href="/p">p</a><a href="/q">q</a>')
        self.page(
            "https://alpha.example.com/p", '<a href="/">h</a><a href="/q">q</a><a href="/r">r</a>'
        )
        scope = self.scope_with_crawl(
            "crawl:\n  max_pages: 10\n  priority_paths: [/p]\n  discover_links: true\n"
        )
        self.crawl(scope=scope)
        self.assertEqual(
            self.state.scrapes,
            [f"https://alpha.example.com{p}" for p in ("/", "/p", "/q", "/r")],
        )

    # --- CR-07: a non-2xx fetch is skipped, writes nothing and is not a failure ---

    def test_non_2xx_priority_path_prints_skipped_and_snapshots_only_homepage(self) -> None:
        (eid,) = self.seed([(SCOPE_ID, "a.example.com")])
        home, missing = "https://a.example.com/", "https://a.example.com/a"
        self.state.statuses[missing] = 404
        scope = self.scope_with_crawl("crawl:\n  max_pages: 2\n  priority_paths: [/a]\n")
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [home, missing])
        self.assertEqual(self.snapshots(), [(eid, home)])
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2, result.stdout)
        self.assertTrue(lines[0].startswith(f"ok {home} snapshot="), lines[0])
        self.assertEqual(lines[1], f"skipped {missing} status=404")

    def test_crawl_continues_after_non_2xx_to_the_next_url(self) -> None:
        (eid,) = self.seed([(SCOPE_ID, "a.example.com")])
        home, failing, ok = (
            "https://a.example.com/",
            "https://a.example.com/a",
            "https://a.example.com/b",
        )
        self.state.statuses[failing] = 500
        scope = self.scope_with_crawl("crawl:\n  max_pages: 3\n  priority_paths: [/a, /b]\n")
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [home, failing, ok])
        self.assertEqual(self.snapshots(), [(eid, home), (eid, ok)])
        self.assertIn(f"skipped {failing} status=500", result.output)
        self.assertIn(f"ok {ok}", result.output)

    def test_500_homepage_is_skipped_without_failing_the_run(self) -> None:
        failing_id, ok_id = self.seed([(SCOPE_ID, "a.example.com"), (SCOPE_ID, "b.example.com")])
        failing, ok = "https://a.example.com/", "https://b.example.com/"
        self.state.statuses[failing] = 500
        result = self.crawl()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [failing, ok])
        self.assertEqual(self.snapshots(), [(ok_id, ok)])
        self.assertIn(f"skipped {failing} status=500", result.output)
        self.assertNotEqual(failing_id, ok_id)

    def test_links_from_a_non_2xx_page_are_not_discovered(self) -> None:
        (eid,) = self.seed([(SCOPE_ID, "alpha.example.com")])
        home, missing, hidden = (
            "https://alpha.example.com/",
            "https://alpha.example.com/missing",
            "https://alpha.example.com/hidden",
        )
        self.page(home, '<a href="/missing">m</a>')
        self.page(missing, '<a href="/hidden">h</a>')
        self.state.statuses[missing] = 404
        scope = self.scope_with_crawl("crawl:\n  max_pages: 5\n  discover_links: true\n")
        result = self.crawl(scope=scope)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.state.scrapes, [home, missing])
        self.assertEqual(self.snapshots(), [(eid, home)])
        self.assertIn(f"skipped {missing} status=404", result.output)
        self.assertNotIn(hidden, self.state.scrapes)


if __name__ == "__main__":
    unittest.main()
