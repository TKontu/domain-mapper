"""Tests for the one-retry transient-fetch policy of ``landscape crawl`` (CR-09).

Offline fakes only: a scripted fetcher, a patched allow-all robots fetch, an in-memory SQLite
database and a temp raw store. Hosts use ``example.com`` (D-003).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.fetcher import FetchError, FetchResult, TransientFetchError
from core import cli
from core.cli import app
from core.runs import create_all
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
URL = "https://www.example.com/"
HTML = b"<html><body><h1>Guide</h1><p>Widgets here.</p></body></html>"
ROBOTS = b"User-agent: *\nAllow: /\n"


class ScriptedFetcher:
    """Return one scripted outcome per call; an exception outcome is raised."""

    def __init__(self, outcomes: Iterable[FetchResult | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def fetch(self, url: str, *, render_js: bool = False) -> FetchResult:
        self.calls.append(url)
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok_result(url: str = URL, status: int = 200) -> FetchResult:
    return FetchResult(
        final_url=url,
        status=status,
        content_type="text/html; charset=utf-8",
        body_bytes=HTML,
    )


def fake_robots(url: str, *, user_agent: str = "", timeout_s: float = 0.0) -> tuple[int, bytes]:
    return 200, ROBOTS


@pytest.fixture()
def engine() -> Iterable[sa.Engine]:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def offline_crawl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAW_STORE_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(cli, "CRAWL_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(cli, "get_robots", fake_robots)


def snapshots(engine: sa.Engine) -> list[SnapshotRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(SnapshotRow)))


def crawler(engine: sa.Engine, fetcher: ScriptedFetcher, sleeps: list[float]) -> cli._Crawler:
    return cli._Crawler(engine, fetcher, sleep=sleeps.append)


def test_transient_once_then_success_retries_ingests_and_does_not_fail(
    engine: sa.Engine, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[str] = []
    real_wait = cli.HostRateLimiter.wait

    def counting_wait(limiter: object, url: str) -> None:
        waits.append(url)
        real_wait(limiter, url)  # type: ignore[arg-type]

    monkeypatch.setattr(cli.HostRateLimiter, "wait", counting_wait)
    fetcher = ScriptedFetcher([TransientFetchError("HTTP 503"), ok_result()])
    sleeps: list[float] = []
    crawl = crawler(engine, fetcher, sleeps)

    result = crawl.crawl(URL, None)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"retry {URL} TransientFetchError: HTTP 503"
    assert lines[1].startswith(f"ok {URL} snapshot=")
    assert len(lines) == 2
    assert fetcher.calls == [URL, URL]
    assert sleeps == [cli.CRAWL_RETRY_BACKOFF_S]
    # robots GET, first page fetch, then one fresh limiter wait before the retry fetch.
    assert waits == ["https://www.example.com/robots.txt", URL, URL]
    assert crawl.failed is False
    assert result is not None
    assert len(snapshots(engine)) == 1


def test_transient_twice_reports_error_and_fails_the_run(
    engine: sa.Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    fetcher = ScriptedFetcher(
        [TransientFetchError("HTTP 429"), TransientFetchError("transport failed: TimeoutError")]
    )
    sleeps: list[float] = []
    crawl = crawler(engine, fetcher, sleeps)

    result = crawl.crawl(URL, None)

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"retry {URL} TransientFetchError: HTTP 429",
        f"error {URL} TransientFetchError: transport failed: TimeoutError",
    ]
    assert len(fetcher.calls) == 2
    assert sleeps == [cli.CRAWL_RETRY_BACKOFF_S]
    assert crawl.failed is True
    assert result is None
    assert snapshots(engine) == []


def test_non_transient_fetch_error_is_not_retried(
    engine: sa.Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    fetcher = ScriptedFetcher([FetchError("site blocked scraping")])
    sleeps: list[float] = []
    crawl = crawler(engine, fetcher, sleeps)

    result = crawl.crawl(URL, None)

    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"error {URL} FetchError: site blocked scraping"]
    assert fetcher.calls == [URL]
    assert sleeps == []
    assert crawl.failed is True
    assert result is None


def test_retry_success_goes_through_non_2xx_skip_handler(
    engine: sa.Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    fetcher = ScriptedFetcher([TransientFetchError("HTTP 500"), ok_result(status=404)])
    sleeps: list[float] = []
    crawl = crawler(engine, fetcher, sleeps)

    assert crawl.crawl(URL, None) is None

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"retry {URL} TransientFetchError: HTTP 500",
        f"skipped {URL} status=404",
    ]
    assert len(fetcher.calls) == 2
    assert crawl.failed is False
    assert snapshots(engine) == []


def test_cli_run_exits_0_after_a_successful_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production caller ``run_crawl`` sees no failure when the retry succeeds."""
    db = tmp_path / "crawl.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", f"sqlite+pysqlite:///{db}")
    # A zero backoff keeps the real ``time.sleep`` call instant while still exercising it.
    monkeypatch.setattr(cli, "CRAWL_RETRY_BACKOFF_S", 0.0)
    fetcher = ScriptedFetcher([TransientFetchError("HTTP 503"), ok_result()])
    monkeypatch.setattr(cli, "_make_fetcher", lambda: fetcher)
    monkeypatch.chdir(tmp_path)  # no stray .env
    runner = CliRunner()

    result = runner.invoke(app, ["crawl", "--scope", EXAMPLE_SCOPE, "--url", URL])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"retry {URL} TransientFetchError: HTTP 503"
    assert lines[1].startswith(f"ok {URL} snapshot=")
    assert fetcher.calls == [URL, URL]
    check = sa.create_engine(f"sqlite+pysqlite:///{db}")
    try:
        assert len(snapshots(check)) == 1
    finally:
        check.dispose()
