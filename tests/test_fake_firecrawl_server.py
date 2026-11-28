"""Tests for the local fake Firecrawl backend (CR-11). Loopback sockets only, no network."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

import re

from adapters.crawl_policy import RobotsPolicy
from adapters.fake_firecrawl_server import FakeFirecrawlServer, main, make_server
from adapters.fetcher import TransientFetchError
from adapters.firecrawl import FirecrawlFetcher
from adapters.http_transport import get_robots, make_post_transport

StartServer = Callable[..., tuple[FakeFirecrawlServer, str]]
ROBOTS = b"User-agent: *\nAllow: /\n"


@pytest.fixture()
def start() -> Iterator[StartServer]:
    """Start servers on port 0; shut every one down after the test."""
    servers: list[FakeFirecrawlServer] = []

    def _start(
        *,
        fail_first: int | None = None,
        crawl_delay: float | None = None,
        robots_redirect: str | None = None,
    ) -> tuple[FakeFirecrawlServer, str]:
        server = make_server(
            port=0, fail_first=fail_first, crawl_delay=crawl_delay, robots_redirect=robots_redirect
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    yield _start
    for server in servers:
        server.shutdown()
        server.server_close()


def post(base: str, url: str) -> tuple[int, bytes]:
    transport = make_post_transport(max_bytes=1024 * 1024, timeout_s=5)
    return transport(
        base + "/v1/scrape", json.dumps({"url": url}).encode(), {"Content-Type": "application/json"}
    )


def fetcher(base: str) -> FirecrawlFetcher:
    return FirecrawlFetcher(base, make_post_transport(max_bytes=1024 * 1024, timeout_s=5))


def test_robots_txt_allows_everything(start: StartServer) -> None:
    _, base = start()
    assert get_robots(base + "/robots.txt", user_agent="test-agent", timeout_s=5) == (200, ROBOTS, None)


def test_scrape_root_returns_heading_and_same_host_links(start: StartServer) -> None:
    _, base = start()
    result = fetcher(base).fetch(base + "/")
    assert result.status == 200
    assert result.final_url == base + "/"
    assert result.content_type == "text/html"
    assert b"<h1>/</h1>" in result.body_bytes
    assert f'href="{base}/a"'.encode() in result.body_bytes
    assert f'href="{base}/b"'.encode() in result.body_bytes


def test_scrape_subpage_has_its_path_as_heading_and_no_links(start: StartServer) -> None:
    _, base = start()
    result = fetcher(base).fetch(base + "/a")
    assert b"<h1>/a</h1>" in result.body_bytes
    assert b"<a " not in result.body_bytes


def test_links_use_the_requested_urls_host_not_the_backend(start: StartServer) -> None:
    _, base = start()
    result = fetcher(base).fetch("http://crawl-fixture.invalid/")
    assert result.final_url == "http://crawl-fixture.invalid/"
    assert b'href="http://crawl-fixture.invalid/a"' in result.body_bytes
    assert b'href="http://crawl-fixture.invalid/b"' in result.body_bytes


def test_fail_first_answers_the_status_then_succeeds(start: StartServer) -> None:
    _, base = start(fail_first=408)
    assert post(base, base + "/") == (408, b"")
    assert post(base, base + "/")[0] == 200


def test_fail_first_counts_each_distinct_url_separately(start: StartServer) -> None:
    _, base = start(fail_first=503)
    assert post(base, base + "/a")[0] == 503
    assert post(base, base + "/b")[0] == 503
    assert post(base, base + "/a")[0] == 200
    assert post(base, base + "/b")[0] == 200


def test_firecrawl_fetcher_retries_once_over_a_real_socket(start: StartServer) -> None:
    _, base = start(fail_first=408)
    crawl = fetcher(base)
    with pytest.raises(TransientFetchError, match="HTTP 408"):
        crawl.fetch(base + "/")
    result = crawl.fetch(base + "/")
    assert result.status == 200
    assert b"<h1>/</h1>" in result.body_bytes


def test_first_failure_is_counted_once_under_concurrency(start: StartServer) -> None:
    _, base = start(fail_first=503)
    transport = make_post_transport(max_bytes=1024 * 1024, timeout_s=5)
    body = json.dumps({"url": base + "/"}).encode()

    def call(_: int) -> int:
        return transport(base + "/v1/scrape", body, {"Content-Type": "application/json"})[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(8)))
    assert statuses.count(503) == 1
    assert statuses.count(200) == 7


def test_unknown_paths_and_methods_answer_404(start: StartServer) -> None:
    _, base = start()
    for path in ("/nope", "/v1/scrape"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + path, timeout=5)
        assert exc.value.code == 404
        exc.value.close()
    request = urllib.request.Request(base + "/v1/scrape", data=b"{}", method="PUT")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 404
    exc.value.close()


@pytest.mark.parametrize(
    "body", [b"", b"not json", b"[]", b'{"url": 3}', b'{"url": "file:///tmp/x"}']
)
def test_malformed_scrape_body_answers_400(start: StartServer, body: bytes) -> None:
    _, base = start()
    request = urllib.request.Request(base + "/v1/scrape", data=body, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 400
    exc.value.close()


@pytest.mark.parametrize("bad", [200, 399, 600, True])
def test_make_server_rejects_a_non_error_fail_first(bad: int) -> None:
    with pytest.raises(ValueError, match="400..599"):
        make_server(port=0, fail_first=bad)


def test_main_rejects_a_non_error_fail_first(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--fail-first", "200"])
    assert exc.value.code == 2
    assert "400..599" in capsys.readouterr().err


def test_root_links_a_its_trailing_slash_variant_and_b_in_order(start: StartServer) -> None:
    _, base = start()
    body = fetcher(base).fetch(base + "/").body_bytes.decode()
    assert re.findall(r'href="([^"]+)"', body) == [base + "/a", base + "/a/", base + "/b"]


def _robots_policy(base: str) -> RobotsPolicy:
    return RobotsPolicy(
        lambda robots_url: get_robots(robots_url, user_agent="test-agent", timeout_s=5),
        "test-agent",
    )


def test_crawl_delay_is_announced_and_parsed_as_a_decimal(start: StartServer) -> None:
    _, base = start(crawl_delay=1.5)
    status, body, _ = get_robots(base + "/robots.txt", user_agent="test-agent", timeout_s=5)
    assert status == 200
    assert body == b"User-agent: *\nCrawl-delay: 1.5\nAllow: /\n"
    policy = _robots_policy(base)
    assert policy.allowed(base + "/a")
    assert policy.crawl_delay(base + "/a") == 1.5


def test_no_crawl_delay_keeps_robots_unchanged(start: StartServer) -> None:
    _, base = start()
    assert _robots_policy(base).crawl_delay(base + "/") is None


@pytest.mark.parametrize("bad", [0, 0.0, -1, float("nan"), float("inf"), True, "1.5"])
def test_make_server_rejects_an_invalid_crawl_delay(bad: object) -> None:
    with pytest.raises(ValueError, match="crawl_delay"):
        make_server(port=0, crawl_delay=bad)  # type: ignore[arg-type]


def test_main_rejects_an_invalid_crawl_delay(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--crawl-delay", "0"])
    assert exc.value.code == 2
    assert "crawl_delay" in capsys.readouterr().err


# --- CR-19: robots.txt redirect modes ------------------------------------------------------


@pytest.mark.parametrize("mode,status", [("301", 301), ("302", 302)])
def test_robots_redirect_answers_location_to_final_body(
    start: StartServer, mode: str, status: int
) -> None:
    _, base = start(robots_redirect=mode, crawl_delay=1.5)
    got_status, body, location = get_robots(base + "/robots.txt", user_agent="ua", timeout_s=5)
    assert (got_status, body, location) == (status, b"", "/robots-final.txt")
    final = get_robots(base + "/robots-final.txt", user_agent="ua", timeout_s=5)
    assert final == (200, b"User-agent: *\nCrawl-delay: 1.5\nAllow: /\n", None)
    policy = _robots_policy(base)
    assert policy.allowed(base + "/a")
    assert policy.crawl_delay(base + "/a") == 1.5


def test_robots_redirect_loop_disallows(start: StartServer) -> None:
    _, base = start(robots_redirect="loop")
    assert get_robots(base + "/robots.txt", user_agent="ua", timeout_s=5)[2] == "/robots-loop.txt"
    assert get_robots(base + "/robots-loop.txt", user_agent="ua", timeout_s=5)[2] == "/robots.txt"
    assert _robots_policy(base).allowed(base + "/a") is False


def test_without_redirect_mode_robots_final_is_404(start: StartServer) -> None:
    _, base = start()
    assert get_robots(base + "/robots-final.txt", user_agent="ua", timeout_s=5)[0] == 404


def test_make_server_rejects_an_unknown_robots_redirect() -> None:
    with pytest.raises(ValueError, match="robots_redirect"):
        make_server(port=0, robots_redirect="303")


def test_main_rejects_an_unknown_robots_redirect(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--robots-redirect", "bogus"])
    assert exc.value.code == 2
