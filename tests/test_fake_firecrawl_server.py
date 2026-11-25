"""Tests for the local fake Firecrawl backend (CR-11). Loopback sockets only, no network."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

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

    def _start(*, fail_first: int | None = None) -> tuple[FakeFirecrawlServer, str]:
        server = make_server(port=0, fail_first=fail_first)
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
    assert get_robots(base + "/robots.txt", user_agent="test-agent", timeout_s=5) == (200, ROBOTS)


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
