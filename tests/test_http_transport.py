"""Tests for the stdlib HTTP transport (CL-01). Local ``http.server`` on 127.0.0.1 only."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from adapters.http_transport import get_robots, make_post_transport

from adapters.fetcher import FetchError


class _Handler(BaseHTTPRequestHandler):
    seen: dict[str, object] = {}

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _Handler.seen = {"body": self.rfile.read(length), "ctype": self.headers["Content-Type"]}
        size = int(self.path.rsplit("/", 1)[-1]) if self.path.startswith("/size/") else 2
        status = 500 if self.path == "/fail" else 200
        self._send(status, b"x" * size)

    def do_GET(self) -> None:  # noqa: N802
        _Handler.seen = {"ua": self.headers["User-Agent"]}
        if self.path == "/missing/robots.txt":
            self._send(404, b"nope")
        else:
            self._send(200, b"a" * (600 * 1024))

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_post_sends_body_and_headers(base: str) -> None:
    post = make_post_transport(max_bytes=10, timeout_s=5)
    status, body = post(base + "/ok", b'{"a":1}', {"Content-Type": "application/json"})
    assert (status, body) == (200, b"xx")
    assert _Handler.seen == {"body": b'{"a":1}', "ctype": "application/json"}


def test_post_read_cap_enforced(base: str) -> None:
    post = make_post_transport(max_bytes=10, timeout_s=5)
    assert post(base + "/size/10", b"", {})[1] == b"x" * 10
    with pytest.raises(FetchError, match="exceeds 10 bytes"):
        post(base + "/size/11", b"", {})


def test_post_non_2xx_returns_status(base: str) -> None:
    post = make_post_transport(max_bytes=10, timeout_s=5)
    assert post(base + "/fail", b"", {})[0] == 500


def test_post_rejects_non_http() -> None:
    post = make_post_transport(max_bytes=10, timeout_s=5)
    with pytest.raises(FetchError):
        post("file:///etc/passwd", b"", {})


def test_robots_get_caps_body_and_sends_user_agent(base: str) -> None:
    status, body = get_robots(base + "/robots.txt", user_agent="ua-test", timeout_s=5)
    assert status == 200
    assert len(body) == 512 * 1024
    assert _Handler.seen == {"ua": "ua-test"}


def test_robots_get_404_status(base: str) -> None:
    assert get_robots(base + "/missing/robots.txt", user_agent="ua", timeout_s=5)[0] == 404
