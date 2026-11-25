"""A local stand-in for the Firecrawl scrape API, for tests and live harnesses.

``landscape crawl`` builds :class:`adapters.firecrawl.FirecrawlFetcher` from
``FIRECRAWL_URL``; point that variable at this server to exercise the crawl stage,
including its one-retry path, end to end over real loopback sockets::

    python -m adapters.fake_firecrawl_server --port 3002 --fail-first 408
    FIRECRAWL_URL=http://127.0.0.1:3002 landscape crawl --url http://127.0.0.1:3002/

Endpoints (bound to ``127.0.0.1`` only):

* ``POST /v1/scrape`` with ``{"url": ...}`` answers a Firecrawl success envelope.
  The page's ``<h1>`` is the requested URL's path; the page for ``/`` links ``/a``
  and ``/b`` on the requested URL's own host. With ``fail_first`` the first scrape
  of each distinct URL answers that status with an empty body and later scrapes
  succeed, so the caller's retry seam is exercised over a real socket.
* ``GET /robots.txt`` answers an allow-all policy, because the CLI fetches robots
  directly from the target host rather than through the fetcher.
* every other path or method answers 404; a ``/v1/scrape`` body that is not a JSON
  object with an http(s) ``url`` answers 400.

This is a **test/live-harness backend, never production**: it fetches, renders and
sanitizes nothing, and it serves only loopback clients.

Bound: the per-URL failure counter holds one entry per distinct URL scraped, so
memory grows with the number of distinct URLs requested (one small integer and
key each) and is never reset while the server runs.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import urlsplit

__all__ = ["FakeFirecrawlServer", "make_server", "main"]

_HOST: Final = "127.0.0.1"
_ROBOTS: Final = b"User-agent: *\nAllow: /\n"
_MAX_BODY_BYTES: Final = 1 << 20
_SCRAPE_PATH: Final = "/v1/scrape"


def _fail_status(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 400 <= value <= 599:
        raise ValueError("fail_first must be an HTTP error status in 400..599")
    return value


def _port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
        raise ValueError("port must be an integer in 0..65535")
    return value


class FakeFirecrawlServer(ThreadingHTTPServer):
    """Threaded loopback server carrying the per-URL first-scrape counter."""

    daemon_threads = True

    def __init__(self, port: int, fail_first: int | None) -> None:
        self.fail_first = _fail_status(fail_first)
        self._lock = threading.Lock()
        self._scrapes: dict[str, int] = {}
        super().__init__((_HOST, _port(port)), _Handler)

    def take_scrape(self, url: str) -> int:
        """Count one scrape of ``url``; return its ordinal (1 for the first)."""
        with self._lock:
            ordinal = self._scrapes.get(url, 0) + 1
            self._scrapes[url] = ordinal
            return ordinal


def make_server(port: int = 0, fail_first: int | None = None) -> FakeFirecrawlServer:
    """Build (but do not start) a server on ``127.0.0.1``; ``port=0`` picks one."""
    return FakeFirecrawlServer(port, fail_first)


def _scrape_body(url: str) -> bytes:
    """A Firecrawl success envelope for ``url`` with deterministic fixture HTML."""
    parts = urlsplit(url)
    path = parts.path or "/"
    origin = f"{parts.scheme}://{parts.netloc}"
    links = ""
    if path == "/":
        links = f'<a href="{origin}/a">a</a> <a href="{origin}/b">b</a>'
    raw_html = (
        "<!doctype html><html><head><title>fake page</title></head>"
        f"<body><h1>{html.escape(path)}</h1>{links}<p>Local fixture page.</p></body></html>"
    )
    payload = {
        "success": True,
        "data": {
            "rawHtml": raw_html,
            "metadata": {"url": url, "statusCode": 200, "contentType": "text/html"},
        },
    }
    return json.dumps(payload).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server: FakeFirecrawlServer

    def log_message(self, *args: object) -> None:
        """Stay silent: a live lane's log is the CLI's, not per-request noise."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
        if urlsplit(self.path).path == "/robots.txt":
            self._send(200, _ROBOTS, "text/plain; charset=utf-8")
        else:
            self._not_found()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
        if urlsplit(self.path).path != _SCRAPE_PATH:
            self._not_found()
            return
        url = self._read_target()
        if url is None:
            self._send(400, b"", "application/json")
            return
        ordinal = self.server.take_scrape(url)
        fail_status = self.server.fail_first
        if fail_status is not None and ordinal == 1:
            self._send(fail_status, b"", "application/json")
            return
        self._send(200, _scrape_body(url), "application/json")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler naming
        self._not_found()

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler naming
        self._not_found()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler naming
        self._not_found()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler naming
        self._not_found()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler naming
        self._not_found()

    def _read_target(self) -> str | None:
        """The requested http(s) URL, or None for a malformed scrape body."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > _MAX_BODY_BYTES:
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, RecursionError):
            return None
        if not isinstance(payload, dict):
            return None
        url = payload.get("url")
        if not isinstance(url, str):
            return None
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None
        return url

    def _not_found(self) -> None:
        self._send(404, b"", "text/plain; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> int:
    """Run until interrupted; print the bound URL first so a lane can capture it."""
    parser = argparse.ArgumentParser(
        prog="fake_firecrawl_server",
        description="Local (127.0.0.1) fake of the Firecrawl /v1/scrape API.",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="loopback port; 0 (default) picks a free one"
    )
    parser.add_argument(
        "--fail-first",
        type=int,
        default=None,
        metavar="STATUS",
        help="first scrape per distinct URL answers this HTTP error status (400..599)",
    )
    args = parser.parse_args(argv)
    try:
        server = make_server(port=args.port, fail_first=args.fail_first)
    except ValueError as exc:
        parser.error(str(exc))
    host, port = server.server_address[:2]
    print(f"fake firecrawl on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
