"""Stdlib HTTP transports for the crawl stage (CL-01).

* :func:`make_post_transport` returns a :data:`adapters.firecrawl.Transport`: one ``urllib`` POST
  with a timeout that reads at most ``max_bytes + 1`` bytes and raises
  :class:`~adapters.fetcher.FetchError` when the response is larger, so the response is never
  buffered past its cap (closes the AD-13 "caps checked after read" residual for this transport).
* :func:`get_robots` is a ``RobotsFetch`` for :class:`adapters.crawl_policy.RobotsPolicy`: one GET
  with the policy's user agent that reads at most :data:`~adapters.crawl_policy.MAX_ROBOTS_BYTES`
  (CR-02 ignores rules beyond the cap anyway). It returns ``(status, body, location)``: on a 3xx
  ``location`` is the raw ``Location`` header (``None`` otherwise), so the policy follows robots
  redirects itself, one capped, timed GET per hop (CR-19, RFC 9309).

The transport itself never follows redirects (a page redirect target would bypass the robots check
and the per-host limiter): a 3xx is returned as its status like any other non-2xx. Non-2xx
responses return an empty body instead of raising, so callers apply their own status
rules. Only http(s) URLs are accepted. Error messages carry the exception type, never the URL.

Production caller: the CLI ``crawl`` stage (:mod:`app.cli`).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit

from adapters.crawl_policy import MAX_ROBOTS_BYTES
from adapters.fetcher import FetchError
from adapters.firecrawl import Transport

__all__ = ["get_robots", "make_post_transport"]


def _check_http(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("transport accepts http(s) URLs only")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect so the 3xx surfaces as an ``HTTPError``."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _request(
    request: urllib.request.Request, timeout_s: float, limit: int
) -> tuple[int, bytes, str | None]:
    """Open ``request``; return ``(status, at most limit bytes, 3xx Location)``.

    Non-2xx bodies are dropped; ``location`` is ``None`` unless the status is a 3xx.
    """
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            return int(response.status), response.read(limit), None
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location") if 300 <= exc.code < 400 and exc.headers else None
        exc.close()
        return int(exc.code), b"", location


def make_post_transport(*, max_bytes: int, timeout_s: float) -> Transport:
    """Build a POST transport that refuses responses over ``max_bytes``."""
    if max_bytes < 1 or timeout_s <= 0:
        raise ValueError("max_bytes and timeout_s must be positive")

    def post(url: str, body: bytes, headers: Mapping[str, str]) -> tuple[int, bytes]:
        _check_http(url)
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        status, data, _ = _request(request, timeout_s, max_bytes + 1)
        if len(data) > max_bytes:
            raise FetchError(f"response exceeds {max_bytes} bytes")
        return status, data

    return post


def get_robots(url: str, *, user_agent: str, timeout_s: float) -> tuple[int, bytes, str | None]:
    """GET ``url`` (a robots.txt) reading at most ``MAX_ROBOTS_BYTES``; one hop, not followed."""
    _check_http(url)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent}, method="GET")
    return _request(request, timeout_s, MAX_ROBOTS_BYTES)
