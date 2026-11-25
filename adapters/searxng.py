"""SearXNG implementation of the :class:`~adapters.search.Search` protocol (AD-12).

SearXNG is the decided search backend (D-004). :class:`SearxngSearch` sends one
GET to ``{base_url}/search?q=...&format=json&pageno=1`` per call and maps the
JSON ``results`` list to :class:`~adapters.search.SearchResult` records ranked
``1..n`` with ``backend="searxng"``, stopping at ``query.limit``.

Query mapping:

* ``site`` is folded into ``q`` as ``site:<host>``.
* ``language`` is sent as ``language=``. SearXNG has no country parameter but
  accepts locale codes, so language *and* country become ``language=<lang>-<CC>``.
  A country without a language is not sent and cannot be filtered client-side
  (results carry no country); results then have ``country=None`` (residual).

Errors: a transport failure, non-2xx status, non-JSON body, missing or ill-typed
``results``, or any entry within the limit lacking a string http(s) ``url`` (or
with non-string ``title``/``content``) raise :class:`SearchError`. Bad entries
are **not** skipped: a malformed response is treated as a failed call. Messages
never contain the base URL, query text or response body.

HTTP goes through an injectable GET transport (the :mod:`adapters.embedding`
pattern, GET variant defined here); the default uses only :mod:`urllib.request`.

Bound: one response body in memory; at most ``query.limit`` results; page 1 only.
No pagination, retries, rate limiting or caching.

The module is **unwired by design**: the first production caller is the Phase 1
Discover row (search channel).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.client import responses as _HTTP_REASONS
from typing import Any, Final, Protocol
from urllib.parse import urlencode, urlsplit

from adapters.embedding import HttpResponse
from adapters.search import Search, SearchError, SearchQuery, SearchResult

__all__ = ["HttpResponse", "SearxngSearch", "Transport", "urllib_get_transport"]

BACKEND: Final = "searxng"
DEFAULT_TIMEOUT_SECONDS: Final = 10.0


class Transport(Protocol):
    """Injected HTTP GET transport; the default is :func:`urllib_get_transport`."""

    def __call__(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse: ...


def urllib_get_transport(url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Default transport: one stdlib GET."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(status=exc.code)


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _validate_timeout(timeout: float) -> float:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite number of seconds > 0")
    return float(timeout)


class SearxngSearch:
    """:class:`~adapters.search.Search` over a SearXNG instance's JSON API."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not _is_http_url(base_url):
            raise ValueError("base_url must be an http(s) URL")
        self._base = base_url.rstrip("/")
        self._transport: Transport = urllib_get_transport if transport is None else transport
        self._timeout = _validate_timeout(timeout)

    def __repr__(self) -> str:  # never expose the instance URL
        return "SearxngSearch(<base_url hidden>)"

    def _url(self, query: SearchQuery) -> str:
        text = query.text.strip()
        if query.site:
            text = f"{text} site:{query.site.strip()}"
        params: dict[str, str] = {"q": text, "format": "json", "pageno": "1"}
        if query.language:
            lang = query.language.strip()
            if query.country:
                lang = f"{lang}-{query.country.strip().upper()}"
            params["language"] = lang
        return f"{self._base}/search?{urlencode(params)}"

    def _fetch(self, url: str) -> list[Any]:
        try:
            response = self._transport(
                url, headers={"Accept": "application/json"}, timeout=self._timeout
            )
        except Exception as exc:
            raise SearchError(f"transport failed: {type(exc).__name__}") from None
        if not 200 <= response.status < 300:
            reason = _HTTP_REASONS.get(response.status, "error")
            raise SearchError(f"HTTP {response.status} {reason}")
        try:
            payload = json.loads(response.body)
        except ValueError:
            raise SearchError("response body is not valid JSON") from None
        if not isinstance(payload, dict):
            raise SearchError("response body is not a JSON object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise SearchError("response has no results list")
        return results

    def search(self, query: SearchQuery) -> list[SearchResult]:
        out: list[SearchResult] = []
        for entry in self._fetch(self._url(query)):
            if len(out) >= query.limit:
                break
            if not isinstance(entry, dict) or not _is_http_url(entry.get("url")):
                raise SearchError(f"result {len(out) + 1} has no http(s) url")
            title = entry.get("title")
            snippet = entry.get("content")
            title = "" if title is None else title
            snippet = "" if snippet is None else snippet
            if not isinstance(title, str) or not isinstance(snippet, str):
                raise SearchError(f"result {len(out) + 1} has a non-string title or content")
            out.append(
                SearchResult(
                    query=query.text,
                    url=entry["url"],
                    title=title,
                    snippet=snippet,
                    rank=len(out) + 1,
                    backend=BACKEND,
                    language=query.language,
                    country=query.country if query.language else None,
                )
            )
        return out


def _conforms() -> Search:
    """Static check: mypy rejects this if the adapter drifts from the protocol."""
    return SearxngSearch(
        "http://search.invalid", transport=lambda url, *, headers, timeout: HttpResponse(200)
    )
