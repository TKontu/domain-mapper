"""Firecrawl implementation of the :class:`~adapters.fetcher.Fetcher` protocol (AD-13).

:class:`FirecrawlFetcher` posts one JSON body to ``{base_url}/v1/scrape`` per call::

    {"url": <requested url>, "formats": ["rawHtml"], "timeout": <timeout_ms>}

and maps Firecrawl's ``{"success": true, "data": {rawHtml, metadata}}`` envelope
to a :class:`~adapters.fetcher.FetchResult`:

* ``rawHtml`` is encoded as UTF-8 into ``body_bytes`` (D-005 keeps the raw bytes
  so the blob store can key them by sha256);
* ``metadata.statusCode`` becomes ``status`` (the transport's 2xx is only the
  fallback when the field is absent or not an integer);
* the final URL is ``metadata.url``, then ``metadata.sourceURL``, then the
  requested URL;
* ``content_type`` is ``metadata.contentType`` or ``text/html; charset=utf-8``.

Firecrawl always renders JavaScript, so ``render_js`` is accepted only for
protocol conformance: both values send the same request and map to no option
(there is no ``waitFor`` tuning). Response headers are not carried by the
Firecrawl envelope, so ``FetchResult.headers`` is empty.

Errors: a non-http(s) request URL, a transport exception, a non-2xx transport
status, a response over the response cap, non-JSON or non-object JSON, JSON
nested too deeply (``RecursionError`` from ``json.loads``), a missing/invalid
``data`` object, ``success`` that is not ``true`` (with the ``error`` text,
truncated), a missing or non-string ``rawHtml``, a ``rawHtml`` that is not
encodable as UTF-8 (a lone surrogate), or ``rawHtml`` over ``max_bytes`` raise
:class:`~adapters.fetcher.FetchError`. Messages never contain the response body
or the base URL; ``repr`` hides the base URL.

HTTP goes through an injected POST transport
``(url, body, headers) -> (status, body_bytes)``; no real transport is shipped
here (an optional thin stdlib one belongs with the wiring row).

Bound: one response in memory. The response is capped at
``2 * max_bytes + 64 KiB`` (JSON escaping and the envelope around ``rawHtml``)
and ``rawHtml`` is additionally capped at ``max_bytes``. Firecrawl returns the
whole body in JSON, so both caps are checked after the transport read unless the
transport itself streams with a cap (residual).

``from_env(transport)`` reads ``FIRECRAWL_URL`` from the process environment
(the ``DM_*`` settings pattern in :mod:`core.config` does not cover adapter
variables); the host is never hard-coded.

The module is **unwired by design**: the first production caller is the CLI
``crawl`` stage row. robots.txt, rate limiting and caching stay the caller's job
(CR-02).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any, Final
from urllib.parse import urlsplit

from adapters.fetcher import DEFAULT_MAX_BYTES, Fetcher, FetchError, FetchResult

__all__ = ["DEFAULT_TIMEOUT_MS", "FirecrawlFetcher", "Transport"]

Transport = Callable[[str, bytes, Mapping[str, str]], tuple[int, bytes]]
"""POST ``body`` to ``url`` with ``headers``; return ``(status, response bytes)``."""

DEFAULT_TIMEOUT_MS: Final = 30_000
"""Firecrawl's own default request timeout, in milliseconds."""

_ENV_VAR: Final = "FIRECRAWL_URL"
_RESPONSE_ENVELOPE_BYTES: Final = 64 * 1024
_MAX_ERROR_CHARS: Final = 200
_DEFAULT_CONTENT_TYPE: Final = "text/html; charset=utf-8"


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _metadata_text(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _truncate(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > _MAX_ERROR_CHARS:
        return text[:_MAX_ERROR_CHARS] + "..."
    return text


def _encode_raw_html(raw_html: str) -> bytes:
    """UTF-8 encode ``rawHtml``; a lone surrogate is malformed, not a crash."""
    try:
        return raw_html.encode("utf-8")
    except UnicodeEncodeError:
        raise FetchError("rawHtml is not encodable as UTF-8") from None


class FirecrawlFetcher:
    """:class:`~adapters.fetcher.Fetcher` over Firecrawl's ``/v1/scrape`` API."""

    def __init__(
        self,
        base_url: str,
        transport: Transport,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        if not _is_http_url(base_url):
            raise ValueError("base_url must be an http(s) URL")
        self._base = base_url.rstrip("/")
        self._transport = transport
        self._max_bytes = _positive_int("max_bytes", max_bytes)
        self._timeout_ms = _positive_int("timeout_ms", timeout_ms)
        self._response_cap = 2 * self._max_bytes + _RESPONSE_ENVELOPE_BYTES

    @classmethod
    def from_env(
        cls,
        transport: Transport,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> FirecrawlFetcher:
        """Build from ``FIRECRAWL_URL``; raise ``ValueError`` when it is unset or blank."""
        base_url = os.environ.get(_ENV_VAR, "").strip()
        if not base_url:
            raise ValueError(f"{_ENV_VAR} is not set; set it to the Firecrawl base URL")
        return cls(base_url, transport, max_bytes=max_bytes, timeout_ms=timeout_ms)

    def __repr__(self) -> str:  # never expose the instance URL
        return "FirecrawlFetcher(<base_url hidden>)"

    def fetch(self, url: str, *, render_js: bool = False) -> FetchResult:
        """Scrape ``url``; ``render_js`` is accepted but changes nothing."""
        if not _is_http_url(url):
            raise FetchError(f"not an http(s) URL: {url!r}")
        request = {"url": url, "formats": ["rawHtml"], "timeout": self._timeout_ms}
        body = json.dumps(request).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            status, response = self._transport(f"{self._base}/v1/scrape", body, headers)
        except FetchError:
            raise
        except Exception as exc:
            raise FetchError(f"transport failed: {type(exc).__name__}") from None
        if not 200 <= status < 300:
            raise FetchError(f"HTTP {status}")
        if len(response) > self._response_cap:
            raise FetchError(f"response exceeds {self._response_cap} bytes")
        data = self._parse(response)
        raw_html = data.get("rawHtml")
        if not isinstance(raw_html, str):
            raise FetchError("response has no rawHtml")
        body_bytes = _encode_raw_html(raw_html)
        if len(body_bytes) > self._max_bytes:
            raise FetchError(f"body exceeds max_bytes={self._max_bytes}")
        metadata = data.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        final_url = _metadata_text(metadata, "url") or _metadata_text(metadata, "sourceURL") or url
        content_type = _metadata_text(metadata, "contentType") or _DEFAULT_CONTENT_TYPE
        code = metadata.get("statusCode")
        status_code = code if isinstance(code, int) and not isinstance(code, bool) else status
        return FetchResult(
            final_url=final_url,
            status=status_code,
            content_type=content_type,
            body_bytes=body_bytes,
        )

    @staticmethod
    def _parse(response: bytes) -> Mapping[str, Any]:
        try:
            payload = json.loads(response)
        except RecursionError:
            raise FetchError("response body is nested too deeply") from None
        except ValueError:
            raise FetchError("response body is not valid JSON") from None
        if not isinstance(payload, Mapping):
            raise FetchError("response body is not a JSON object")
        if payload.get("success") is True:
            data = payload.get("data")
            if not isinstance(data, Mapping):
                raise FetchError("response has no data object")
            return data
        error = payload.get("error")
        if error:
            raise FetchError(f"Firecrawl error: {_truncate(error)}")
        raise FetchError("response was not a successful Firecrawl response")


def _conforms() -> Fetcher:
    """Static check: mypy rejects this if the adapter drifts from the protocol."""
    return FirecrawlFetcher("http://firecrawl.invalid", lambda url, body, headers: (200, b""))
