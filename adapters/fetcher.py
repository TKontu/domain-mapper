"""Fetcher adapter interface for the Crawl & snapshot stage (AD-06).

ARCHITECTURE.md §4 (Crawl & snapshot) fixes the contract
``fetch(url, *, render_js=False) -> FetchResult``. This module defines only the
interface: :class:`Fetcher` is the protocol, :class:`FetchResult` its frozen
record and :class:`FetchError` the failure type. :class:`InMemoryFetcher` is a
bounded fake for tests and the example scope.

No real scraping tool, network call, robots.txt handling, rate limiting or raw
storage lives here (raw storage is an open decision).

Bound: a fetch returns at most ``max_bytes`` of body (default
:data:`DEFAULT_MAX_BYTES`); larger bodies raise :class:`FetchError`. A real
implementation must stream the response and stop reading at the cap rather
than download and then check.

The module is **unwired by design**: the first production caller is the
Phase 1 Crawl & snapshot row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable
from urllib.parse import urlsplit

DEFAULT_MAX_BYTES: Final = 10 * 1024 * 1024


class FetchError(ValueError):
    """An invalid fetch request or a failed fetch."""


@dataclass(frozen=True)
class FetchResult:
    """One fetched response; headers are stored as a read-only copy.

    ``fetched_at`` must be timezone-aware; it defaults to the current UTC time
    (aware non-UTC values are kept as given).
    """

    final_url: str
    status: int
    content_type: str
    body_bytes: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        tzinfo = self.fetched_at.tzinfo
        if tzinfo is None or tzinfo.utcoffset(self.fetched_at) is None:
            raise FetchError(f"fetched_at must be timezone-aware: {self.fetched_at!r}")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    @property
    def sha256(self) -> str:
        """Hex SHA-256 of ``body_bytes``."""
        return hashlib.sha256(self.body_bytes).hexdigest()


@runtime_checkable
class Fetcher(Protocol):
    """Fetches one URL, optionally rendering JavaScript."""

    def fetch(self, url: str, *, render_js: bool = False) -> FetchResult: ...


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError(f"not an http(s) URL: {url!r}")


class InMemoryFetcher:
    """Fake :class:`Fetcher` over canned results (tests, example scope).

    ``redirects`` maps a URL to one target URL (a single hop; the target is
    looked up in ``results`` directly). ``render_js`` does not change the
    result; every call is recorded in :attr:`calls` as ``(url, render_js)``.
    """

    def __init__(
        self,
        results: Mapping[str, FetchResult],
        *,
        redirects: Mapping[str, str] | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise FetchError("max_bytes must be a positive integer")
        self._results = dict(results)
        self._redirects = dict(redirects or {})
        self._max_bytes = max_bytes
        self.calls: list[tuple[str, bool]] = []

    def fetch(self, url: str, *, render_js: bool = False) -> FetchResult:
        _check_url(url)
        self.calls.append((url, render_js))
        target = self._redirects.get(url, url)
        if target != url:
            _check_url(target)
        result = self._results.get(target)
        if result is None:
            raise FetchError(f"no canned result for {target!r}")
        if len(result.body_bytes) > self._max_bytes:
            raise FetchError(f"body exceeds {self._max_bytes} bytes: {target!r}")
        return result


def _conforms() -> Fetcher:
    """Static check: mypy rejects this if the fake drifts from the protocol."""
    return InMemoryFetcher({})
