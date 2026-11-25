"""Search adapter interface for the Discover search channel (AD-05).

ARCHITECTURE.md §4 (Discover) lists a Search channel whose queries are built
from scope vocabulary × language × country. This module defines only the
interface: :class:`Search` is the protocol the pipeline talks to,
:class:`SearchQuery` and :class:`SearchResult` are its frozen records, and
:class:`SearchError` is the failure type. :class:`InMemorySearch` is a bounded
fake for tests and the example scope.

The search backend is an open decision (todo.md "Search backend for
discovery"); no real backend, network call or dependency lives here.

Bound: :meth:`Search.search` returns a materialised ``list`` of at most
``query.limit`` results, ranked ``1..n`` in backend order. A real backend must
push the limit to the source rather than fetch and truncate.

The module is **unwired by design**: the first production caller is the
Phase 1 Discover row (search channel).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable
from urllib.parse import urlsplit

DEFAULT_LIMIT: Final = 25


class SearchError(ValueError):
    """An invalid search query or a failed search call."""


@dataclass(frozen=True)
class SearchQuery:
    """One search request: text plus optional language/country/site filters."""

    text: str
    limit: int = DEFAULT_LIMIT
    language: str | None = None
    country: str | None = None
    site: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise SearchError("query text must be a non-blank string")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
            raise SearchError("limit must be a positive integer")


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit as returned by a backend; not a database row."""

    query: str
    url: str
    title: str
    snippet: str
    rank: int
    backend: str
    language: str | None = None
    country: str | None = None


@runtime_checkable
class Search(Protocol):
    """A web search backend used by the Discover search channel."""

    def search(self, query: SearchQuery) -> list[SearchResult]: ...


def _host_matches(url: str, site: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    site = site.lower().strip(".")
    return host == site or host.endswith("." + site)


class InMemorySearch:
    """Fake :class:`Search` over a fixed document list (tests, example scope).

    A document matches when the query text is a case-insensitive substring of
    its title or snippet and every given filter matches (language and country
    by case-insensitive equality, site by host or subdomain).
    """

    def __init__(
        self, documents: Iterable[Mapping[str, str | None]], *, backend: str = "memory"
    ) -> None:
        self._docs = tuple(dict(d) for d in documents)
        self._backend = backend

    def search(self, query: SearchQuery) -> list[SearchResult]:
        needle = query.text.strip().lower()
        out: list[SearchResult] = []
        for doc in self._docs:
            if len(out) >= query.limit:
                break
            url = doc.get("url") or ""
            title = doc.get("title") or ""
            snippet = doc.get("snippet") or ""
            lang = doc.get("language")
            country = doc.get("country")
            if needle not in title.lower() and needle not in snippet.lower():
                continue
            if query.language and (lang or "").lower() != query.language.lower():
                continue
            if query.country and (country or "").lower() != query.country.lower():
                continue
            if query.site and not _host_matches(url, query.site):
                continue
            out.append(
                SearchResult(
                    query=query.text,
                    url=url,
                    title=title,
                    snippet=snippet,
                    rank=len(out) + 1,
                    backend=self._backend,
                    language=lang,
                    country=country,
                )
            )
        return out


def _conforms() -> Search:
    """Static check: mypy rejects this if the fake drifts from the protocol."""
    return InMemorySearch(())
