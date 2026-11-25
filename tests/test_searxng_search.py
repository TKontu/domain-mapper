"""Tests for the SearXNG implementation of the Search protocol (AD-12). No network."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from adapters.search import Search, SearchError, SearchQuery, SearchResult
from adapters.searxng import HttpResponse, SearxngSearch

BASE = "http://search.invalid"


class FakeTransport:
    def __init__(self, response: HttpResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.calls.append((url, dict(headers), timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def ok(payload: Any) -> HttpResponse:
    return HttpResponse(status=200, body=json.dumps(payload).encode())


def hits(n: int) -> dict[str, Any]:
    return {
        "results": [
            {"url": f"https://h{i}.example/p", "title": f"T{i}", "content": f"S{i}"}
            for i in range(1, n + 1)
        ]
    }


def run(
    query: SearchQuery, response: HttpResponse | Exception
) -> tuple[list[SearchResult], FakeTransport]:
    t = FakeTransport(response)
    return SearxngSearch(BASE, transport=t).search(query), t


def params(t: FakeTransport) -> tuple[str, dict[str, list[str]]]:
    assert len(t.calls) == 1
    parts = urlsplit(t.calls[0][0])
    return f"{parts.scheme}://{parts.netloc}{parts.path}", parse_qs(parts.query)


def test_satisfies_protocol() -> None:
    assert isinstance(SearxngSearch(BASE, transport=FakeTransport(ok(hits(0)))), Search)


def test_query_string_basic() -> None:
    _, t = run(SearchQuery("a b&c"), ok(hits(0)))
    path, q = params(t)
    assert path == BASE + "/search"
    assert q == {"q": ["a b&c"], "format": ["json"], "pageno": ["1"]}
    assert "a+b%26c" in t.calls[0][0] or "a%20b%26c" in t.calls[0][0]


def test_trailing_slash_base() -> None:
    t = FakeTransport(ok(hits(0)))
    SearxngSearch(BASE + "/", transport=t).search(SearchQuery("x"))
    assert params(t)[0] == BASE + "/search"


def test_language_and_site() -> None:
    _, t = run(SearchQuery("widgets", language="fi", site="a.example"), ok(hits(0)))
    q = params(t)[1]
    assert q["q"] == ["widgets site:a.example"]
    assert q["language"] == ["fi"]


def test_country_folded_into_language_locale() -> None:
    _, t = run(SearchQuery("w", language="fi", country="fi"), ok(hits(0)))
    assert params(t)[1]["language"] == ["fi-FI"]


def test_country_without_language_not_sent() -> None:
    out, t = run(SearchQuery("w", country="fi"), ok(hits(2)))
    q = params(t)[1]
    assert "language" not in q and "country" not in q
    assert all(r.country is None for r in out)


def test_mapping_and_ranks() -> None:
    out, _ = run(SearchQuery("w", language="en"), ok(hits(3)))
    assert out == [
        SearchResult(
            query="w",
            url=f"https://h{i}.example/p",
            title=f"T{i}",
            snippet=f"S{i}",
            rank=i,
            backend="searxng",
            language="en",
            country=None,
        )
        for i in (1, 2, 3)
    ]


def test_missing_title_content_default_empty() -> None:
    out, _ = run(SearchQuery("w"), ok({"results": [{"url": "http://x.example/"}]}))
    assert (out[0].title, out[0].snippet) == ("", "")


def test_null_title_content_default_empty() -> None:
    out, _ = run(
        SearchQuery("w"),
        ok({"results": [{"url": "http://x.example/", "title": None, "content": None}]}),
    )
    assert (out[0].title, out[0].snippet) == ("", "")


def test_limit_truncation() -> None:
    out, _ = run(SearchQuery("w", limit=2), ok(hits(5)))
    assert [r.rank for r in out] == [1, 2]


def test_bad_entry_beyond_limit_ignored() -> None:
    payload = hits(1)
    payload["results"].append({"url": 5})
    out, _ = run(SearchQuery("w", limit=1), ok(payload))
    assert len(out) == 1


@pytest.mark.parametrize("status", [301, 404, 500])
def test_non_2xx(status: int) -> None:
    with pytest.raises(SearchError, match=str(status)):
        run(SearchQuery("w"), HttpResponse(status=status, body=b'{"results": []}'))


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[1]", b"{}", b'{"results": {}}', b'{"results": null}'],
)
def test_bad_body(body: bytes) -> None:
    with pytest.raises(SearchError):
        run(SearchQuery("w"), HttpResponse(status=200, body=body))


@pytest.mark.parametrize(
    "entry",
    [
        "str",
        {},
        {"url": 3},
        {"url": "ftp://x.example/"},
        {"url": "javascript:alert(1)"},
        {"url": "http://"},
        {"url": "http://x.example/", "title": 1},
        {"url": "http://x.example/", "content": []},
    ],
)
def test_bad_entry_raises(entry: Any) -> None:
    with pytest.raises(SearchError):
        run(SearchQuery("w"), ok({"results": [entry]}))


def test_transport_exception_wrapped() -> None:
    with pytest.raises(SearchError, match="OSError"):
        run(SearchQuery("w"), OSError("secret http://search.invalid detail"))


def test_error_message_does_not_leak_base_url() -> None:
    with pytest.raises(SearchError) as info:
        run(SearchQuery("w"), HttpResponse(status=500))
    assert "search.invalid" not in str(info.value)


def test_timeout_passed() -> None:
    t = FakeTransport(ok(hits(0)))
    SearxngSearch(BASE, transport=t, timeout=3).search(SearchQuery("w"))
    assert t.calls[0][2] == 3.0


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan, True, "5"])
def test_timeout_validation(timeout: Any) -> None:
    with pytest.raises(ValueError):
        SearxngSearch(BASE, transport=FakeTransport(ok(hits(0))), timeout=timeout)


@pytest.mark.parametrize("base", ["", "search.invalid", "ftp://search.invalid", 5])
def test_base_url_validation(base: Any) -> None:
    with pytest.raises(ValueError):
        SearxngSearch(base, transport=FakeTransport(ok(hits(0))))


def test_no_network_default_transport_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    run(SearchQuery("w"), ok(hits(1)))
