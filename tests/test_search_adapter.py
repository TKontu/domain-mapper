"""Tests for the Search adapter protocol and in-memory fake (AD-05)."""

from __future__ import annotations

import dataclasses

import pytest

from adapters.search import (
    InMemorySearch,
    Search,
    SearchError,
    SearchQuery,
    SearchResult,
)

DOCS = [
    ("https://a.example/one", "Alpha widgets", "alpha widgets maker", "en", "fi"),
    ("https://b.example/two", "Beta widgets", "beta widget service", "fi", "fi"),
    ("https://a.example/three", "Gamma", "widgets and gadgets", "en", "se"),
    ("https://c.example/four", "Delta", "unrelated text", "en", "fi"),
]


def make() -> InMemorySearch:
    return InMemorySearch(
        [
            {"url": u, "title": t, "snippet": s, "language": lang, "country": c}
            for u, t, s, lang, c in DOCS
        ],
        backend="memory",
    )


def test_fake_satisfies_protocol() -> None:
    assert isinstance(make(), Search)

    class NoSearch:
        pass

    assert not isinstance(NoSearch(), Search)


def test_results_ranked_and_labelled() -> None:
    results = make().search(SearchQuery(text="widgets", limit=10))
    assert [r.url for r in results] == [
        "https://a.example/one",
        "https://b.example/two",
        "https://a.example/three",
    ]
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.query == "widgets" and r.backend == "memory" for r in results)
    assert results[0].language == "en" and results[0].country == "fi"


def test_limit_enforced() -> None:
    results = make().search(SearchQuery(text="widgets", limit=2))
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_rejected(limit: int) -> None:
    with pytest.raises(SearchError):
        SearchQuery(text="widgets", limit=limit)


def test_blank_text_rejected() -> None:
    with pytest.raises(SearchError):
        SearchQuery(text="  ")


def test_language_filter() -> None:
    results = make().search(SearchQuery(text="widgets", language="fi"))
    assert [r.url for r in results] == ["https://b.example/two"]


def test_country_filter() -> None:
    results = make().search(SearchQuery(text="widgets", country="se"))
    assert [r.url for r in results] == ["https://a.example/three"]


def test_site_filter() -> None:
    results = make().search(SearchQuery(text="widgets", site="a.example"))
    assert [r.url for r in results] == [
        "https://a.example/one",
        "https://a.example/three",
    ]
    assert [r.rank for r in results] == [1, 2]


def test_case_insensitive_and_no_match() -> None:
    assert len(make().search(SearchQuery(text="ALPHA"))) == 1
    assert make().search(SearchQuery(text="zzz")) == []


def test_records_frozen() -> None:
    r = make().search(SearchQuery(text="alpha"))[0]
    assert isinstance(r, SearchResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.rank = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        SearchQuery(text="x").limit = 3  # type: ignore[misc]
