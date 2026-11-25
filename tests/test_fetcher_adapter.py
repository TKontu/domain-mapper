"""Tests for the Fetcher adapter protocol and in-memory fake (AD-06)."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from adapters.fetcher import Fetcher, FetchError, FetchResult, InMemoryFetcher

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


def page(url: str, body: bytes = b"<html>hi</html>", status: int = 200) -> FetchResult:
    return FetchResult(
        final_url=url,
        status=status,
        content_type="text/html",
        body_bytes=body,
        headers={"Content-Type": "text/html"},
        fetched_at=WHEN,
    )


def test_fake_satisfies_protocol() -> None:
    assert isinstance(InMemoryFetcher({}), Fetcher)

    class NoFetch:
        pass

    assert not isinstance(NoFetch(), Fetcher)


def test_fields_are_exactly_contract() -> None:
    names = [f.name for f in dataclasses.fields(FetchResult)]
    assert names == ["final_url", "status", "content_type", "body_bytes", "headers", "fetched_at"]


def test_fetch_returns_canned_result() -> None:
    r = page("https://a.example/")
    f = InMemoryFetcher({"https://a.example/": r})
    assert f.fetch("https://a.example/") == r
    assert f.calls == [("https://a.example/", False)]


def test_render_js_accepted_and_recorded() -> None:
    f = InMemoryFetcher({"https://a.example/": page("https://a.example/")})
    f.fetch("https://a.example/", render_js=True)
    assert f.calls == [("https://a.example/", True)]


def test_unknown_url_raises() -> None:
    with pytest.raises(FetchError):
        InMemoryFetcher({}).fetch("https://missing.example/")


@pytest.mark.parametrize("url", ["ftp://a.example/", "file:///etc/passwd", "a.example", ""])
def test_non_http_url_rejected(url: str) -> None:
    f = InMemoryFetcher({url: page("https://a.example/")})
    with pytest.raises(FetchError):
        f.fetch(url)


def test_redirect_hop_changes_final_url() -> None:
    f = InMemoryFetcher(
        {"https://b.example/new": page("https://b.example/new")},
        redirects={"http://a.example/old": "https://b.example/new"},
    )
    r = f.fetch("http://a.example/old")
    assert r.final_url == "https://b.example/new"


def test_redirect_to_unknown_raises() -> None:
    f = InMemoryFetcher({}, redirects={"https://a.example/": "https://b.example/"})
    with pytest.raises(FetchError):
        f.fetch("https://a.example/")


def test_max_body_bound() -> None:
    f = InMemoryFetcher({"https://a.example/": page("https://a.example/", b"x" * 11)}, max_bytes=10)
    with pytest.raises(FetchError):
        f.fetch("https://a.example/")
    ok = InMemoryFetcher(
        {"https://a.example/": page("https://a.example/", b"x" * 10)}, max_bytes=10
    )
    assert len(ok.fetch("https://a.example/").body_bytes) == 10


def test_invalid_max_bytes() -> None:
    with pytest.raises(FetchError):
        InMemoryFetcher({}, max_bytes=0)


def test_fetched_at_defaults_to_aware_utc() -> None:
    r = FetchResult("https://a.example/", 200, "text/html", b"")
    assert r.fetched_at.tzinfo is not None
    assert r.fetched_at.utcoffset() == timedelta(0)


def test_naive_fetched_at_raises() -> None:
    with pytest.raises(FetchError):
        FetchResult("https://a.example/", 200, "text/html", b"", fetched_at=datetime(2026, 1, 1))


def test_aware_non_utc_fetched_at_kept_as_given() -> None:
    tz = timezone(timedelta(hours=2))
    when = datetime(2026, 1, 1, 12, tzinfo=tz)
    r = FetchResult("https://a.example/", 200, "text/html", b"", fetched_at=when)
    assert r.fetched_at is when
    assert r.fetched_at.utcoffset() == timedelta(hours=2)


def test_result_frozen_and_headers_immutable() -> None:
    src = {"X-A": "1"}
    r = FetchResult("https://a.example/", 200, "text/html", b"", src, WHEN)
    src["X-A"] = "2"
    assert r.headers["X-A"] == "1"
    with pytest.raises(TypeError):
        r.headers["X-A"] = "3"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.status = 404  # type: ignore[misc]


def test_sha256_stable() -> None:
    body = b"hello"
    a = page("https://a.example/", body)
    b = page("https://a.example/", body)
    assert a.sha256 == b.sha256 == hashlib.sha256(body).hexdigest()
