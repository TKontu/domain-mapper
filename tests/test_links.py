"""Unit tests for the same-host link extractor (CR-06); ``example.com``/``.invalid`` only (D-003)."""

from __future__ import annotations

from core.links import MAX_HREFS_PER_PAGE, extract_links, normalise_url

HOST = "alpha.example.com"
BASE = "https://alpha.example.com/dir/page"


def links(body: str, base: str = BASE) -> list[str]:
    return extract_links(f"<html><body>{body}</body></html>", base_url=base, host=HOST)


def test_relative_absolute_and_protocol_relative_hrefs() -> None:
    body = (
        '<a href="rel">r</a><a href="/abs">a</a>'
        '<a href="https://alpha.example.com/full">f</a><a href="//alpha.example.com/proto">p</a>'
        '<a href="http://alpha.example.com/plain">h</a>'
    )
    assert links(body) == [
        "https://alpha.example.com/dir/rel",
        "https://alpha.example.com/abs",
        "https://alpha.example.com/full",
        "https://alpha.example.com/proto",
        "http://alpha.example.com/plain",
    ]


def test_resolves_against_final_url_after_redirect() -> None:
    assert links('<a href="next">n</a>', base="https://alpha.example.com/moved/here") == [
        "https://alpha.example.com/moved/next"
    ]


def test_off_host_subdomain_and_non_http_dropped() -> None:
    body = (
        '<a href="https://other.invalid/x">o</a><a href="https://www.alpha.example.com/y">s</a>'
        '<a href="mailto:a@example.com">m</a><a href="javascript:void(0)">j</a>'
        '<a href="ftp://alpha.example.com/f">f</a><a href="tel:123">t</a>'
    )
    assert links(body) == []


def test_fragment_query_and_duplicates_collapse() -> None:
    body = '<a href="/x">1</a><a href="/x#y">2</a><a href="/x?q=1">3</a><a href="/z">4</a>'
    assert links(body) == ["https://alpha.example.com/x", "https://alpha.example.com/z"]


def test_non_html_extensions_skipped() -> None:
    body = "".join(f'<a href="/f{ext}">f</a>' for ext in (".pdf", ".JPG", ".png", ".zip"))
    assert links(body + '<a href="/page.html">p</a>') == ["https://alpha.example.com/page.html"]


def test_malformed_hrefs_ignored() -> None:
    body = (
        '<a href="http://[bad">b</a><a href="https://alpha.example.com:99999/p">p</a>'
        '<a>none</a><a href>empty</a><a href="/ok">ok</a><link href="/css-link">'
    )
    assert links(body) == ["https://alpha.example.com/ok"]


def test_href_cap() -> None:
    body = "".join(f'<a href="/p{i}">x</a>' for i in range(MAX_HREFS_PER_PAGE + 10))
    assert len(links(body)) == MAX_HREFS_PER_PAGE


def test_normalise_url() -> None:
    assert normalise_url("HTTPS://Alpha.Example.com") == "https://alpha.example.com/"
    assert normalise_url("https://alpha.example.com/x?a#b") == "https://alpha.example.com/x"
    assert normalise_url("mailto:x@example.com") is None
