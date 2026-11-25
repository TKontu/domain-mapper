"""Tests for the stdlib HTML to Markdown parser (AD-10)."""

from __future__ import annotations

import pytest

from adapters.html_parser import HtmlParser
from adapters.parser import ParseError, Parser
from core.parse_chunk import parse_and_chunk

URL = "https://example.test/page"
HTML = "text/html"


def parse(body: bytes, content_type: str = HTML, *, parser: HtmlParser | None = None):
    return (parser or HtmlParser()).parse(body, content_type=content_type, url=URL)


def test_html_parser_satisfies_protocol() -> None:
    assert isinstance(HtmlParser(), Parser)

    class NoParse:
        pass

    assert not isinstance(NoParse(), Parser)


def test_empty_body_yields_empty_markdown() -> None:
    parsed = parse(b"")
    assert parsed.markdown == ""
    assert parsed.pages == ()
    assert parsed.title is None
    assert parsed.parser == "html"


def test_headings_h1_to_h6_become_atx() -> None:
    body = b"<h1>One</h1><h2>Two</h2><h3>Three</h3><h4>Four</h4><h5>Five</h5><h6>Six</h6>"
    assert parse(body).markdown == (
        "# One\n\n## Two\n\n### Three\n\n#### Four\n\n##### Five\n\n###### Six"
    )


def test_heading_markup_inside_is_collapsed_to_text() -> None:
    body = b"<h2>  A <em>mixed</em>   heading </h2>"
    assert parse(body).markdown == "## A mixed heading"


def test_paragraphs_are_blank_line_separated() -> None:
    body = b"<p>First   paragraph</p><p>Second\nparagraph</p>"
    assert parse(body).markdown == "First paragraph\n\nSecond paragraph"


def test_whitespace_and_newlines_collapse() -> None:
    body = b"<div>\n  alpha \t beta\n  gamma\n</div>"
    assert parse(body).markdown == "alpha beta gamma"


def test_nested_lists_are_indented() -> None:
    body = (
        b"<ul><li>alpha</li><li>beta<ul><li>beta-one</li><li>beta-two</li></ul></li>"
        b"<li>gamma</li></ul>"
    )
    assert parse(body).markdown == "- alpha\n- beta\n  - beta-one\n  - beta-two\n- gamma"


def test_ordered_lists_use_dash_items() -> None:
    body = b"<ol><li>one</li><li>two<ol><li>two-a</li></ol></li></ol>"
    assert parse(body).markdown == "- one\n- two\n  - two-a"


def test_links_become_markdown_links() -> None:
    body = b'<p>See <a href="https://example.test/x">the page</a> now.</p>'
    assert parse(body).markdown == "See [the page](https://example.test/x) now."


def test_link_text_is_trimmed_but_surrounding_spaces_survive() -> None:
    body = b'<p>a<a href="/rel">  b  </a>c</p>'
    assert parse(body).markdown == "a [b](/rel) c"


def test_scriptable_link_targets_are_not_emitted_as_links() -> None:
    body = (
        b'<p><a href="javascript:alert(1)">click</a>, '
        b'<a href="java&#9;script:alert(1)">bad</a> and <a href=" /ok ">ok</a></p>'
    )
    assert parse(body).markdown == "click, bad and [ok](/ok)"


@pytest.mark.parametrize(
    "tag",
    ["script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form"],
)
def test_boilerplate_tags_are_dropped(tag: str) -> None:
    body = (
        f"<body><p>kept</p><{tag}><h2>dropped heading</h2>"
        f"<p>dropped text</p></{tag}><p>after</p></body>"
    ).encode()
    markdown = parse(body).markdown
    assert markdown == "kept\n\nafter"
    assert "dropped" not in markdown


def test_title_is_captured_and_removed_from_body() -> None:
    body = (
        b"<html><head><title>  Example &amp;  Guide </title></head><body><p>Body</p></body></html>"
    )
    parsed = parse(body)
    assert parsed.title == "Example & Guide"
    assert parsed.markdown == "Body"


def test_missing_or_empty_title_is_none() -> None:
    assert parse(b"<p>Body</p>").title is None
    assert parse(b"<title>   </title><p>Body</p>").title is None


def test_charset_parameter_decodes_body() -> None:
    body = "<p>caf\u00e9</p>".encode("latin-1")
    assert parse(body, "text/html; charset=iso-8859-1").markdown == "caf\u00e9"


def test_default_decoding_is_utf8_with_replacement() -> None:
    assert parse("<p>caf\u00e9</p>".encode()).markdown == "caf\u00e9"
    assert parse(b"<p>\xff</p>").markdown == "\ufffd"


def test_xhtml_content_type_is_accepted() -> None:
    body = b"<html><body><h1>X</h1></body></html>"
    assert parse(body, "application/xhtml+xml").markdown == "# X"


def test_oversize_body_is_rejected_before_decoding() -> None:
    parser = HtmlParser(max_bytes=10)
    with pytest.raises(ParseError, match="max_bytes"):
        parser.parse(b"<p>0123456789</p>", content_type="text/html; charset=no-such-codec", url=URL)


def test_default_byte_cap_is_five_megabytes() -> None:
    parser = HtmlParser()
    with pytest.raises(ParseError, match="max_bytes"):
        parser.parse(b"x" * 5_000_001, content_type=HTML, url=URL)


def test_unknown_charset_is_rejected() -> None:
    with pytest.raises(ParseError, match="charset"):
        parse(b"<p>x</p>", "text/html; charset=no-such-codec")


@pytest.mark.parametrize("content_type", ["application/pdf", "text/plain", "", "   "])
def test_unsupported_content_type_is_rejected(content_type: str) -> None:
    with pytest.raises(ParseError, match="content type"):
        parse(b"<p>x</p>", content_type)


def test_max_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        HtmlParser(max_bytes=0)


def test_parse_and_chunk_uses_html_heading_paths() -> None:
    body = (
        b"<html><head><title>Guide</title></head><body>"
        b"<h1>Top</h1><p>intro</p><h2>Section</h2><p>body text</p>"
        b"</body></html>"
    )
    chunks = list(parse_and_chunk(body, content_type=HTML, url=URL, parser=HtmlParser()))
    paths = [chunk.heading_path for chunk in chunks]
    assert ("Top",) in paths
    assert ("Top", "Section") in paths
