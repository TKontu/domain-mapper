"""Tests for the Parser adapter protocol and in-memory fake (AD-07)."""

from __future__ import annotations

import dataclasses

import pytest

from adapters.parser import (
    InMemoryParser,
    ParseError,
    ParsedDocument,
    ParsedPage,
    Parser,
)


def html_document() -> ParsedDocument:
    return ParsedDocument(
        markdown="# Heading\n\nBody text.",
        pages=(),
        title="Example page",
        parser="memory",
    )


def pdf_document() -> ParsedDocument:
    return ParsedDocument(
        markdown="Page one\n\nPage two",
        pages=(
            ParsedPage(number=1, markdown="Page one"),
            ParsedPage(number=2, markdown="Page two"),
        ),
        title="Example PDF",
        parser="memory",
    )


def make() -> InMemoryParser:
    return InMemoryParser({"text/html": html_document(), "application/pdf": pdf_document()})


def test_fake_satisfies_protocol() -> None:
    assert isinstance(make(), Parser)

    class NoParse:
        pass

    assert not isinstance(NoParse(), Parser)


def test_html_returns_document_without_pages() -> None:
    parsed = make().parse(b"<h1>Heading</h1>", content_type="text/html", url="https://a.example/")
    assert isinstance(parsed, ParsedDocument)
    assert parsed.markdown.startswith("# Heading")
    assert parsed.pages == ()
    assert parsed.title == "Example page"
    assert parsed.parser == "memory"


def test_pdf_returns_numbered_pages() -> None:
    parsed = make().parse(
        b"%PDF-1.7", content_type="application/pdf", url="https://a.example/f.pdf"
    )
    assert [page.number for page in parsed.pages] == [1, 2]
    assert [page.markdown for page in parsed.pages] == ["Page one", "Page two"]
    assert parsed.markdown == "Page one\n\nPage two"
    assert parsed.title == "Example PDF"


def test_content_type_ignores_parameters_and_case() -> None:
    parser = make()
    assert parser.parse(b"", content_type="TEXT/HTML; charset=UTF-8", url="u") == html_document()
    assert make().parse(b"", content_type="Application/PDF", url="u") == pdf_document()


def test_unsupported_content_type_raises() -> None:
    with pytest.raises(ParseError):
        make().parse(b"", content_type="image/png", url="https://a.example/i.png")
    with pytest.raises(ParseError):
        make().parse(b"", content_type="", url="https://a.example/")
    with pytest.raises(ParseError):
        InMemoryParser({}).parse(b"", content_type="text/html", url="https://a.example/")


@pytest.mark.parametrize(
    "numbers",
    [
        (2, 3),
        (0, 1),
        (1, 1),
        (1, 3),
        (2, 1),
    ],
)
def test_non_contiguous_page_numbers_rejected(numbers: tuple[int, ...]) -> None:
    with pytest.raises(ParseError):
        ParsedDocument(
            markdown="text",
            pages=tuple(ParsedPage(number=n, markdown=f"page {n}") for n in numbers),
            title=None,
            parser="memory",
        )


def test_parse_error_is_value_error() -> None:
    assert issubclass(ParseError, ValueError)


def test_records_frozen() -> None:
    parsed = make().parse(b"", content_type="text/html", url="u")
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.markdown = "changed"  # type: ignore[misc]
    page = pdf_document().pages[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        page.number = 2  # type: ignore[misc]
