"""Tests for the parse-and-chunk stage function (CH-02)."""

from __future__ import annotations

import pytest

from adapters.parser import InMemoryParser, ParsedDocument, ParseError, ParsedPage
from core.chunking import chunk_document
from core.parse_chunk import parse_and_chunk

URL = "https://example.test/doc"
SECTION = " ".join(f"w{i}" for i in range(700))
HTML_DOC = ParsedDocument(
    markdown=f"# Alpha\n\nintro text\n\n## Beta\n\n{SECTION}\n",
    pages=(),
    title="Alpha",
    parser="test",
)
PAGED_DOC = ParsedDocument(
    markdown="# One\n\nfirst page\n\n# Two\n\nsecond page\n",
    pages=(
        ParsedPage(number=1, markdown="# One\n\nfirst page\n"),
        ParsedPage(number=2, markdown="# Two\n\nsecond page\n"),
    ),
    title=None,
    parser="test",
)
EMPTY_DOC = ParsedDocument(markdown="", pages=(), title=None, parser="test")


class CountingParser:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document
        self.calls = 0

    def parse(self, body: bytes, *, content_type: str, url: str) -> ParsedDocument:
        self.calls += 1
        return self.document


@pytest.mark.parametrize(
    ("content_type", "document"),
    [("text/html; charset=utf-8", HTML_DOC), ("application/pdf", PAGED_DOC)],
)
def test_matches_chunk_document(content_type: str, document: ParsedDocument) -> None:
    parser = InMemoryParser({"text/html": HTML_DOC, "application/pdf": PAGED_DOC})
    got = list(parse_and_chunk(b"body", content_type=content_type, url=URL, parser=parser))
    assert got == list(chunk_document(document))
    assert got


def test_two_page_document_keeps_pages() -> None:
    parser = InMemoryParser({"application/pdf": PAGED_DOC})
    got = list(parse_and_chunk(b"%PDF", content_type="application/pdf", url=URL, parser=parser))
    assert [chunk.page for chunk in got] == [1, 2]


def test_parameters_are_forwarded() -> None:
    parser = InMemoryParser({"text/html": HTML_DOC})
    got = list(
        parse_and_chunk(
            b"x",
            content_type="text/html",
            url=URL,
            parser=parser,
            max_tokens=200,
            min_tokens=100,
            overlap_tokens=20,
        )
    )
    assert got == list(chunk_document(HTML_DOC, max_tokens=200, min_tokens=100, overlap_tokens=20))


def test_unsupported_content_type_raises_parse_error() -> None:
    parser = InMemoryParser({"text/html": HTML_DOC})
    with pytest.raises(ParseError):
        parse_and_chunk(b"x", content_type="image/png", url=URL, parser=parser)


@pytest.mark.parametrize(
    "params",
    [
        {"max_tokens": 0},
        {"overlap_tokens": -1},
        {"max_tokens": 10, "min_tokens": 5, "overlap_tokens": 10},
        {"max_tokens": 10, "min_tokens": 11, "overlap_tokens": 1},
    ],
)
def test_invalid_parameters_raise_before_parsing(params: dict[str, int]) -> None:
    parser = CountingParser(HTML_DOC)
    with pytest.raises(ValueError):
        parse_and_chunk(b"x", content_type="text/html", url=URL, parser=parser, **params)
    assert parser.calls == 0


def test_parser_called_exactly_once() -> None:
    parser = CountingParser(HTML_DOC)
    chunks = parse_and_chunk(b"x", content_type="text/html", url=URL, parser=parser)
    assert parser.calls == 1
    list(chunks)
    assert parser.calls == 1


def test_empty_document_yields_no_chunks() -> None:
    parser = CountingParser(EMPTY_DOC)
    assert list(parse_and_chunk(b"", content_type="text/html", url=URL, parser=parser)) == []
