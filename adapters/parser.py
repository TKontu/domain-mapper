"""Parser adapter interface for the Parse & chunk stage (AD-07).

ARCHITECTURE.md §4 (Parse & chunk) turns a fetched body into clean Markdown:
HTML keeps its headings, and PDFs go through MinerU to Markdown with page
numbers and tables preserved. This module defines only the interface:
:class:`Parser` is the protocol the pipeline talks to, :class:`ParsedDocument`
and :class:`ParsedPage` are its frozen records, and :class:`ParseError` is the
failure type. :class:`InMemoryParser` is a fake for tests and the example scope.

No real HTML converter or MinerU client lives here (non-goals of AD-07); those
arrive with the first implementation of this interface.

Records: ``pages`` is empty for HTML and holds 1-based, contiguous page numbers
(``1..n``) for paginated sources such as PDFs; a document whose pages are not
contiguous is rejected at construction. ``pages`` must be a ``tuple``; a list,
generator or other sequence raises :class:`ParseError` (AD-09) so the frozen
record never holds a mutable or one-shot sequence. Lists are not coerced.

Bound: :meth:`Parser.parse` holds one document's Markdown and pages in memory.
A real implementation must cap input size and page count before converting.

The module is **unwired by design**: the first production caller is the
Phase 1 Parse & chunk row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ParseError(ValueError):
    """A body that cannot be parsed, or an unsupported content type."""


@dataclass(frozen=True)
class ParsedPage:
    """One 1-based page of a paginated document, as Markdown."""

    number: int
    markdown: str


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed body: full Markdown plus per-page Markdown for PDFs."""

    markdown: str
    pages: tuple[ParsedPage, ...]
    title: str | None
    parser: str

    def __post_init__(self) -> None:
        if not isinstance(self.pages, tuple):
            raise ParseError("pages must be a tuple of ParsedPage records")
        for page in self.pages:
            if not isinstance(page, ParsedPage):
                raise ParseError("pages must contain ParsedPage records")
        numbers = [page.number for page in self.pages]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ParseError("page numbers must be contiguous 1..n")


@runtime_checkable
class Parser(Protocol):
    """A document parser used by the Parse & chunk stage."""

    def parse(self, body: bytes, *, content_type: str, url: str) -> ParsedDocument: ...


def _normalise_content_type(content_type: str) -> str:
    """Lower-case a media type and drop its parameters (e.g. ``; charset=``)."""
    if not isinstance(content_type, str):
        raise ParseError("content type must be a string")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type:
        raise ParseError("content type must be non-blank")
    return media_type


class InMemoryParser:
    """Fake :class:`Parser` over canned documents keyed by content type.

    Matching ignores parameters and case: ``text/html; charset=utf-8`` finds
    the ``text/html`` entry. Any other content type raises :class:`ParseError`.
    """

    def __init__(self, documents: Mapping[str, ParsedDocument]) -> None:
        self._documents = {
            _normalise_content_type(content_type): document
            for content_type, document in documents.items()
        }

    def parse(self, body: bytes, *, content_type: str, url: str) -> ParsedDocument:
        try:
            return self._documents[_normalise_content_type(content_type)]
        except KeyError:
            raise ParseError(f"unsupported content type: {content_type!r}") from None


def _conforms() -> Parser:
    """Static check: mypy rejects this if the fake drifts from the protocol."""
    return InMemoryParser({})
