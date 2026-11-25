"""Parse & chunk stage function (CH-02).

ARCHITECTURE.md §4 (Parse & chunk): :func:`parse_and_chunk` parses one fetched
body with a :class:`~adapters.parser.Parser` and yields the heading-aware
chunks :func:`core.chunking.chunk_document` produces for the parsed document.
It is pure and offline and **unwired by design**: the production caller is the
later Parse & chunk persistence row (CLI ``parse`` stage storing ``chunk``
rows).

Semantics:

* Invalid chunk parameters raise :class:`ValueError` eagerly, before the
  parser is called. Validation is delegated to ``chunk_document`` so the rules
  live in one place.
* The parser is called exactly once, at call time; its
  :class:`~adapters.parser.ParseError` propagates unchanged.
* Chunks are yielded lazily; an empty parsed document yields none.

Scale: holds one parsed document (bounded by the Parser's caps).
"""

from __future__ import annotations

from collections.abc import Iterator

from adapters.parser import ParsedDocument, Parser
from core.chunking import Chunk, chunk_document

_EMPTY = ParsedDocument(markdown="", pages=(), title=None, parser="validate")


def parse_and_chunk(
    body: bytes,
    *,
    content_type: str,
    url: str,
    parser: Parser,
    max_tokens: int = 600,
    min_tokens: int = 300,
    overlap_tokens: int = 60,
) -> Iterator[Chunk]:
    """Parse ``body`` once and return an iterator over its chunks."""
    params = {
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "overlap_tokens": overlap_tokens,
    }
    chunk_document(_EMPTY, **params)  # eager validation only; nothing iterated
    document = parser.parse(body, content_type=content_type, url=url)
    return chunk_document(document, **params)
