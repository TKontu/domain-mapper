"""Heading-aware Markdown chunking for the Parse & chunk stage (CH-01).

ARCHITECTURE.md §4 (Parse & chunk) asks for heading-aware chunks of roughly
300–600 tokens with overlap, each keeping its page, heading path and character
offsets; the ``chunk`` table in ARCHITECTURE.md records ``seq``, ``text``,
``page``, ``heading_path``, ``char_start`` and ``char_end``.
:func:`chunk_document` turns a :class:`~adapters.parser.ParsedDocument` into
such chunks without I/O. It is **unwired by design**: the first production
caller is the Phase 1 Parse & chunk row.

Semantics:

* A chunk never spans an ATX heading boundary (``#``..``######``) or a page
  boundary. Each page's Markdown is chunked on its own, so the heading path
  restarts at every page.
* ``heading_path`` names the headings enclosing the chunk, outermost first.
  The heading line itself is not part of the chunk text; the path carries it.
* ``char_start``/``char_end`` index the Markdown the chunk was cut from:
  ``doc.markdown`` when ``doc.pages`` is empty, otherwise that page's
  ``markdown``. ``source[char_start:char_end] == text`` always holds.
* A token is approximated as one whitespace-split word (there is no tokenizer
  dependency); counts are therefore word counts, not model tokens.
* Sections longer than ``max_tokens`` are split at paragraph (blank-line)
  boundaries where possible, falling back to word boundaries for an oversized
  paragraph. Consecutive chunks of one section overlap by ``overlap_tokens``
  words. ``min_tokens`` is only a target for where splitting stops: small
  adjacent sections are never merged, which keeps every ``heading_path``
  exact.
* Headings inside fenced code blocks (``` or ~~~) are not headings.
  Whitespace-only spans produce no chunk; an empty document yields none.
* Invalid parameters raise :class:`ValueError` eagerly, before iteration.

Scale: a generator that holds one already-bounded document (the Parser's caps
apply) and emits chunks lazily; no further bound is imposed here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from adapters.parser import ParsedDocument

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_WORD = re.compile(r"\S+")
_BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")


@dataclass(frozen=True)
class Chunk:
    """One heading- and page-bounded slice of a document's Markdown."""

    seq: int
    text: str
    page: int | None
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int


def chunk_document(
    doc: ParsedDocument,
    *,
    max_tokens: int = 600,
    min_tokens: int = 300,
    overlap_tokens: int = 60,
) -> Iterator[Chunk]:
    """Yield heading-aware :class:`Chunk` records for ``doc``.

    ``max_tokens`` is a hard ceiling per chunk, ``min_tokens`` the preferred
    stopping point when a paragraph boundary allows it, and ``overlap_tokens``
    the word overlap between consecutive chunks of one long section. See the
    module docstring for the full semantics.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must not be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    if min_tokens > max_tokens:
        raise ValueError("min_tokens must not exceed max_tokens")
    return _iter_chunks(
        doc,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
    )


def _iter_chunks(
    doc: ParsedDocument,
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> Iterator[Chunk]:
    if doc.pages:
        sources: tuple[tuple[int | None, str], ...] = tuple(
            (page.number, page.markdown) for page in doc.pages
        )
    else:
        sources = ((None, doc.markdown),)

    seq = 0
    for page, source in sources:
        for heading_path, body_start, body_end in _sections(source):
            for chunk_start, chunk_end in _split_section(
                source,
                body_start,
                body_end,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_tokens=overlap_tokens,
            ):
                yield Chunk(
                    seq=seq,
                    text=source[chunk_start:chunk_end],
                    page=page,
                    heading_path=heading_path,
                    char_start=chunk_start,
                    char_end=chunk_end,
                )
                seq += 1


def _sections(source: str) -> Iterator[tuple[tuple[str, ...], int, int]]:
    """Yield ``(heading_path, body_start, body_end)`` per ATX section."""
    stack: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None
    section_path: tuple[str, ...] = ()
    section_start = 0
    offset = 0

    for line in source.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        fence_match = _FENCE.match(content)
        if fence is not None:
            if fence_match is not None:
                marker = fence_match.group(1)
                rest = fence_match.group(2)
                if marker[0] == fence[0] and len(marker) >= fence[1] and rest.strip() == "":
                    fence = None
            offset += len(line)
            continue
        if fence_match is not None:
            marker = fence_match.group(1)
            fence = (marker[0], len(marker))
            offset += len(line)
            continue

        heading = _ATX_HEADING.match(content)
        if heading is not None:
            yield section_path, section_start, offset
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _heading_title(heading.group(2))))
            section_start = offset + len(line)
            section_path = tuple(title for _, title in stack)
        offset += len(line)

    yield section_path, section_start, len(source)


def _heading_title(raw: str | None) -> str:
    if raw is None:
        return ""
    return _CLOSING_HASHES.sub("", raw).strip()


def _split_section(
    source: str,
    start: int,
    end: int,
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(char_start, char_end)`` windows over one section's words."""
    words = [(match.start(), match.end()) for match in _WORD.finditer(source, start, end)]
    if not words:
        return

    paragraph_breaks: set[int] = set()
    for index in range(1, len(words)):
        gap = source[words[index - 1][1] : words[index][0]]
        if _BLANK_LINE.search(gap):
            paragraph_breaks.add(index)

    total = len(words)
    index = 0
    while index < total:
        limit = min(total, index + max_tokens)
        if limit == total:
            stop = total
        else:
            stop = limit
            for candidate in range(limit, index, -1):
                if candidate in paragraph_breaks and candidate - index >= min_tokens:
                    stop = candidate
                    break
        yield words[index][0], words[stop - 1][1]
        if stop == total:
            return
        index = max(index + 1, stop - overlap_tokens)
