"""Stdlib HTML to Markdown parser (AD-10).

ARCHITECTURE.md §4 (Parse & chunk) turns fetched HTML into clean Markdown with
its headings kept. :class:`HtmlParser` is the first real implementation of the
:class:`~adapters.parser.Parser` protocol: it converts one HTML body to Markdown
using only the standard library's :mod:`html.parser`.

Conversion:

* ``h1``..``h6`` become ATX headings (``#``..``######``).
* Paragraphs and other block elements are separated by a blank line.
* List items become ``- `` items; nested lists are indented two spaces per
  level. Ordered lists use the same dash marker (the protocol carries no list
  kind, and the chunker only needs the text).
* ``a[href]`` becomes ``[text](href)``. Script-capable targets
  (``javascript:``, ``vbscript:``, ``data:``) keep their text but are not
  emitted as links.
* ``script``, ``style``, ``noscript``, ``template``, ``nav``, ``header``,
  ``footer``, ``aside`` and ``form`` are dropped with their content.
* ``title`` supplies :attr:`ParsedDocument.title` and is removed from the body.
* Runs of whitespace collapse to one space; ``br`` becomes a line break.
* Table cell text is flattened into paragraphs; no table Markdown is produced.

Caps and decoding:

* Bodies larger than ``max_bytes`` (default 5,000,000) raise
  :class:`~adapters.parser.ParseError` before any decoding.
* Only ``text/html`` and ``application/xhtml+xml`` are accepted; other media
  types raise ``ParseError``.
* Decoding uses the ``charset=`` parameter when present, else UTF-8, with
  ``errors="replace"``. An unknown codec raises ``ParseError``.
* ``pages`` is always empty; ``parser`` is ``"html"``.

The class is **unwired by design**: the CLI ``parse`` stage row wires it via
:func:`core.parse_chunk.parse_and_chunk`.

Scale: one body is held in memory, bounded by ``max_bytes``; the parsed
Markdown is proportional to the decoded input.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from adapters.parser import ParsedDocument, ParseError

DEFAULT_MAX_BYTES = 5_000_000

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_DROPPED_TAGS = frozenset(
    {"script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form"}
)
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "body",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "hr",
        "html",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
    }
)
_UNSAFE_SCHEMES = ("javascript:", "vbscript:", "data:")
_SPACE_RUN = re.compile(r"\s+")
_HREF_NOISE = re.compile(r"[\s\x00-\x1f\x7f]+")
_NEWLINE_PADDING = re.compile(r" *\n *")


def _collapse(text: str) -> str:
    """Collapse whitespace inside one inline span, keeping line breaks."""
    text = _SPACE_RUN.sub(" ", text)
    text = _NEWLINE_PADDING.sub("\n", text)
    return text.strip()


def _is_safe_href(href: str) -> bool:
    """Reject hrefs whose scheme can execute script when rendered."""
    normalised = _HREF_NOISE.sub("", href).lower()
    return not normalised.startswith(_UNSAFE_SCHEMES)


def _split_content_type(content_type: str) -> tuple[str, str | None]:
    """Return ``(media_type, charset)`` from a content type header value."""
    if not isinstance(content_type, str):
        raise ParseError("content type must be a string")
    pieces = content_type.split(";")
    media_type = pieces[0].strip().lower()
    if not media_type:
        raise ParseError("unsupported content type: content type must be non-blank")
    charset: str | None = None
    for piece in pieces[1:]:
        name, separator, value = piece.partition("=")
        if separator and name.strip().lower() == "charset":
            candidate = value.strip().strip('"').strip("'")
            if candidate:
                charset = candidate
    return media_type, charset


class _MarkdownBuilder(HTMLParser):
    """Collect ATX Markdown blocks from one HTML body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._inline = ""
        self._heading: int | None = None
        self._list_level: int | None = None
        self._list_depth = 0
        self._title_parts: list[str] = []
        self._title: str | None = None
        self._in_title = False
        self._in_head = False
        self._drop_depth = 0
        self._links: list[tuple[int, str]] = []

    @property
    def title(self) -> str | None:
        return self._title

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._in_title:
            return
        if tag in _DROPPED_TAGS:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag == "title":
            self._in_title = True
            self._title_parts = []
            return
        if tag == "head":
            self._in_head = True
            return
        if self._in_head:
            return
        if tag in _HEADINGS:
            self._flush_pending()
            self._heading = _HEADINGS[tag]
            return
        if tag in {"ul", "ol"}:
            self._flush_pending()
            self._list_depth += 1
            return
        if tag == "li":
            self._flush_pending()
            self._list_level = self._list_depth
            return
        if tag == "br":
            self._inline += "\n"
            return
        if tag == "a":
            href = dict(attrs).get("href")
            if href is not None and _is_safe_href(href):
                self._links.append((len(self._inline), href.strip()))
            return
        if tag in _BLOCK_TAGS:
            self._flush_pending()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._in_title:
            self._in_title = False
            title = _collapse("".join(self._title_parts))
            self._title = title or None
            return
        if self._in_title:
            return
        if tag in _DROPPED_TAGS:
            if self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag == "head":
            self._in_head = False
            return
        if self._in_head:
            return
        if tag in _HEADINGS:
            self._flush_pending()
            self._heading = None
            return
        if tag in {"ul", "ol"}:
            self._flush_pending()
            if self._list_depth:
                self._list_depth -= 1
            self._list_level = self._list_depth or None
            return
        if tag == "li":
            self._flush_pending()
            self._list_level = None
            return
        if tag == "a":
            self._close_link()
            return
        if tag in _BLOCK_TAGS:
            self._flush_pending()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._drop_depth or self._in_head:
            return
        self._inline += _SPACE_RUN.sub(" ", data)

    def _close_link(self) -> None:
        if not self._links:
            return
        start, href = self._links.pop()
        raw = self._inline[start:]
        leading = raw[: len(raw) - len(raw.lstrip())]
        trailing = raw[len(raw.rstrip()) :]
        inner = raw.strip()
        if inner:
            prefix = " " if leading else ""
            suffix = " " if trailing else ""
            replacement = f"{prefix}[{inner}]({href}){suffix}"
        else:
            replacement = " " if leading or trailing else ""
        self._inline = self._inline[:start] + replacement

    def _flush_pending(self) -> None:
        raw = self._inline
        self._inline = ""
        text = _collapse(raw)
        if not text:
            return
        if self._heading is not None:
            self.blocks.append(("heading", f"{'#' * self._heading} {text}"))
        elif self._list_level is not None:
            indent = "  " * (self._list_level - 1)
            body = ("\n" + indent + "  ").join(text.split("\n"))
            self.blocks.append(("item", f"{indent}- {body}"))
        else:
            self.blocks.append(("paragraph", text))

    def markdown(self) -> str:
        self._flush_pending()
        parts: list[str] = []
        previous: str | None = None
        for kind, text in self.blocks:
            if parts:
                parts.append("\n" if kind == "item" and previous == "item" else "\n\n")
            parts.append(text)
            previous = kind
        return "".join(parts)


class HtmlParser:
    """Convert one HTML body to Markdown without external dependencies."""

    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes

    def parse(self, body: bytes, *, content_type: str, url: str) -> ParsedDocument:
        media_type, charset = _split_content_type(content_type)
        if media_type not in _HTML_TYPES:
            raise ParseError(f"unsupported content type: {content_type!r}")
        if len(body) > self._max_bytes:
            raise ParseError(f"body exceeds max_bytes ({self._max_bytes})")
        codec = charset or "utf-8"
        try:
            text = body.decode(codec, errors="replace")
        except LookupError:
            raise ParseError(f"unknown charset: {charset!r}") from None
        builder = _MarkdownBuilder()
        builder.feed(text)
        builder.close()
        return ParsedDocument(
            markdown=builder.markdown(),
            pages=(),
            title=builder.title,
            parser="html",
        )
