"""Tests for heading-aware Markdown chunking (CH-01)."""

from __future__ import annotations

import pytest

from adapters.parser import ParsedDocument, ParsedPage
from core.chunking import Chunk, chunk_document


def make_document(markdown: str, pages: tuple[ParsedPage, ...] = ()) -> ParsedDocument:
    return ParsedDocument(markdown=markdown, pages=pages, title=None, parser="test")


def sources(document: ParsedDocument) -> dict[int | None, str]:
    if document.pages:
        return {page.number: page.markdown for page in document.pages}
    return {None: document.markdown}


def assert_offsets(document: ParsedDocument, chunks: list[Chunk]) -> None:
    """Every chunk's text must be the exact slice it claims in its source."""
    by_page = sources(document)
    for chunk in chunks:
        assert by_page[chunk.page][chunk.char_start : chunk.char_end] == chunk.text


def assert_sequence(chunks: list[Chunk]) -> None:
    assert [chunk.seq for chunk in chunks] == list(range(len(chunks)))


def test_chunk_is_a_frozen_dataclass() -> None:
    chunk = Chunk(
        seq=0,
        text="text",
        page=None,
        heading_path=("A",),
        char_start=0,
        char_end=4,
    )
    assert chunk.heading_path == ("A",)
    with pytest.raises(AttributeError):
        chunk.text = "other"  # type: ignore[misc]


def test_nested_heading_paths_and_no_merging() -> None:
    document = make_document(
        "# A\n\nintro under a\n\n## B\n\ntext under b\n\n### C\n\ntext under c\n"
    )
    chunks = list(chunk_document(document))

    assert [chunk.heading_path for chunk in chunks] == [
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
    ]
    assert [chunk.text for chunk in chunks] == [
        "intro under a",
        "text under b",
        "text under c",
    ]
    assert [chunk.page for chunk in chunks] == [None, None, None]
    assert_sequence(chunks)
    assert_offsets(document, chunks)


def test_preamble_before_first_heading_has_empty_path() -> None:
    document = make_document("preamble text\n\n# A\n\nbody text\n")
    chunks = list(chunk_document(document))

    assert [chunk.heading_path for chunk in chunks] == [(), ("A",)]
    assert [chunk.text for chunk in chunks] == ["preamble text", "body text"]


def test_pages_keep_numbers_and_per_page_offsets() -> None:
    page_one = "# One\n\nalpha beta gamma\n"
    page_two = "## Two\n\ndelta epsilon\n"
    document = make_document(
        markdown="this markdown is ignored when pages are present",
        pages=(
            ParsedPage(number=1, markdown=page_one),
            ParsedPage(number=2, markdown=page_two),
        ),
    )
    chunks = list(chunk_document(document))

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.heading_path for chunk in chunks] == [("One",), ("Two",)]
    assert [chunk.text for chunk in chunks] == ["alpha beta gamma", "delta epsilon"]
    assert chunks[0].char_start == page_one.index("alpha")
    assert chunks[1].char_start == page_two.index("delta")
    assert_sequence(chunks)
    assert_offsets(document, chunks)


def test_a_chunk_never_spans_a_heading_boundary() -> None:
    document = make_document("# A\n\nalpha\n\n# B\n\nbeta\n")
    chunks = list(chunk_document(document))

    assert all("# B" not in chunk.text and "beta" not in chunk.text for chunk in chunks[:1])
    assert chunks[0].text == "alpha"
    assert chunks[1].text == "beta"


def test_oversized_section_splits_with_word_overlap() -> None:
    paragraphs = [" ".join(f"p{index}w{word}" for word in range(30)) for index in range(6)]
    markdown = "# Long\n\n" + "\n\n".join(paragraphs) + "\n"
    document = make_document(markdown)

    chunks = list(chunk_document(document, max_tokens=50, min_tokens=20, overlap_tokens=10))

    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)
    assert all(chunk.heading_path == ("Long",) for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        previous_words = previous.text.split()
        following_words = following.text.split()
        assert previous_words[-10:] == following_words[:10]
    assert_sequence(chunks)
    assert_offsets(document, chunks)


def test_oversized_paragraph_splits_at_word_boundaries() -> None:
    paragraph = " ".join(f"word{index}" for index in range(130))
    document = make_document("# Big\n\n" + paragraph + "\n")

    chunks = list(chunk_document(document, max_tokens=50, min_tokens=20, overlap_tokens=10))

    assert len(chunks) == 3
    assert [len(chunk.text.split()) for chunk in chunks] == [50, 50, 50]
    assert_sequence(chunks)
    assert_offsets(document, chunks)


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_headings_inside_fenced_code_are_not_headings(fence: str) -> None:
    markdown = f"# Real\n\ntext before\n\n{fence}\n# not a heading\n{fence}\n\ntext after\n"
    document = make_document(markdown)

    chunks = list(chunk_document(document))

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Real",)
    assert "# not a heading" in chunks[0].text
    assert "text after" in chunks[0].text
    assert_offsets(document, chunks)


def test_empty_and_whitespace_only_documents_have_no_chunks() -> None:
    assert list(chunk_document(make_document(""))) == []
    assert list(chunk_document(make_document("   \n\n\t\n \n"))) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": -5},
        {"overlap_tokens": -1},
        {"overlap_tokens": 600},
        {"max_tokens": 100, "overlap_tokens": 100},
        {"max_tokens": 100, "min_tokens": 101},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        chunk_document(make_document("some text"), **kwargs)


def test_offsets_hold_for_a_mixed_document() -> None:
    document = make_document(
        "preamble\n\n"
        "# A\n\nfirst paragraph\n\nsecond paragraph\n\n"
        "## B\n\n```\n# code\nbody\n```\n\n"
        "# C\n\nlast words here\n"
    )
    chunks = list(chunk_document(document))

    assert len(chunks) == 4
    assert_offsets(document, chunks)
    assert_sequence(chunks)
