"""Tests for core.grounding (GR-01). All fixtures are neutral invented text (D-003)."""

from __future__ import annotations

import pytest
from core.grounding import (
    GroundingResult,
    ground_and_locate,
    ground_and_locate_precomputed,
    is_negation_quote,
    precompute_content_maps,
    verify_quote_in_source,
)

CONTENT = (
    "# Garden notes\n\n"
    "The blue   kettle sits on the\nwooden shelf.\n\n"
    "Visitors may read the **quiet hour** schedule in the [lobby guide](http://example.test/g) for details.\n\n"
    "Lanterns, ribbons, and paper boats; all made by hand!\n\n"
    "Opening time: 9 to 5 — every weekday"
)


def _assert_span_covers(result: GroundingResult, content: str, words: list[str]) -> None:
    assert result.source_offset is not None and result.source_end is not None
    span = content[result.source_offset : result.source_end]
    assert span == result.matched_span
    low = span.lower()
    for w in words:
        assert w.lower() in low


def test_tier1_exact() -> None:
    r = ground_and_locate("wooden shelf", CONTENT)
    assert r.match_tier == 1 and r.score == 1.0
    _assert_span_covers(r, CONTENT, ["wooden shelf"])
    assert CONTENT[r.source_offset : r.source_end] == "wooden shelf"  # type: ignore[misc]


def test_tier1_whitespace_and_case() -> None:
    r = ground_and_locate("THE BLUE KETTLE sits on the wooden", CONTENT)
    assert r.match_tier == 1 and r.score == 1.0
    _assert_span_covers(r, CONTENT, ["blue", "kettle", "wooden"])
    assert r.matched_span == "The blue   kettle sits on the\nwooden"


def test_tier2_punctuation_stripped() -> None:
    r = ground_and_locate("lanterns ribbons and paper boats all made by hand", CONTENT)
    assert r.match_tier == 2 and r.score == 0.95
    _assert_span_covers(r, CONTENT, ["lanterns", "hand"])


def test_tier3_markdown_stripped() -> None:
    r = ground_and_locate("lobby guide for details", CONTENT)
    assert r.match_tier == 3 and r.score == 0.9
    _assert_span_covers(r, CONTENT, ["lobby guide", "details"])


def test_tier4_fuzzy_block() -> None:
    r = ground_and_locate("blue kettle sits on the metal", CONTENT)
    assert r.match_tier == 4
    assert 0.6 <= r.score < 1.0
    assert r.score == pytest.approx(5 / 6)
    _assert_span_covers(r, CONTENT, ["blue", "kettle", "sits"])


def test_unicode_dash_and_trailing_ellipsis() -> None:
    r = ground_and_locate("9 to 5 - every weekday...", CONTENT)
    assert r.match_tier == 1
    _assert_span_covers(r, CONTENT, ["9 to 5", "weekday"])


def test_no_match() -> None:
    r = ground_and_locate("submarine orchestra plays jazz", CONTENT)
    assert r == GroundingResult(0.0, None, None, None, 0)


@pytest.mark.parametrize("quote", ["", "   ", "...", None])
def test_empty_or_none_quote(quote: str | None) -> None:
    r = ground_and_locate(quote, CONTENT)  # type: ignore[arg-type]
    assert r.match_tier == 0 and r.score == 0.0 and r.source_offset is None


def test_empty_content() -> None:
    assert ground_and_locate("anything", "").match_tier == 0


@pytest.mark.parametrize(
    "quote",
    [
        "wooden shelf",
        "THE BLUE KETTLE sits on the wooden",
        "lanterns ribbons and paper boats all made by hand",
        "lobby guide for details",
        "blue kettle sits on the metal",
        "submarine orchestra plays jazz",
        "",
    ],
)
def test_precomputed_equals_plain(quote: str) -> None:
    maps = precompute_content_maps(CONTENT)
    assert ground_and_locate_precomputed(quote, CONTENT, maps) == ground_and_locate(quote, CONTENT)


EXPANDING = "İİİ x y z target"


def test_expanding_lowercase_keeps_span_plain() -> None:
    r = ground_and_locate("target", EXPANDING)
    assert r.match_tier == 1
    assert r.source_offset is not None and r.source_end is not None
    assert EXPANDING[r.source_offset : r.source_end] == "target"
    assert r.matched_span == "target"


def test_expanding_lowercase_keeps_span_precomputed() -> None:
    maps = precompute_content_maps(EXPANDING)
    r = ground_and_locate_precomputed("target", EXPANDING, maps)
    assert r.match_tier == 1
    assert r.source_offset is not None and r.source_end is not None
    assert EXPANDING[r.source_offset : r.source_end] == "target"
    assert r.matched_span == "target"


def test_expanding_lowercase_quote_keeps_span() -> None:
    r = ground_and_locate("İİİ X Y", EXPANDING)
    assert r.match_tier == 1
    assert r.source_offset is not None and r.source_end is not None
    assert EXPANDING[r.source_offset : r.source_end] == "İİİ x y"


def test_expanding_lowercase_tier2_span() -> None:
    content = "İİİ x, y: lantern"
    r = ground_and_locate("x y lantern", content)
    assert r.match_tier == 2
    assert r.source_offset is not None and r.source_end is not None
    assert content[r.source_offset : r.source_end] == "x, y: lantern"


def test_expanding_lowercase_tier3_span() -> None:
    content = "İİİ note\n\nsee the [lobby guide](http://example.test/g) for details"
    r = ground_and_locate("lobby guide for details", content)
    assert r.match_tier == 3
    assert r.source_offset is not None and r.source_end is not None
    span = content[r.source_offset : r.source_end]
    assert span.startswith("lobby guide")
    assert span.endswith("for details")


def test_verify_quote_in_source_tiers() -> None:
    src = "The river boat, painted green, leaves at noon.\nTickets cost two coins."
    assert verify_quote_in_source("the RIVER boat,  painted", src) == 1.0
    assert verify_quote_in_source("river boat painted green leaves", src) == 0.95
    partial = verify_quote_in_source("painted green leaves at midnight", src)
    assert 0.0 < partial < 0.95
    assert partial == 0.8
    assert verify_quote_in_source("", src) == 0.0
    assert verify_quote_in_source("boat", "") == 0.0
    assert verify_quote_in_source("one two three four five six seven eight nine ten", src) < 0.5


def test_word_window_quote_longer_than_content() -> None:
    assert verify_quote_in_source("alpha beta gamma delta", "alpha beta") == 0.0


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("No mention of opening hours", True),
        ("Not specified in the text", True),
        ("N/A - no information", True),
        ("none provided", True),
        ("The kettle is blue", False),
        ("Nothing about it", False),
        ("", False),
        (None, False),
    ],
)
def test_is_negation_quote(quote: str | None, expected: bool) -> None:
    assert is_negation_quote(quote) is expected
