"""Deterministic quote grounding: locate a claimed quote in its source chunk.

Pure functions, no I/O, no LLM, no dependencies beyond the standard library.
Ported from TKontu/raappa ``src/services/extraction/grounding.py`` at commit
6c73d9f690564230c5088b9c9c5e35dd980d2d16 (same owner; that repository has no
licence file). Only the quote-location core is ported; schema/field/entity
scoring and value checks were dropped (see docs/items/GR-01.md).

Tiers used by ``ground_and_locate`` (decreasing strictness):
1. lowercase + collapsed-whitespace substring        -> 1.0
2. additionally punctuation-stripped                 -> 0.95
3. additionally markdown-stripped                    -> 0.9
4. block/line word-overlap fuzzy match (>= 0.6)      -> overlap ratio
Offsets always refer to the original content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

_POS_WS_RE = re.compile(r"\s+")
_POS_STRIP_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_BOLD_ITALIC_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_MD_TABLE_SEP_RE = re.compile(r"^\|[\s\-:]+\|[\s\-:|]*$", re.MULTILINE)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_TRAILING_ELLIPSIS_RE = re.compile(r"\.{2,}$|…$")
_UNICODE_DASHES = str.maketrans(
    {
        "\u2013": "-",  # en-dash
        "\u2014": "-",  # em-dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
    }
)


@dataclass
class GroundingResult:
    """Result from ground_and_locate: match score + original-content positions."""

    score: float  # 0.0-1.0
    source_offset: int | None  # char position in original content
    source_end: int | None  # end char position in original content
    matched_span: str | None  # actual source text at [offset:end]
    match_tier: int  # 1-4, or 0 for unmatched


class ContentMaps(NamedTuple):
    """Pre-computed normalized content and offset map for position tracing."""

    norm_content: str
    norm_map: list[int]


def _normalize_string(s: str) -> str:
    """Normalize a string for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", s.lower().strip())


def _preprocess_quote(quote: str) -> str:
    """Clean up common LLM quoting artifacts before matching."""
    q = quote.strip()
    q = _TRAILING_ELLIPSIS_RE.sub("", q).strip()
    q = q.translate(_UNICODE_DASHES)
    return q


def _pos_normalize(s: str) -> str:
    """Lowercase, collapse whitespace, normalize dashes (for position tracing)."""
    s = s.translate(_UNICODE_DASHES)
    return _POS_WS_RE.sub(" ", s.lower().strip())


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, tracking char-to-original positions.

    Returns (normalized_text, offset_map) where offset_map[i] = index in original.
    ``str.lower()`` can expand one character into several (e.g. ``"İ"`` ->
    ``"i̇"``), so each output character gets its own entry: the map stays
    aligned with the normalized text, never with the input.
    """
    result: list[str] = []
    offset_map: list[int] = []
    text = text.translate(_UNICODE_DASHES)
    prev_space = True  # Start true to skip leading whitespace

    for i, ch in enumerate(text):
        if ch in " \t\n\r":
            if not prev_space and result:
                result.append(" ")
                offset_map.append(i)
            prev_space = True
        else:
            lowered = ch.lower()
            if lowered:
                result.append(lowered)
                offset_map.extend([i] * len(lowered))
            prev_space = False

    # Strip trailing space
    if result and result[-1] == " ":
        result.pop()
        offset_map.pop()

    return "".join(result), offset_map


def _punct_strip_with_map(text: str) -> tuple[str, list[int]]:
    """Strip punctuation, tracking positions back to input text.

    Input should already be normalized (lowercase, collapsed whitespace).
    offset_map[i] = index in input text.
    """
    result: list[str] = []
    offset_map: list[int] = []
    prev_space = False

    for i, ch in enumerate(text):
        if re.match(r"[^\w\s]", ch):
            continue
        if ch == " ":
            if not prev_space and result:
                result.append(" ")
                offset_map.append(i)
            prev_space = True
        else:
            result.append(ch)
            offset_map.append(i)
            prev_space = False

    if result and result[-1] == " ":
        result.pop()
        offset_map.pop()

    return "".join(result), offset_map


def _strip_markdown_with_map(text: str) -> tuple[str, list[int]]:
    """Strip markdown syntax, tracking positions back to original text."""
    keep = [True] * len(text)

    # Images: ![alt](url) — keep alt only
    for m in _MD_IMAGE_RE.finditer(text):
        for j in range(m.start(), m.start(1)):
            keep[j] = False
        for j in range(m.end(1), m.end()):
            keep[j] = False

    # Links: [text](url) — keep text only
    for m in _MD_LINK_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "!":
            continue  # Already handled as image
        keep[m.start()] = False
        for j in range(m.end(1), m.end()):
            keep[j] = False

    # Bold/italic markers
    for m in _MD_BOLD_ITALIC_RE.finditer(text):
        marker_len = len(m.group(1))
        for j in range(m.start(), m.start() + marker_len):
            keep[j] = False
        for j in range(m.end() - marker_len, m.end()):
            keep[j] = False

    # Table separator rows
    for m in _MD_TABLE_SEP_RE.finditer(text):
        for j in range(m.start(), m.end()):
            keep[j] = False

    # Inline code backticks
    for m in _MD_INLINE_CODE_RE.finditer(text):
        keep[m.start()] = False
        keep[m.end() - 1] = False

    result: list[str] = []
    offset_map: list[int] = []
    for i, ch in enumerate(text):
        if not keep[i]:
            continue
        if ch == "|":
            result.append(" ")
        else:
            result.append(ch)
        offset_map.append(i)

    return "".join(result), offset_map


def _compose_maps(map_a: list[int], map_b: list[int]) -> list[int]:
    """Compose two offset maps: result[i] = map_a[map_b[i]]."""
    return [map_a[j] if j < len(map_a) else map_a[-1] for j in map_b]


def _tier1_locate(
    norm_quote: str,
    norm_content: str,
    norm_map: list[int],
) -> GroundingResult | None:
    """Tier 1: Normalized substring match with position."""
    pos = norm_content.find(norm_quote)
    if pos < 0:
        return None
    end = pos + len(norm_quote)
    orig_start = norm_map[pos] if pos < len(norm_map) else 0
    orig_end = (norm_map[end - 1] + 1) if end - 1 < len(norm_map) else len(norm_map)
    return GroundingResult(
        score=1.0,
        source_offset=orig_start,
        source_end=orig_end,
        matched_span=None,
        match_tier=1,
    )


def _tier2_locate(
    norm_quote_stripped: str,
    norm_content: str,
    norm_map: list[int],
) -> GroundingResult | None:
    """Tier 2: Punct-stripped match with position tracking."""
    content_stripped, punct_map = _punct_strip_with_map(norm_content)
    pos = content_stripped.find(norm_quote_stripped)
    if pos < 0:
        return None
    end = pos + len(norm_quote_stripped)
    composed = _compose_maps(norm_map, punct_map)
    orig_start = composed[pos] if pos < len(composed) else 0
    orig_end = (composed[end - 1] + 1) if end - 1 < len(composed) else 0
    return GroundingResult(
        score=0.95,
        source_offset=orig_start,
        source_end=orig_end,
        matched_span=None,
        match_tier=2,
    )


def _tier3_locate(
    norm_quote_stripped: str,
    content: str,
) -> GroundingResult | None:
    """Tier 3: Markdown-stripped + punct-stripped with position tracking."""
    md_stripped, md_map = _strip_markdown_with_map(content)
    md_norm, md_norm_map = _normalize_with_map(md_stripped)
    md_punct, md_punct_map = _punct_strip_with_map(md_norm)

    pos = md_punct.find(norm_quote_stripped)
    if pos < 0:
        return None
    end = pos + len(norm_quote_stripped)

    # Compose: md_punct → md_norm → md_stripped → original
    map_to_md_norm = _compose_maps(md_norm_map, md_punct_map) if md_punct_map else md_norm_map
    map_to_original = _compose_maps(md_map, map_to_md_norm) if map_to_md_norm else md_map

    orig_start = map_to_original[pos] if pos < len(map_to_original) else 0
    orig_end = (map_to_original[end - 1] + 1) if end - 1 < len(map_to_original) else 0
    return GroundingResult(
        score=0.9,
        source_offset=orig_start,
        source_end=orig_end,
        matched_span=None,
        match_tier=3,
    )


def _tier4_locate(
    norm_quote: str,
    content: str,
    threshold: float = 0.6,
) -> GroundingResult | None:
    """Tier 4: Block-level fuzzy matching with position tracking."""
    quote_words = norm_quote.split()
    if not quote_words:
        return None

    quote_set = set(quote_words)
    blocks = content.split("\n\n")

    best_overlap = 0.0
    best_block_idx = -1
    best_block = ""

    block_positions: list[int] = []
    char_pos = 0
    for i, block in enumerate(blocks):
        block_positions.append(char_pos)
        char_pos += len(block) + 2

        stripped = _strip_markdown_with_map(block)[0]
        norm_block = _pos_normalize(stripped)
        block_words = set(norm_block.split())
        if not block_words:
            continue

        overlap = len(quote_set & block_words) / len(quote_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_block_idx = i
            best_block = block

    if best_overlap < threshold or best_block_idx < 0:
        return None

    # Try line-level within winning block
    lines = best_block.split("\n")
    best_line_overlap = 0.0
    best_line_offset = block_positions[best_block_idx]
    best_line_text = best_block

    line_pos = block_positions[best_block_idx]
    for line in lines:
        stripped_line = _strip_markdown_with_map(line)[0]
        norm_line = _pos_normalize(stripped_line)
        line_words = set(norm_line.split())
        if line_words:
            line_overlap = len(quote_set & line_words) / len(quote_set)
            if line_overlap > best_line_overlap:
                best_line_overlap = line_overlap
                best_line_offset = line_pos
                best_line_text = line
        line_pos += len(line) + 1

    if best_line_overlap >= threshold:
        return GroundingResult(
            score=best_line_overlap,
            source_offset=best_line_offset,
            source_end=best_line_offset + len(best_line_text),
            matched_span=best_line_text,
            match_tier=4,
        )

    return GroundingResult(
        score=best_overlap,
        source_offset=block_positions[best_block_idx],
        source_end=block_positions[best_block_idx] + len(best_block),
        matched_span=best_block[:200],
        match_tier=4,
    )


def _locate(quote: str, content: str, norm_content: str, norm_map: list[int]) -> GroundingResult:
    """Run tiers 1-4 against original content and its normalised map."""
    if not quote or not content:
        return GroundingResult(0.0, None, None, None, 0)

    clean_quote = _preprocess_quote(quote)
    if not clean_quote:
        return GroundingResult(0.0, None, None, None, 0)

    norm_quote = _pos_normalize(clean_quote)
    if not norm_quote:
        return GroundingResult(0.0, None, None, None, 0)

    # Tier 1: Normalized substring
    result = _tier1_locate(norm_quote, norm_content, norm_map)

    norm_quote_stripped = _POS_STRIP_PUNCT_RE.sub("", norm_quote)
    norm_quote_stripped = _POS_WS_RE.sub(" ", norm_quote_stripped).strip()
    # Tier 2: Punct-stripped
    if result is None and norm_quote_stripped:
        result = _tier2_locate(norm_quote_stripped, norm_content, norm_map)
    # Tier 3: Markdown + punct stripped
    if result is None and norm_quote_stripped:
        result = _tier3_locate(norm_quote_stripped, content)
    if result is not None:
        result.matched_span = content[result.source_offset : result.source_end]
        return result

    # Tier 4: Block fuzzy
    result = _tier4_locate(norm_quote, content)
    if result is not None:
        return result

    return GroundingResult(0.0, None, None, None, 0)


def ground_and_locate(quote: str, content: str) -> GroundingResult:
    """Unified grounding + location: tries 4 tiers with decreasing strictness."""
    if not quote or not content:
        return GroundingResult(0.0, None, None, None, 0)
    norm_content, norm_map = _normalize_with_map(content)
    return _locate(quote, content, norm_content, norm_map)


def precompute_content_maps(content: str) -> ContentMaps:
    """Pre-compute normalized content and offset map for reuse across quotes."""
    norm_content, norm_map = _normalize_with_map(content)
    return ContentMaps(norm_content=norm_content, norm_map=norm_map)


def ground_and_locate_precomputed(
    quote: str,
    content: str,
    content_maps: ContentMaps,
) -> GroundingResult:
    """Like ground_and_locate but reuses pre-computed content maps."""
    return _locate(quote, content, content_maps.norm_content, content_maps.norm_map)


# Punctuation to strip for lenient matching (keep alphanumeric + spaces)
_STRIP_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def verify_quote_in_source(quote: str, source_content: str) -> float:
    """Check if a claimed quote actually exists in the source content.

    Uses multi-tier matching with increasing leniency:
    1. Normalized substring (lowercase, collapsed whitespace) → 1.0
    2. Punctuation-stripped substring → 0.95
    3. Word-level sliding window → best overlap ratio

    Args:
        quote: The claimed quote string.
        source_content: The full source text (content or cleaned_content).

    Returns:
        Similarity score 0.0-1.0. ≥0.8 means the quote is source-grounded.
    """
    if not quote or not source_content:
        return 0.0

    norm_quote = _normalize_string(quote)
    norm_content = _normalize_string(source_content)

    if not norm_quote:
        return 0.0

    # Tier 1: exact normalized substring
    if norm_quote in norm_content:
        return 1.0

    # Tier 2: strip punctuation and retry
    stripped_quote = _STRIP_PUNCT_RE.sub("", norm_quote)
    stripped_quote = re.sub(r"\s+", " ", stripped_quote).strip()
    stripped_content = _STRIP_PUNCT_RE.sub("", norm_content)
    stripped_content = re.sub(r"\s+", " ", stripped_content).strip()

    if stripped_quote and stripped_quote in stripped_content:
        return 0.95

    # Tier 3: word-level sliding window
    return _word_window_similarity(stripped_quote, stripped_content)


def _word_window_similarity(quote: str, content: str) -> float:
    """Slide a word window over content and find best overlap with quote.

    For a quote of N words, slides an N-word window across content and
    computes what fraction of quote words match at each position.

    Returns best ratio found (0.0-1.0).
    """
    quote_words = quote.split()
    content_words = content.split()

    if not quote_words or not content_words:
        return 0.0

    n = len(quote_words)
    if n > len(content_words):
        return 0.0

    quote_set = set(quote_words)
    best = 0.0

    # Build initial window
    window_counts: dict[str, int] = {}
    for w in content_words[:n]:
        window_counts[w] = window_counts.get(w, 0) + 1

    # Count matches in initial window
    matching = sum(1 for w in quote_words if window_counts.get(w, 0) > 0)
    best = matching / n

    # Early exit if perfect
    if best >= 0.95:
        return best

    # Slide window across content
    for i in range(1, len(content_words) - n + 1):
        # Remove word leaving the window
        leaving = content_words[i - 1]
        window_counts[leaving] -= 1
        if window_counts[leaving] == 0:
            del window_counts[leaving]

        # Add word entering the window
        entering = content_words[i + n - 1]
        window_counts[entering] = window_counts.get(entering, 0) + 1

        # Only recount if entering/leaving words are in the quote
        if leaving in quote_set or entering in quote_set:
            matching = sum(1 for w in quote_words if window_counts.get(w, 0) > 0)
            ratio = matching / n
            if ratio > best:
                best = ratio
                if best >= 0.95:
                    return best

    return round(best, 4)


_NEGATION_RE = re.compile(
    r"^(no|not|n/?a|none)\b.{0,50}"
    r"(mention|explicit|specified|found|available|provided|information|data|details|certif)",
    re.IGNORECASE,
)


def is_negation_quote(quote: str | None) -> bool:
    """Check if a quote is a negation (LLM saying 'not found' rather than quoting source).

    Returns True for quotes like "No mention of...", "N/A", "Not specified".
    These indicate the LLM fabricated a 'not found' response rather than
    quoting actual source text.
    """
    if not quote:
        return False
    return bool(_NEGATION_RE.match(quote.strip()))
