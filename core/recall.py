"""LLM-recall Discover channel (DS-03; ARCHITECTURE.md §4 stage 1, §6).

One structured LLM call (``purpose="recall"``) asks for providers of one scope segment in one
country. The model's output is untrusted data: it is schema-checked field by field, only
http(s) URLs are kept, duplicates by URL are dropped and the result is capped at ``limit``.
Candidates are tagged ``channel="llm_recall"`` — an unverified channel; Resolve or another
channel confirms them later. Pure: no database access; the caller stores the rows.

Production caller: ``landscape recall`` (``--limit`` <= 100). Consumer: ``landscape resolve``
via ``candidate`` rows. No page text enters the prompt, so no neutralisation is needed.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit

from core.candidate_store import NewCandidate
from core.scope_pack import ScopePack


class _Result(Protocol):
    @property
    def parsed(self) -> dict[str, object] | None: ...


class RecallLLM(Protocol):
    """The slice of ``adapters.llm.LLM`` recall needs (core does not import adapters)."""

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> _Result: ...


CHANNEL = "llm_recall"
PURPOSE = "recall"
MAX_LIMIT = 100

RECALL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "url": {"type": "string"}},
                "required": ["name", "url"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


class RecallError(ValueError):
    """The model's output violated the recall schema; the message names the field only."""


def scope_countries(pack: ScopePack) -> set[str]:
    """Concrete country codes listed by the scope's layers (the ``*`` wildcard excluded)."""
    return {c for layer in pack.scope.layers for c in layer.countries if c != "*"}


def validate_request(pack: ScopePack, *, segment: str, country: str, limit: int) -> None:
    """Raise ``ValueError`` for a blank/unknown segment or country or a limit outside 1..100."""
    if not segment.strip():
        raise ValueError("--segment must be non-blank")
    if not country.strip():
        raise ValueError("--country must be non-blank")
    if segment not in {s.id for s in pack.scope.segments}:
        raise ValueError(f"unknown segment {segment!r}")
    if country not in scope_countries(pack):
        raise ValueError(f"country {country!r} is not in the scope pack")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"--limit must be between 1 and {MAX_LIMIT}")


def query_label(segment: str, country: str) -> str:
    """The ``candidate.query`` value recording which request produced a row."""
    return f"recall segment={segment} country={country}"


def build_prompt(pack: ScopePack, *, segment: str, country: str, limit: int) -> str:
    """Render the recall request for one segment and country."""
    seg = next(s for s in pack.scope.segments if s.id == segment)
    return (
        f"List up to {limit} real organisations that provide the following in country {country}.\n"
        f"Domain: {pack.scope.title}\n"
        f"Segment id: {seg.id}\n"
        f"Segment: {seg.label}\n"
        f"Definition: {seg.definition}\n\n"
        "Your answer is treated as an unverified hint: every entry will be checked against "
        "other sources before use.\n"
        "Give each organisation's name and its official website URL (http or https).\n"
        "Do not invent URLs or organisations; omit an entry if you do not know its website.\n"
        'Reply with JSON only: {"candidates": [{"name": "...", "url": "..."}]}'
    )


def _is_http_url(url: str) -> bool:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


_NAME_MAX = 200


def _clean_name(name: str) -> str:
    """Collapse whitespace/control runs (Cc, Cf, Zl, Zp, Zs) to one space; strip; cap."""
    chars = [
        " " if ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch
        for ch in name
    ]
    collapsed = " ".join("".join(chars).split())
    return collapsed[:_NAME_MAX].rstrip()


def _parse(parsed: object) -> list[tuple[str, str]]:
    if not isinstance(parsed, dict):
        raise RecallError("response must be a JSON object")
    items = parsed.get("candidates")
    if not isinstance(items, list):
        raise RecallError("candidates must be a list")
    out: list[tuple[str, str]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise RecallError(f"candidates[{i}] must be an object")
        name, url = item.get("name"), item.get("url")
        if not isinstance(name, str):
            raise RecallError(f"candidates[{i}].name must be a string")
        if not isinstance(url, str):
            raise RecallError(f"candidates[{i}].url must be a string")
        out.append((_clean_name(name), url.strip()))
    return out


def recall_candidates(
    llm: RecallLLM,
    pack: ScopePack,
    *,
    segment: str,
    country: str,
    limit: int,
    run_id: int,
    now: datetime | None = None,
) -> list[NewCandidate]:
    """Make one recall call and return at most ``limit`` deduplicated ``llm_recall`` rows."""
    validate_request(pack, segment=segment, country=country, limit=limit)
    prompt = build_prompt(pack, segment=segment, country=country, limit=limit)
    result = llm.complete(prompt, purpose=PURPOSE, run_id=run_id, json_schema=RECALL_SCHEMA)
    pairs = _parse(result.parsed)
    found_at = (now or datetime.now(UTC)).astimezone(UTC)
    query = query_label(segment, country)
    seen: set[str] = set()
    rows: list[NewCandidate] = []
    for name, url in pairs:
        if not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        rows.append(
            NewCandidate(
                name=name or url,
                url=url,
                channel=CHANNEL,
                query=query,
                found_at=found_at,
                country_hint=country,
                segment_hint=segment,
            )
        )
        if len(rows) >= limit:
            break
    return rows
