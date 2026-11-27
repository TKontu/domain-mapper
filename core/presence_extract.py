"""Deterministic market presence extraction from stored chunks (MP-03).

A scope pack's optional ``presence_terms`` maps an uppercase alpha-2 country to literal terms.
:func:`compile_terms` builds one case-insensitive, word-boundary regex per scope;
:func:`extract_presence` streams each entity's chunks (``chunk.snapshot_id`` ->
``snapshot.entity_id``) in id order and returns one :class:`PresenceMatch` per matched
(entity, country) citing the lowest matching chunk id. No LLM, no type inference.
Production caller: ``app.cli.run_presence_extract``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.snapshot_store import SnapshotRow

_STREAM_BATCH = 500


@dataclass(frozen=True)
class PresenceMatch:
    entity_id: int
    country: str
    chunk_id: int


@dataclass(frozen=True)
class CompiledTerms:
    pattern: re.Pattern[str]
    countries: Mapping[str, tuple[str, ...]]  # casefolded term -> countries


def compile_terms(presence_terms: Mapping[str, Sequence[str]]) -> CompiledTerms:
    """Compile all terms into one regex; a term shared by countries maps to each of them."""
    by_term: dict[str, set[str]] = {}
    for country, terms in presence_terms.items():
        for term in terms:
            by_term.setdefault(term.strip().casefold(), set()).add(country)
    ordered = sorted(by_term, key=lambda t: (-len(t), t))  # longest first
    alternation = "|".join(re.escape(t) for t in ordered)
    pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)
    return CompiledTerms(pattern, {t: tuple(sorted(c)) for t, c in by_term.items()})


def extract_presence(
    session: Session, compiled: CompiledTerms, entity_ids: Iterable[int]
) -> list[PresenceMatch]:
    """Return matches sorted by (entity, country); one streamed chunk query per entity."""
    every = set().union(*compiled.countries.values()) if compiled.countries else set()
    found: list[PresenceMatch] = []
    for entity_id in entity_ids:
        first: dict[str, int] = {}
        stmt = (
            sa.select(ChunkRow.id, ChunkRow.text)
            .join(SnapshotRow, SnapshotRow.id == ChunkRow.snapshot_id)
            .where(SnapshotRow.entity_id == entity_id)
            .order_by(ChunkRow.id)
            .execution_options(yield_per=_STREAM_BATCH)
        )
        result = session.execute(stmt)
        try:
            for chunk_id, text in result:
                for match in compiled.pattern.finditer(text):
                    for country in compiled.countries.get(match.group(0).casefold(), ()):
                        first.setdefault(country, chunk_id)
                if len(first) == len(every):
                    break
        finally:
            result.close()
        found.extend(PresenceMatch(entity_id, c, first[c]) for c in sorted(first))
    return found
