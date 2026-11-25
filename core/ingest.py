"""Single-page ingest: fetch → snapshot → parse → chunk rows (IN-01).

ARCHITECTURE.md §4 (Crawl & snapshot; Parse & chunk). :func:`ingest_page`
fetches one URL once, records one ``snapshot`` row
(:func:`core.snapshot_store.record_snapshot`), parses and chunks the body
(:func:`core.parse_chunk.parse_and_chunk`) and stores the chunks
(:func:`core.chunk_store.store_chunks`).

Errors:

* :class:`~adapters.fetcher.FetchError` propagates before any row is written.
* :class:`~adapters.parser.ParseError` propagates *after* the snapshot row
  exists; no chunk rows are written for it (the fetch was a real observation).

Scale: one body in memory (bounded by the fetcher's byte cap); chunks stream
lazily into ``store_chunks``, never materialised as a list.

**Unwired by design**: the CLI ``crawl``/``parse`` stage row is the production
caller. No crawl policy (robots, rate limits) or raw-byte storage lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from adapters.fetcher import Fetcher
from adapters.parser import Parser
from core.chunk_store import store_chunks
from core.parse_chunk import parse_and_chunk
from core.snapshot_store import record_snapshot


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one :func:`ingest_page` call."""

    snapshot_id: int
    chunks_written: int
    final_url: str


def ingest_page(
    engine: Engine,
    *,
    fetcher: Fetcher,
    parser: Parser,
    url: str,
    entity_id: int | None,
    source_kind: str,
    render_js: bool = False,
) -> IngestResult:
    """Fetch ``url`` once, snapshot it, then parse and store its chunks."""
    result = fetcher.fetch(url, render_js=render_js)
    snapshot_id = record_snapshot(
        engine, entity_id=entity_id, url=url, result=result, source_kind=source_kind
    )
    chunks = parse_and_chunk(
        result.body_bytes,
        content_type=result.content_type,
        url=result.final_url,
        parser=parser,
    )
    written = store_chunks(engine, snapshot_id, chunks)
    return IngestResult(snapshot_id=snapshot_id, chunks_written=written, final_url=result.final_url)
