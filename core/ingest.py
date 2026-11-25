"""Single-page ingest: fetch → raw blob → snapshot → parse → chunk rows (IN-01, RS-02).

ARCHITECTURE.md §4 (Crawl & snapshot; Parse & chunk). :func:`ingest_page`
fetches one URL once, records one ``snapshot`` row
(:func:`core.snapshot_store.record_snapshot`), parses and chunks the body
(:func:`core.parse_chunk.parse_and_chunk`) and stores the chunks
(:func:`core.chunk_store.store_chunks`).

Raw bytes (RS-02, D-005): when a keyword-only ``blob_store`` is given, the
body is stored with ``blob_store.put`` after the fetch and *before*
``record_snapshot``. The returned digest must equal the snapshot row's
``sha256`` (``hashlib.sha256(body).hexdigest()``); otherwise
:class:`~adapters.blob_store.BlobStoreError` is raised and no row is written.
:attr:`IngestResult.raw_digest` carries the digest, or ``None`` without a store
(``None`` keeps the IN-01 behaviour exactly).

Errors:

* :class:`~adapters.fetcher.FetchError` propagates before any row is written.
* :class:`~adapters.blob_store.BlobStoreError` from ``put`` (or a digest
  mismatch) propagates before any row is written; nothing is put on a
  ``FetchError``.
* :class:`~adapters.parser.ParseError` propagates *after* the snapshot row
  exists; no chunk rows are written for it (the fetch was a real observation).

Scale: one body in memory (bounded by the fetcher's byte cap); chunks stream
lazily into ``store_chunks``, never materialised as a list.

**Unwired by design**: the CLI ``crawl``/``parse`` stage row is the production
caller, including of ``blob_store`` (no config/env wiring here yet). No crawl
policy (robots, rate limits) lives here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from sqlalchemy import Engine

from adapters.blob_store import BlobStore, BlobStoreError
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
    raw_digest: str | None = None
    # Additive (CR-06): the fetched body and its content type, for same-host link discovery.
    body_bytes: bytes = field(default=b"", repr=False, compare=False)
    content_type: str = ""


def ingest_page(
    engine: Engine,
    *,
    fetcher: Fetcher,
    parser: Parser,
    url: str,
    entity_id: int | None,
    source_kind: str,
    render_js: bool = False,
    blob_store: BlobStore | None = None,
) -> IngestResult:
    """Fetch ``url`` once, snapshot it, then parse and store its chunks."""
    result = fetcher.fetch(url, render_js=render_js)
    raw_digest: str | None = None
    if blob_store is not None:
        raw_digest = blob_store.put(result.body_bytes)
        expected = hashlib.sha256(result.body_bytes).hexdigest()
        if raw_digest != expected:
            raise BlobStoreError(
                f"blob store returned digest {raw_digest!r}, expected sha256 {expected}"
            )
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
    return IngestResult(
        snapshot_id=snapshot_id,
        chunks_written=written,
        final_url=result.final_url,
        raw_digest=raw_digest,
        body_bytes=result.body_bytes,
        content_type=result.content_type,
    )
