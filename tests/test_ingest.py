"""Tests for single-page ingest (IN-01). Offline, in-memory SQLite, neutral fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.fetcher import FetchError, FetchResult, InMemoryFetcher
from adapters.html_parser import HtmlParser
from adapters.parser import ParseError
from core.chunk_store import ChunkRow
from core.ingest import IngestResult, ingest_page
from core.runs import create_all
from core.snapshot_store import SnapshotRow

URL = "https://example.test/page"
FINAL = "https://example.test/page/"
HTML = (
    b"<html><head><title>Example</title></head><body>"
    b"<h1>Guide</h1><p>Intro text about widgets.</p>"
    b"<h2>Setup</h2><p>Install the widget carefully.</p>"
    b"</body></html>"
)


def memory_engine() -> Engine:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    create_all(engine)
    return engine


def fetcher(body: bytes = HTML, content_type: str = "text/html; charset=utf-8") -> InMemoryFetcher:
    result = FetchResult(final_url=FINAL, status=200, content_type=content_type, body_bytes=body)
    return InMemoryFetcher({FINAL: result}, redirects={URL: FINAL})


def snapshots(engine: Engine) -> list[SnapshotRow]:
    with Session(engine) as s:
        return list(s.scalars(sa.select(SnapshotRow).order_by(SnapshotRow.id)))


def chunks(engine: Engine) -> list[ChunkRow]:
    with Session(engine) as s:
        return list(s.scalars(sa.select(ChunkRow).order_by(ChunkRow.snapshot_id, ChunkRow.seq)))


def test_happy_path_writes_snapshot_and_heading_chunks() -> None:
    engine = memory_engine()
    f = fetcher()
    result = ingest_page(
        engine, fetcher=f, parser=HtmlParser(), url=URL, entity_id=7, source_kind="web"
    )
    assert isinstance(result, IngestResult)
    assert result.final_url == FINAL
    assert f.calls == [(URL, False)]
    [snap] = snapshots(engine)
    assert snap.id == result.snapshot_id
    assert snap.url == URL and snap.final_url == FINAL and snap.entity_id == 7
    assert snap.sha256 == hashlib.sha256(HTML).hexdigest()
    rows = chunks(engine)
    assert len(rows) == result.chunks_written > 0
    assert all(r.snapshot_id == snap.id for r in rows)
    paths = [tuple(r.heading_path) for r in rows]
    assert ("Guide",) in paths or ("Guide", "Setup") in paths
    assert all(p and p[0] == "Guide" for p in paths)
    setup = [r for r in rows if "Install" in r.text]
    assert setup and tuple(setup[0].heading_path) == ("Guide", "Setup")


def test_render_js_is_passed_to_fetcher() -> None:
    f = fetcher()
    ingest_page(
        memory_engine(),
        fetcher=f,
        parser=HtmlParser(),
        url=URL,
        entity_id=None,
        source_kind="web",
        render_js=True,
    )
    assert f.calls == [(URL, True)]


def test_fetch_error_writes_no_snapshot() -> None:
    engine = memory_engine()
    f = InMemoryFetcher({})
    with pytest.raises(FetchError):
        ingest_page(
            engine, fetcher=f, parser=HtmlParser(), url=URL, entity_id=None, source_kind="web"
        )
    assert snapshots(engine) == []
    assert chunks(engine) == []


def test_parse_error_keeps_snapshot_and_writes_no_chunks() -> None:
    engine = memory_engine()
    with pytest.raises(ParseError):
        ingest_page(
            engine,
            fetcher=fetcher(b"\x00\x01", "application/octet-stream"),
            parser=HtmlParser(),
            url=URL,
            entity_id=None,
            source_kind="web",
        )
    assert len(snapshots(engine)) == 1
    assert chunks(engine) == []


def test_reingest_creates_new_snapshot() -> None:
    engine = memory_engine()
    kw = dict(parser=HtmlParser(), url=URL, entity_id=None, source_kind="web")
    a = ingest_page(engine, fetcher=fetcher(), **kw)  # type: ignore[arg-type]
    b = ingest_page(engine, fetcher=fetcher(), **kw)  # type: ignore[arg-type]
    assert a.snapshot_id != b.snapshot_id
    assert len(snapshots(engine)) == 2
    assert {r.snapshot_id for r in chunks(engine)} == {a.snapshot_id, b.snapshot_id}


def test_chunks_stream_without_materialising(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.ingest as ingest_mod

    seen: dict[str, object] = {}

    def spy_store(engine: Engine, snapshot_id: int, items: object, **_: object) -> int:
        seen["type"] = type(items)
        assert isinstance(items, Iterator)
        return sum(1 for _ in items)

    monkeypatch.setattr(ingest_mod, "store_chunks", spy_store)
    result = ingest_page(
        memory_engine(),
        fetcher=fetcher(),
        parser=HtmlParser(),
        url=URL,
        entity_id=None,
        source_kind="web",
    )
    assert seen["type"] is not list
    assert result.chunks_written > 0
