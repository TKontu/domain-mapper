"""Tests for deterministic presence extraction (MP-03): ``landscape presence-extract``; SQLite."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.chunk_store import ChunkRow
from core.entity_store import Entity
from core.presence_extract import compile_terms, extract_presence
from core.presence_store import MarketPresenceRow
from core.runs import create_all
from core.scope_pack import ScopePackError, load_scope_pack
from core.snapshot_store import SnapshotRow

SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


@pytest.fixture
def engine(tmp_path, monkeypatch) -> Engine:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    engine = create_engine(url)
    create_all(engine)
    scope_id = load_scope_pack(SCOPE).scope.id
    with Session(engine) as session, session.begin():
        for eid in (1, 2, 3):
            session.add(Entity(id=eid, scope_id=scope_id, name=f"E{eid}",
                               primary_domain=f"e{eid}.example", status="unscreened"))  # fmt: skip
        for sid, eid in ((10, 1), (11, 1), (20, 2)):
            session.add(SnapshotRow(id=sid, entity_id=eid, url="https://x.example/",
                                    final_url="https://x.example/", content_type="text/html",
                                    sha256="0" * 64, fetched_at=datetime.now(UTC),
                                    source_kind="crawl"))  # fmt: skip
        texts = [
            (100, 10, "We serve clients from our office."),
            (101, 10, "Offices in AVALOR and port avalor."),
            (102, 11, "Also in Avalor; Avalorian cuisine nearby."),
            (103, 11, "Partners in Brennmark."),
            (200, 20, "Head office in Calvessa."),
            (201, 20, "Brennmarkers unite."),
        ]
        for cid, sid, text in texts:
            session.add(ChunkRow(id=cid, snapshot_id=sid, seq=cid, text=text, lang="en",
                                 page=None, heading_path=[], char_start=0,
                                 char_end=len(text)))  # fmt: skip
    try:
        yield engine
    finally:
        engine.dispose()


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def rows(engine: Engine) -> list[tuple[int, str, str, int | None]]:
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(
            sa.select(MarketPresenceRow.entity_id, MarketPresenceRow.country,
                      MarketPresenceRow.presence_type, MarketPresenceRow.evidence_id)
            .order_by(MarketPresenceRow.entity_id, MarketPresenceRow.country)
        )]  # fmt: skip


def test_extract_records_lowest_chunk_and_respects_word_boundary(engine):
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "site")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "entity 1 country XA chunk 101",
        "entity 1 country XB chunk 103",
        "entity 2 country XC chunk 200",
        "extracted 3",
    ]
    assert rows(engine) == [(1, "XA", "site", 101), (1, "XB", "site", 103), (2, "XC", "site", 200)]


def test_rerun_is_idempotent(engine):
    invoke("presence-extract", "--scope", str(SCOPE), "--type", "site")
    before = rows(engine)
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "site")
    assert result.exit_code == 0, result.output
    assert rows(engine) == before


def test_entity_filter_and_no_cross_attribution(engine):
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "sales", "--entity", "2")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["entity 2 country XC chunk 200", "extracted 1"]
    assert rows(engine) == [(2, "XC", "sales", 200)]


def test_entity_without_chunks_matches_nothing(engine):
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "hq", "--entity", "3")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["extracted 0"]


def test_dry_run_writes_nothing(engine):
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "site", "--dry-run")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "extracted 3"
    assert rows(engine) == []


def test_bad_type_exits_2(engine):
    result = invoke("presence-extract", "--scope", str(SCOPE), "--type", "office")
    assert result.exit_code == 2
    assert rows(engine) == []


def _copy_scope(tmp_path: Path, mutate) -> Path:
    dest = tmp_path / "pack"
    dest.mkdir()
    for item in SCOPE.iterdir():
        if item.is_file():
            (dest / item.name).write_bytes(item.read_bytes())
    data = yaml.safe_load((dest / "scope.yaml").read_text())
    mutate(data)
    (dest / "scope.yaml").write_text(yaml.safe_dump(data))
    return dest


def test_scope_without_presence_terms_exits_2(engine, tmp_path):
    pack = _copy_scope(tmp_path, lambda d: d.pop("presence_terms"))
    result = invoke("presence-extract", "--scope", str(pack), "--type", "site")
    assert result.exit_code == 2
    assert "presence_terms" in result.output
    assert rows(engine) == []


@pytest.mark.parametrize("terms", [{"xa": ["Avalor"]}, {"XAA": ["Avalor"]}, {"XA": [""]},
                                   {"XA": []}])  # fmt: skip
def test_invalid_presence_terms_fail_validation(tmp_path, terms):
    pack = _copy_scope(tmp_path, lambda d: d.update(presence_terms=terms))
    with pytest.raises(ScopePackError):
        load_scope_pack(pack)


def test_compile_terms_escapes_and_shares_terms():
    compiled = compile_terms({"XA": ["a.b"], "XB": ["A.B"]})
    assert compiled.pattern.search("x a.b y")
    assert not compiled.pattern.search("x aXb y")
    assert compiled.countries["a.b"] == ("XA", "XB")


def test_extract_presence_function(engine):
    compiled = compile_terms(load_scope_pack(SCOPE).scope.presence_terms or {})
    with Session(engine) as session:
        found = extract_presence(session, compiled, [2, 1])
    assert [(m.entity_id, m.country, m.chunk_id) for m in found] == [
        (2, "XC", 200), (1, "XA", 101), (1, "XB", 103),
    ]  # fmt: skip
