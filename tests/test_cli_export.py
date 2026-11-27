"""Tests for ``landscape export --format json`` (AS-12); SQLite fixture, CLI via CliRunner."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.assess import Extraction, Quote
from core.assessment_store import AssessmentRow, record_assessment
from core.chunk_store import ChunkRow
from core.snapshot_store import SnapshotRow
from core.runs import Run, create_all
from core.verify import Route

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")

KEYS = {
    "id",
    "entity_id",
    "criterion_id",
    "status",
    "confidence",
    "rationale",
    "rubric_version",
    "extractor_model",
    "verifier_model",
    "decided_by",
    "route_kind",
    "run_id",
    "created_at",
    "evidence",
}


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
    try:
        yield engine
    finally:
        engine.dispose()


def add_run(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
        run = Run(stage="assess", status="finished", started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def add(engine, run_id, entity_id, criterion="c1", version="1", quotes=(), kind="accept") -> int:
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status="strong", quotes=tuple(quotes), rationale="x", checks_passed=True, failures=()
        ),
        route=Route(kind, None, "r"),
        rubric_version=version,
        extractor_model="e",
        verifier_model="v",
        run_id=run_id,
    )


def invoke(out: Path, *args: str):
    return CliRunner().invoke(
        cli.app, ["export", "--scope", EXAMPLE_SCOPE, "--format", "json", "--out", str(out), *args]
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_json_lines_with_key_set_and_view_order(engine, tmp_path) -> None:
    rid = add_run(engine)
    later_entity = add(
        engine,
        rid,
        6,
        criterion="c2",
        version="2",
        quotes=[Quote(3, "a", True, "exact", (0, 1)), Quote(4, "b", False, None, None)],
    )
    earlier_entity = add(engine, rid, 5, quotes=[Quote(9, "old", True, "exact", (0, 3))])
    out = tmp_path / "current.jsonl"
    result = invoke(out)
    assert result.exit_code == 0, result.output
    assert result.output == f"exported 2 assessments to {out}\n"
    rows = read_rows(out)
    assert [row["id"] for row in rows] == [earlier_entity, later_entity]
    assert all(set(row) == KEYS for row in rows)
    first, second = rows
    assert first == {
        "id": earlier_entity,
        "entity_id": 5,
        "criterion_id": "c1",
        "status": "strong",
        "confidence": None,
        "rationale": "x",
        "rubric_version": "1",
        "extractor_model": "e",
        "verifier_model": "v",
        "decided_by": "auto",
        "route_kind": "accept",
        "run_id": rid,
        "created_at": first["created_at"],
        "evidence": [
            {
                "chunk_id": 9,
                "quote": "old",
                "grounded": True,
                "snapshot_id": None,
                "url": None,
                "fetched_at": None,
            }
        ],
    }
    assert second["evidence"] == [
        {
            "chunk_id": 3,
            "quote": "a",
            "grounded": True,
            "snapshot_id": None,
            "url": None,
            "fetched_at": None,
        },
        {
            "chunk_id": 4,
            "quote": "b",
            "grounded": False,
            "snapshot_id": None,
            "url": None,
            "fetched_at": None,
        },
    ]
    for row in rows:
        parsed = datetime.fromisoformat(str(row["created_at"]))
        assert parsed.tzinfo is not None
    assert not Path(f"{out}.tmp").exists()


def test_newer_write_supersedes_older(engine, tmp_path) -> None:
    rid = add_run(engine)
    old = add(engine, rid, 5, quotes=[Quote(9, "old", True, "exact", (0, 3))])
    new = add(engine, rid, 5, quotes=[Quote(3, "new", True, "exact", (0, 3))])
    with Session(engine) as session, session.begin():
        session.get(AssessmentRow, old).created_at = datetime(2020, 1, 1, tzinfo=UTC)  # type: ignore[union-attr]
    out = tmp_path / "current.jsonl"
    result = invoke(out)
    assert result.exit_code == 0, result.output
    rows = read_rows(out)
    assert [row["id"] for row in rows] == [new]
    assert rows[0]["evidence"] == [
        {
            "chunk_id": 3,
            "quote": "new",
            "grounded": True,
            "snapshot_id": None,
            "url": None,
            "fetched_at": None,
        }
    ]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--entity", "2"], ["c", "b"]),
        (["--criterion", "c2"], ["b"]),
        (["--rubric-version", "2"], ["a", "b"]),
        (["--entity", "2", "--criterion", "c2", "--rubric-version", "2"], ["b"]),
        (["--rubric-version", "9"], []),
    ],
)
def test_filters_pass_through(engine, tmp_path, args, expected) -> None:
    rid = add_run(engine)
    ids = {
        "a": add(engine, rid, 1, criterion="c1", version="2"),
        "b": add(engine, rid, 2, criterion="c2", version="2"),
        "c": add(engine, rid, 2, criterion="c1", version="1"),
    }
    out = tmp_path / "current.jsonl"
    result = invoke(out, *args)
    assert result.exit_code == 0, result.output
    assert [row["id"] for row in read_rows(out)] == [ids[name] for name in expected]


def test_empty_db_writes_empty_file_and_zero(engine, tmp_path) -> None:
    out = tmp_path / "current.jsonl"
    result = invoke(out)
    assert result.exit_code == 0, result.output
    assert result.output == f"exported 0 assessments to {out}\n"
    assert out.read_bytes() == b""
    assert not Path(f"{out}.tmp").exists()


def test_bad_format_exits_2_without_file(engine, tmp_path) -> None:
    out = tmp_path / "current.jsonl"
    result = CliRunner().invoke(
        cli.app,
        ["export", "--scope", EXAMPLE_SCOPE, "--format", "parquet", "--out", str(out)],
    )
    assert result.exit_code == 2
    assert "unsupported format" in result.output
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_missing_scope_exits_2_without_file(engine, tmp_path) -> None:
    out = tmp_path / "current.jsonl"
    result = CliRunner().invoke(
        cli.app,
        ["export", "--scope", str(tmp_path / "nope"), "--format", "json", "--out", str(out)],
    )
    assert result.exit_code == 2
    assert "scope directory not found" in result.output
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_unwritable_output_directory_exits_2_without_file(engine, tmp_path) -> None:
    rid = add_run(engine)
    add(engine, rid, 1)
    out = tmp_path / "absent" / "current.jsonl"
    result = invoke(out)
    assert result.exit_code == 2
    assert "cannot write" in result.output
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()
    assert not out.parent.exists()


def test_export_consumes_streaming_generator(engine, tmp_path, monkeypatch) -> None:
    rid = add_run(engine)
    ids = [add(engine, rid, e) for e in (1, 2, 3)]
    real = cli.iter_current_assessments
    seen: list[int] = []

    def spy(engine, **kwargs):
        for row in real(engine, **kwargs):
            seen.append(row.id)
            yield row

    monkeypatch.setattr(cli, "iter_current_assessments", spy)
    monkeypatch.setattr(cli, "current_assessments", None)
    out = tmp_path / "current.jsonl"
    result = invoke(out)
    assert result.exit_code == 0, result.output
    assert seen == ids
    assert [row["id"] for row in read_rows(out)] == ids


def _add_source(engine: Engine, *, with_snapshot: bool) -> int:
    """Store a chunk (and optionally its snapshot); return the chunk id."""
    with Session(engine) as session, session.begin():
        snapshot_id = 777
        if with_snapshot:
            snap = SnapshotRow(
                entity_id=5,
                url="https://example.org/start",
                final_url="https://example.org/final",
                content_type="text/html",
                sha256="0" * 64,
                fetched_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
                source_kind="web",
            )
            session.add(snap)
            session.flush()
            snapshot_id = snap.id
        chunk = ChunkRow(
            snapshot_id=snapshot_id,
            seq=0,
            text="quote text",
            heading_path=[],
            char_start=0,
            char_end=10,
        )
        session.add(chunk)
        session.flush()
        return chunk.id


def test_evidence_carries_snapshot_source(engine, tmp_path) -> None:
    rid = add_run(engine)
    chunk_id = _add_source(engine, with_snapshot=True)
    add(engine, rid, 5, quotes=[Quote(chunk_id, "q", True, "exact", (0, 1))])
    out = tmp_path / "current.jsonl"
    assert invoke(out).exit_code == 0
    [row] = read_rows(out)
    [ev] = row["evidence"]
    with Session(engine) as session:
        snap_id = session.query(SnapshotRow.id).scalar()
    assert ev == {
        "chunk_id": chunk_id,
        "quote": "q",
        "grounded": True,
        "snapshot_id": snap_id,
        "url": "https://example.org/final",
        "fetched_at": "2026-01-02T03:04:05+00:00",
    }


@pytest.mark.parametrize("chunk_present", [True, False])
def test_evidence_source_null_when_chunk_or_snapshot_missing(
    engine, tmp_path, chunk_present
) -> None:
    rid = add_run(engine)
    chunk_id = _add_source(engine, with_snapshot=False) if chunk_present else 4242
    add(engine, rid, 5, quotes=[Quote(chunk_id, "q", False, None, None)])
    out = tmp_path / "current.jsonl"
    assert invoke(out).exit_code == 0
    [row] = read_rows(out)
    assert row["evidence"] == [
        {
            "chunk_id": chunk_id,
            "quote": "q",
            "grounded": False,
            "snapshot_id": None,
            "url": None,
            "fetched_at": None,
        }
    ]
