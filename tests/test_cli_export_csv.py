"""Tests for ``landscape export --format csv`` (XP-04); SQLite under a temp dir."""

from __future__ import annotations

import csv
import io
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.chunk_store import ChunkRow
from core.runs import Run, create_all
from core.snapshot_store import SnapshotRow
from core.verify import Route

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
HEADER = [
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
    "evidence_count",
    "grounded_count",
    "first_url",
    "first_fetched_at",
]


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


def add(engine, run_id, entity_id, criterion="c1", version="1", quotes=(), rationale="x") -> int:
    return record_assessment(
        engine,
        entity_id=entity_id,
        criterion_id=criterion,
        extraction=Extraction(
            status="strong",
            quotes=tuple(quotes),
            rationale=rationale,
            checks_passed=True,
            failures=(),
        ),
        route=Route("accept", None, "r"),
        rubric_version=version,
        extractor_model="e",
        verifier_model="v",
        run_id=run_id,
    )


def add_chunk(engine: Engine) -> int:
    with Session(engine) as session, session.begin():
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
        chunk = ChunkRow(
            snapshot_id=snap.id, seq=0, text="q", heading_path=[], char_start=0, char_end=1
        )
        session.add(chunk)
        session.flush()
        return chunk.id


def invoke(out: Path, fmt: str, *args: str):
    return CliRunner().invoke(
        cli.app, ["export", "--scope", EXAMPLE_SCOPE, "--format", fmt, "--out", str(out), *args]
    )


def read_rows(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))


def test_csv_rows_with_counts_first_source_and_escaping(engine, tmp_path) -> None:
    rid = add_run(engine)
    chunk_id = add_chunk(engine)
    a = add(
        engine,
        rid,
        5,
        quotes=[Quote(chunk_id, "a", True, "exact", (0, 1)), Quote(99, "b", False, None, None)],
        rationale="=SUM(1)",
    )
    b = add(engine, rid, 6, criterion="c2", version="2")
    out = tmp_path / "current.csv"
    result = invoke(out, "csv")
    assert result.exit_code == 0, result.output
    assert result.output == f"exported 2 assessments to {out}\n"
    header, first, second = read_rows(out)
    assert header == HEADER
    assert first[:12] == [str(a), "5", "c1", "strong", "", "'=SUM(1)", "1", "e", "v", "auto",
                          "accept", str(rid)]  # fmt: skip
    datetime.fromisoformat(first[12])
    assert first[13:] == ["2", "1", "https://example.org/final", "2026-01-02T03:04:05+00:00"]
    assert second[0] == str(b)
    assert second[13:] == ["0", "0", "", ""]


def test_csv_first_source_empty_when_snapshot_missing(engine, tmp_path) -> None:
    rid = add_run(engine)
    add(engine, rid, 5, quotes=[Quote(4242, "q", True, "exact", (0, 1))])
    out = tmp_path / "current.csv"
    assert invoke(out, "csv").exit_code == 0
    [_, row] = read_rows(out)
    assert row[13:] == ["1", "1", "", ""]


def test_csv_empty_view_writes_header_only(engine, tmp_path) -> None:
    out = tmp_path / "current.csv"
    result = invoke(out, "csv")
    assert result.exit_code == 0, result.output
    assert result.output == f"exported 0 assessments to {out}\n"
    assert read_rows(out) == [HEADER]
    assert out.read_bytes().endswith(b"\r\n")
    assert not Path(f"{out}.tmp").exists()


@pytest.mark.parametrize(
    ("args", "entities"),
    [
        (("--entity", "6"), ["6"]),
        (("--criterion", "c1"), ["5"]),
        (("--rubric-version", "2"), ["6"]),
    ],
)
def test_csv_filters_match_json(engine, tmp_path, args, entities) -> None:
    rid = add_run(engine)
    add(engine, rid, 5)
    add(engine, rid, 6, criterion="c2", version="2")
    out = tmp_path / "current.csv"
    assert invoke(out, "csv", *args).exit_code == 0
    assert [row[1] for row in read_rows(out)[1:]] == entities


def test_unknown_format_exits_2_before_setup(engine, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load", lambda scope: pytest.fail("setup ran"))
    out = tmp_path / "current.parquet"
    result = invoke(out, "parquet")
    assert result.exit_code == 2
    assert "known: json, csv" in result.output
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()


def test_csv_streams_from_generator(engine, tmp_path, monkeypatch) -> None:
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
    out = tmp_path / "current.csv"
    assert invoke(out, "csv").exit_code == 0
    assert seen == ids
    assert [row[0] for row in read_rows(out)[1:]] == [str(i) for i in ids]


def test_csv_unwritable_directory_exits_2(engine, tmp_path) -> None:
    out = tmp_path / "absent" / "current.csv"
    result = invoke(out, "csv")
    assert result.exit_code == 2
    assert "cannot write" in result.output
    assert not out.parent.exists()


def test_json_output_unchanged_by_csv_support(engine, tmp_path) -> None:
    rid = add_run(engine)
    chunk_id = add_chunk(engine)
    add(engine, rid, 5, quotes=[Quote(chunk_id, "a", True, "exact", (0, 1))])
    out = tmp_path / "current.jsonl"
    assert invoke(out, "json").exit_code == 0
    [row] = [cli.json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert out.read_text(encoding="utf-8") == (cli.json.dumps(row, ensure_ascii=False) + "\n")
    assert "evidence_count" not in row
