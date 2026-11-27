"""RT-01: ``landscape presence-extract`` runs through StageRunner (assignment 132/A1); SQLite."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.chunk_store import ChunkRow
from core.runs import Run, WorkUnit
from tests.test_presence_extract import SCOPE, engine, invoke, rows  # noqa: F401

MATCHES = [
    "entity 1 country XA chunk 101",
    "entity 1 country XB chunk 103",
    "entity 2 country XC chunk 200",
]


def extract(*extra: str):
    return invoke("presence-extract", "--scope", str(SCOPE), "--type", "site", *extra)


def runs(engine: Engine) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(sa.select(Run.stage, Run.status))]


def units(engine: Engine) -> dict[str, tuple[str, int]]:
    with engine.connect() as conn:
        return {
            k: (s, a)
            for k, s, a in conn.execute(
                sa.select(WorkUnit.unit_key, WorkUnit.status, WorkUnit.attempts).where(
                    WorkUnit.stage == "presence-extract"
                )
            )
        }


def test_first_run_one_run_row_and_done_units(engine):
    result = extract("--entity", "1", "--entity", "2")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [*MATCHES, "extracted 3"]
    assert runs(engine) == [("presence-extract", "finished")]
    assert units(engine) == {"1:site": ("done", 1), "2:site": ("done", 1)}


def test_rerun_unchanged_skips_without_duplicates(engine):
    extract("--entity", "1", "--entity", "2")
    before = rows(engine)
    result = extract("--entity", "1", "--entity", "2")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["extracted 0", "skipped 2"]
    assert rows(engine) == before
    assert units(engine) == {"1:site": ("done", 1), "2:site": ("done", 1)}


def test_added_chunk_reruns_only_that_unit(engine):
    extract("--entity", "1", "--entity", "2")
    with Session(engine) as session, session.begin():
        text = "Now in Avalor too."
        session.add(ChunkRow(id=202, snapshot_id=20, seq=202, text=text, lang="en", page=None,
                             heading_path=[], char_start=0, char_end=len(text)))  # fmt: skip
    result = extract("--entity", "1", "--entity", "2")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "entity 2 country XA chunk 202",
        "entity 2 country XC chunk 200",
        "extracted 2",
        "skipped 1",
    ]
    assert units(engine)["1:site"] == ("done", 1)
    assert sorted(k for k, (s, _) in units(engine).items() if s == "done") == ["1:site", "2:site"]
    assert (2, "XA", "site", 202) in rows(engine)


def test_all_entities_streamed_as_units(engine):
    result = extract()
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [*MATCHES, "extracted 3"]
    assert set(units(engine)) == {"1:site", "2:site", "3:site"}


def test_dry_run_writes_no_run_or_units(engine):
    result = extract("--dry-run")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [*MATCHES, "extracted 3"]
    assert runs(engine) == [] and units(engine) == {} and rows(engine) == []


def test_unknown_entity_exits_2_without_run(engine):
    result = extract("--entity", "99")
    assert result.exit_code == 2
    assert "unknown entity 99" in result.output
    assert runs(engine) == [] and rows(engine) == []
