"""Tests for XP-01: ``landscape scores-export --format csv``; SQLite under a temp dir, offline."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from tests.test_cli_scores_export import (  # noqa: F401  (engine is a fixture)
    EXAMPLE_SCOPE,
    add,
    add_entity,
    add_run,
    add_screen,
    engine,
    scores_json,
)

FIELDS = ("score", "coverage", "applicable", "scored", "evidenced")


def export_csv(out: Path, fmt: str = "csv"):
    return CliRunner().invoke(
        cli.app, ["scores-export", "--scope", EXAMPLE_SCOPE, "--out", str(out), "--format", fmt]
    )


def topic_ids() -> list[str]:
    return [t.id for t in cli.load_scope_pack(Path(EXAMPLE_SCOPE)).rubric.topics]


def header() -> list[str]:
    base = ["entity_id", "segments", "layers", "rubric_version"]
    return base + [f"{tid}.{f}" for tid in topic_ids() for f in FIELDS]


def expected_row(obj: dict) -> list[str]:
    def cell(v):
        return "" if v is None else str(v)

    row = [str(obj["entity_id"]), ";".join(obj["segments"]), "", obj["rubric_version"]]
    by = {t["topic_id"]: t for t in obj["topics"]}
    for tid in topic_ids():
        row += [cell(by[tid][f]) for f in FIELDS]
    return row


def read_rows(out: Path) -> list[list[str]]:
    data = out.read_bytes()
    assert b"\r" not in data
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


def test_csv_header_and_rows_match_scores_json(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    b = add_entity(engine, "beta")
    add_entity(engine, "gamma")
    add_screen(engine, rid, a, ["agency", "freelancer_network"])
    add_screen(engine, rid, b, ["freelancer_network"])
    add(engine, rid, a, "iso17100", "strong")
    add(engine, rid, a, "public_terms", "weak", grounded=False)
    add(engine, rid, b, "public_terms", "standard")
    out = tmp_path / "scores.csv"
    result = export_csv(out)
    assert result.exit_code == 0, result.output
    rows = read_rows(out)
    assert rows[0] == header()
    assert rows[1:] == [expected_row(scores_json(a)), expected_row(scores_json(b))]
    assert rows[1][1] == "agency;freelancer_network"
    assert any("." in cell for cell in rows[1][4:])  # unrounded float present
    assert result.stdout == f"exported 2 entities to {out} (skipped 1 without screen result)\n"
    assert not Path(f"{out}.tmp").exists()


def test_null_scores_are_empty_cells(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    add_screen(engine, rid, a, ["agency"])
    out = tmp_path / "scores.csv"
    assert export_csv(out).exit_code == 0
    _, row = read_rows(out)
    scores = row[4::5]
    assert scores and all(cell == "" for cell in scores)
    assert row == expected_row(scores_json(a))


def test_comma_and_quote_values_are_quoted(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    add_screen(engine, rid, a, ['odd,"seg"'])
    out = tmp_path / "scores.csv"
    assert export_csv(out).exit_code == 0
    assert f'\n{a},"odd,""seg""",,' in out.read_text(encoding="utf-8")
    assert read_rows(out)[1][1] == 'odd,"seg"'


def test_empty_scope_writes_header_only(engine, tmp_path) -> None:
    out = tmp_path / "scores.csv"
    result = export_csv(out)
    assert result.exit_code == 0, result.output
    assert read_rows(out) == [header()]
    assert result.stdout == f"exported 0 entities to {out} (skipped 0 without screen result)\n"


def test_explicit_jsonl_matches_default(engine, tmp_path) -> None:
    rid = add_run(engine)
    a = add_entity(engine, "alpha")
    add_screen(engine, rid, a, ["agency"])
    add(engine, rid, a, "iso17100", "strong")
    default, explicit = tmp_path / "d.jsonl", tmp_path / "e.jsonl"
    assert (
        CliRunner()
        .invoke(cli.app, ["scores-export", "--scope", EXAMPLE_SCOPE, "--out", str(default)])
        .exit_code
        == 0
    )
    assert export_csv(explicit, "jsonl").exit_code == 0
    assert explicit.read_bytes() == default.read_bytes()


def test_unknown_format_exits_2_before_setup(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def no_setup(*_a, **_k):
        raise AssertionError("setup reached")

    monkeypatch.setattr(cli, "_load", no_setup)
    out = tmp_path / "scores.parquet"
    result = export_csv(out, "parquet")
    assert result.exit_code == 2
    assert "unknown format" in result.stderr
    assert not out.exists()
    assert not Path(f"{out}.tmp").exists()
