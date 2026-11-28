"""Tests for ``landscape export --gate-report`` (EV-04); SQLite fixture, neutral pack."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from tests.test_cli_export import EXAMPLE_SCOPE, add, add_run, engine  # noqa: F401

REPORT = {
    "version": 1,
    "overall": "fail",
    "criteria": {
        "c1": {"release": "release", "reasons": []},
        "c2": {"release": "experimental", "reasons": ["n<10"]},
    },
}


def invoke(out: Path, fmt: str, *args: str):
    return CliRunner().invoke(
        cli.app, ["export", "--scope", EXAMPLE_SCOPE, "--format", fmt, "--out", str(out), *args]
    )


def _seed(engine, tmp_path) -> Path:
    rid = add_run(engine)
    add(engine, rid, 1, criterion="c1")
    add(engine, rid, 2, criterion="c2")
    add(engine, rid, 3, criterion="c3")
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(REPORT), encoding="utf-8")
    return path


def test_json_rows_gain_release_label(engine, tmp_path):
    report = _seed(engine, tmp_path)
    out = tmp_path / "x.jsonl"
    result = invoke(out, "json", "--gate-report", str(report))
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["criterion_id"]: r["release"] for r in rows} == {
        "c1": "release",
        "c2": "experimental",
        "c3": "ungated",
    }


def test_csv_gains_trailing_release_column(engine, tmp_path):
    report = _seed(engine, tmp_path)
    out = tmp_path / "x.csv"
    result = invoke(out, "csv", "--gate-report", str(report))
    assert result.exit_code == 0, result.output
    rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [*cli.EXPORT_CSV_HEADER, "release"]
    crit = rows[0].index("criterion_id")
    assert {r[crit]: r[-1] for r in rows[1:]} == {
        "c1": "release",
        "c2": "experimental",
        "c3": "ungated",
    }


def test_without_flag_no_release_field(engine, tmp_path):
    _seed(engine, tmp_path)
    out = tmp_path / "x.jsonl"
    assert invoke(out, "json").exit_code == 0
    assert all("release" not in json.loads(line) for line in out.read_text().splitlines())
    out_csv = tmp_path / "x.csv"
    assert invoke(out_csv, "csv").exit_code == 0
    assert out_csv.read_text().splitlines()[0].split(",") == list(cli.EXPORT_CSV_HEADER)


def test_malformed_report_exits_2_no_output(engine, tmp_path):
    _seed(engine, tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 1, "overall": "pass", "criteria": {"c1": {"release": "zz-secret"}}}')
    for fmt in ("json", "csv"):
        out = tmp_path / f"x.{fmt}"
        result = invoke(out, fmt, "--gate-report", str(bad))
        assert result.exit_code == 2
        assert str(bad) in result.output
        assert "zz-secret" not in result.output
        assert not out.exists()
        assert not Path(f"{out}.tmp").exists()


def test_missing_report_exits_2(engine, tmp_path):
    out = tmp_path / "x.jsonl"
    result = invoke(out, "json", "--gate-report", str(tmp_path / "none.json"))
    assert result.exit_code == 2
    assert not out.exists()
