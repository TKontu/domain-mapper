"""Tests for GS-04 ``landscape gold-import``; neutral ``scopes/_example`` pack only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from core.gold_import import GoldImportError, import_labels
from sqlalchemy import create_engine
from typer.testing import CliRunner

from app import cli
from core.gold_pick import cell_id
from core.gold_set import load_gold_set
from core.runs import create_all

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
STATUSES = {"strong", "standard", "weak", "red_flag", "silent", "na"}
SECRET = "SECRET-QUOTE-TEXT"


def write_fixture(tmp_path: Path, answers: dict[str, object]) -> tuple[Path, Path]:
    cells = tmp_path / "cells.jsonl"
    rows = [
        {"cell_id": cell_id(d, c), "primary_domain": d, "criterion_id": c, "evidence": []}
        for d, c in [
            ("a.example", "iso17100"),
            ("a.example", "public_terms"),
            ("b.example", "iso17100"),
        ]
    ]
    cells.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    return cells, root


def answer(status: str, quote: str = "q", note: str = "n") -> dict[str, str]:
    return {"status": status, "quote": quote, "note": note}


def test_import_writes_loadable_labels_and_notes(tmp_path):
    a1 = cell_id("a.example", "iso17100")
    a2 = cell_id("a.example", "public_terms")
    cells, root = write_fixture(tmp_path, {a1: answer("strong", "Q1", "N1"), a2: answer("silent")})
    out = tmp_path / "out"
    summary = import_labels(root, cells, out, STATUSES)
    assert (summary.imported, summary.unanswered) == (2, 1)
    labels = list(load_gold_set(out / "labels.jsonl", STATUSES))
    assert {(x.primary_domain, x.criterion_id, x.status) for x in labels} == {
        ("a.example", "iso17100", "strong"),
        ("a.example", "public_terms", "silent"),
    }
    notes = [json.loads(line) for line in (out / "label_notes.jsonl").read_text().splitlines()]
    first = next(n for n in notes if n["cell_id"] == a1)
    assert first["quote"] == "Q1" and first["note"] == "N1"
    assert first["primary_domain"] == "a.example"


def test_bad_status_names_cell_not_text(tmp_path):
    a1 = cell_id("a.example", "iso17100")
    cells, root = write_fixture(tmp_path, {a1: answer("bogus", SECRET, SECRET)})
    with pytest.raises(GoldImportError) as exc:
        import_labels(root, cells, tmp_path / "out", STATUSES)
    assert a1 in str(exc.value) and SECRET not in str(exc.value)
    assert not (tmp_path / "out" / "labels.jsonl").exists()


def test_unknown_cell_errors(tmp_path):
    cells, root = write_fixture(tmp_path, {"deadbeef": answer("strong", SECRET)})
    with pytest.raises(GoldImportError) as exc:
        import_labels(root, cells, tmp_path / "out", STATUSES)
    assert "deadbeef" in str(exc.value) and SECRET not in str(exc.value)


def test_refuses_overwrite_without_force(tmp_path):
    a1 = cell_id("a.example", "iso17100")
    cells, root = write_fixture(tmp_path, {a1: answer("weak")})
    out = tmp_path / "out"
    out.mkdir()
    (out / "labels.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(GoldImportError):
        import_labels(root, cells, out, STATUSES)
    assert (out / "labels.jsonl").read_text() == "old\n"
    import_labels(root, cells, out, STATUSES, force=True)
    assert [x.status for x in load_gold_set(out / "labels.jsonl", STATUSES)] == ["weak"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    engine = create_engine(url)
    create_all(engine)
    engine.dispose()


def test_cli_import_then_eval(tmp_path, db):
    a1 = cell_id("a.example", "iso17100")
    cells, root = write_fixture(tmp_path, {a1: answer("strong", SECRET)})
    out = tmp_path / "out"
    args = ["gold-import", "--scope", EXAMPLE_SCOPE, "--root", str(root), "--cells", str(cells)]
    result = CliRunner().invoke(cli.app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "imported 1" in result.output and "unanswered 2" in result.output
    assert SECRET not in result.output
    again = CliRunner().invoke(cli.app, [*args, "--out", str(out)])
    assert again.exit_code == 2
    forced = CliRunner().invoke(cli.app, [*args, "--out", str(out), "--force"])
    assert forced.exit_code == 0, forced.output
    gold = str(out / "labels.jsonl")
    ev = CliRunner().invoke(cli.app, ["eval", "--scope", EXAMPLE_SCOPE, "--gold", gold])
    assert ev.exit_code == 0, ev.output
    assert "unmatched 1" in ev.output
