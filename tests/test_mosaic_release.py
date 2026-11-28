"""Tests for FE-06: ``landscape mosaic --gate-report`` release labels and ``--hide-experimental``."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from app import cli

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"

SCORES = [
    {
        "entity_id": 7,
        "topics": [
            {"topic_id": "certification", "score": 1.0, "coverage": 1.0},
            {"topic_id": "terms", "score": 0.5, "coverage": 1.0},
        ],
    }
]
EXPORT = [
    {"entity_id": 7, "criterion_id": "iso17100", "status": "strong", "evidence": []},
    {"entity_id": 7, "criterion_id": "public_terms", "status": "weak", "evidence": []},
]


def _report(criteria: dict[str, str]) -> dict:
    return {
        "version": 1,
        "overall": "fail",
        "criteria": {c: {"release": r, "reasons": []} for c, r in criteria.items()},
    }


def _setup(tmp_path: Path, report: object | None = None) -> list[str]:
    scores = tmp_path / "s.jsonl"
    export = tmp_path / "e.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in SCORES), encoding="utf-8")
    export.write_text("".join(json.dumps(r) + "\n" for r in EXPORT), encoding="utf-8")
    args = ["mosaic", "--scope", str(EXAMPLE_SCOPE), "--scores", str(scores)]
    args += ["--export", str(export), "--out", str(tmp_path / "page.html")]
    if report is not None:
        path = tmp_path / "gate.json"
        path.write_text(report if isinstance(report, str) else json.dumps(report), "utf-8")
        args += ["--gate-report", str(path)]
    return args


def _run(args: list[str]) -> tuple[int, str]:
    result = CliRunner().invoke(cli.app, args)
    return result.exit_code, result.output


def test_headers_carry_release_label_and_marker(tmp_path: Path) -> None:
    rc, _ = _run(_setup(tmp_path, _report({"iso17100": "release", "public_terms": "experimental"})))
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    heads = re.findall(r'<th data-criterion="([^"]+)" data-release="([^"]+)"', page)
    assert heads == [("iso17100", "release"), ("public_terms", "experimental")]
    assert ">public_terms (experimental)</th>" in page
    assert ">iso17100</th>" in page
    assert page.count('data-status="weak"') == 1


def test_absent_criterion_is_ungated_and_never_hidden(tmp_path: Path) -> None:
    rc, _ = _run([*_setup(tmp_path, _report({"iso17100": "experimental"})), "--hide-experimental"])
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert re.findall(r'data-criterion="([^"]+)" data-release="([^"]+)"', page) == [
        ("public_terms", "ungated")
    ]


def test_hide_experimental_drops_header_cells_and_entity_rows(tmp_path: Path) -> None:
    report = _report({"iso17100": "release", "public_terms": "experimental"})
    ents = tmp_path / "ents"
    rc, _ = _run([*_setup(tmp_path, report), "--hide-experimental", "--entity-dir", str(ents)])
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "public_terms" not in page
    assert 'data-status="weak"' not in page and 'data-status="strong"' in page
    assert re.findall(r'colspan="(\d+)" data-topic="([^"]+)"', page) == [
        ("2", "certification"),
        ("1", "terms"),
    ]
    assert "0.50 / 1.00" in page  # all-hidden topic keeps its score/coverage column
    (entity_page,) = list(ents.glob("*.html"))
    text = entity_page.read_text(encoding="utf-8")
    assert 'data-criterion="iso17100"' in text and "public_terms" not in text
    assert 'data-topic="terms"' in text


def test_hide_experimental_without_report_exits_2(tmp_path: Path) -> None:
    args = _setup(tmp_path)
    args[args.index("--scores") + 1] = str(tmp_path / "missing.jsonl")
    rc, output = _run([*args, "--hide-experimental"])
    assert rc == 2
    assert "--gate-report" in output
    assert not (tmp_path / "page.html").exists()


def test_malformed_report_exits_2_without_output(tmp_path: Path) -> None:
    rc, output = _run(_setup(tmp_path, '{"secret-content": '))
    assert rc == 2
    assert "gate.json" in output and "invalid JSON" in output
    assert "secret-content" not in output
    assert not (tmp_path / "page.html").exists()


def test_without_flags_output_has_no_release_attrs(tmp_path: Path) -> None:
    rc, _ = _run(_setup(tmp_path))
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "data-release" not in page and "(experimental)" not in page
