"""Tests for the FE-03 ``landscape mosaic --segment`` filter; offline."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from app.mosaic import filter_by_segments, render_mosaic
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"

SCORES = [
    {"entity_id": 3, "segments": ["freelancer_network"], "topics": []},
    {"entity_id": 1, "segments": ["agency"], "topics": []},
    {"entity_id": 2, "segments": [], "topics": []},
    {"entity_id": 5, "segments": ["agency", "freelancer_network"], "topics": []},
]
EXPORT = [
    {"entity_id": 1, "criterion_id": "iso17100", "status": "strong", "evidence": []},
]


def _files(tmp_path: Path) -> tuple[Path, Path]:
    scores = tmp_path / "scores.jsonl"
    export = tmp_path / "export.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in SCORES), encoding="utf-8")
    export.write_text("".join(json.dumps(r) + "\n" for r in EXPORT), encoding="utf-8")
    return scores, export


def _run(tmp_path: Path, *extra: str) -> object:
    scores, export = _files(tmp_path)
    args = ["mosaic", "--scope", str(EXAMPLE_SCOPE), "--scores", str(scores)]
    args += ["--export", str(export), "--out", str(tmp_path / "m.html"), *extra]
    return CliRunner().invoke(cli.app, args)


def test_filter_keeps_matching_in_input_order_and_unions() -> None:
    kept = [r["entity_id"] for r in filter_by_segments(iter(SCORES), {"agency"})]
    assert kept == [1, 5]
    both = filter_by_segments(iter(SCORES), {"agency", "freelancer_network"})
    assert [r["entity_id"] for r in both] == [3, 1, 5]


def test_cli_segment_filters_rows_and_entity_pages(tmp_path: Path) -> None:
    ents = tmp_path / "ents"
    result = _run(tmp_path, "--segment", "agency", "--entity-dir", str(ents))
    assert result.exit_code == 0, result.output
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert '<tr data-entity="1">' in page and '<tr data-entity="5">' in page
    assert '<tr data-entity="3">' not in page and '<tr data-entity="2">' not in page
    assert page.index('data-entity="1"') < page.index('data-entity="5"')
    assert "segment: Agencies" in page
    names = sorted(p.name.split("-")[0] for p in ents.iterdir())
    assert names == ["1", "5"]


def test_cli_repeated_segment_unions(tmp_path: Path) -> None:
    result = _run(tmp_path, "--segment", "agency", "--segment", "freelancer_network")
    assert result.exit_code == 0, result.output
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    for e in ("3", "1", "5"):
        assert f'<tr data-entity="{e}">' in page
    assert '<tr data-entity="2">' not in page


def test_cli_unknown_segment_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    result = _run(tmp_path, "--segment", "nope_seg", "--entity-dir", str(tmp_path / "ents"))
    assert result.exit_code == 2
    assert "nope_seg" in result.output
    assert not (tmp_path / "m.html").exists() and not (tmp_path / "ents").exists()


def test_without_flag_output_unchanged(tmp_path: Path) -> None:
    pack = load_scope_pack(EXAMPLE_SCOPE)
    assert render_mosaic(pack, iter(SCORES), iter(EXPORT), heading=None) == render_mosaic(
        pack, iter(SCORES), iter(EXPORT)
    )
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert page == render_mosaic(pack, iter(SCORES), iter(EXPORT))
    assert "segment:" not in page


def test_heading_is_escaped() -> None:
    pack = load_scope_pack(EXAMPLE_SCOPE)
    page = render_mosaic(pack, iter(SCORES), iter(EXPORT), heading="segment: <b>&")
    assert "segment: &lt;b&gt;&amp;" in page and "<b>&" not in page
