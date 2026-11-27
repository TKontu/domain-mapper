"""Tests for FE-04: ``layers`` in scores-export lines and ``landscape mosaic --layer``; offline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from app.mosaic import filter_by_layers, render_mosaic
from core.presence_store import record_presence
from core.scope_pack import load_scope_pack
from tests.test_cli_scores_export import (  # noqa: F401  (engine is a fixture)
    EXAMPLE_SCOPE,
    add_entity,
    add_run,
    add_screen,
    engine,
    export,
)

SCORES = [
    {"entity_id": 3, "segments": ["agency"], "layers": ["region", "global"], "topics": []},
    {"entity_id": 1, "segments": ["agency"], "layers": ["home", "region", "global"], "topics": []},
    {"entity_id": 2, "segments": ["freelancer_network"], "layers": [], "topics": []},
    {"entity_id": 4, "segments": ["agency"], "topics": []},
    {"entity_id": 5, "segments": ["freelancer_network"], "layers": ["home"], "topics": []},
]
EXPORT = [{"entity_id": 1, "criterion_id": "iso17100", "status": "strong", "evidence": []}]


def _run(tmp_path: Path, *extra: str):
    scores = tmp_path / "scores.jsonl"
    exp = tmp_path / "export.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in SCORES), encoding="utf-8")
    exp.write_text("".join(json.dumps(r) + "\n" for r in EXPORT), encoding="utf-8")
    args = ["mosaic", "--scope", EXAMPLE_SCOPE, "--scores", str(scores), "--export", str(exp)]
    return CliRunner().invoke(cli.app, [*args, "--out", str(tmp_path / "m.html"), *extra])


def _rows(tmp_path: Path) -> list[str]:
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    return [e for e in ("1", "2", "3", "4", "5") if f'<tr data-entity="{e}">' in page]


def test_scores_export_lines_carry_layers_fetched_once_per_batch(
    engine, tmp_path, monkeypatch
) -> None:
    rid = add_run(engine)
    a, b, c = (add_entity(engine, n) for n in ("a", "b", "c"))
    for e in (a, b, c):
        add_screen(engine, rid, e, ["agency"])
    record_presence(engine, a, "XA", "hq")
    record_presence(engine, c, "XB", "sales")
    calls: list[list[int]] = []
    real = cli.presences_many

    def counting(engine_, ids):
        calls.append(list(ids))
        return real(engine_, ids)

    monkeypatch.setattr(cli, "presences_many", counting)
    monkeypatch.setattr(cli, "SCORES_EXPORT_BATCH", 2)
    out = tmp_path / "s.jsonl"
    result = export(out)
    assert result.exit_code == 0, result.output
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [(x["entity_id"], x["layers"]) for x in lines] == [
        (a, ["home", "region", "global"]),
        (b, []),
        (c, ["region", "global"]),
    ]
    assert calls == [[a, b], [c]]


def test_filter_by_layers_input_order_union_and_missing_key_dropped() -> None:
    assert [r["entity_id"] for r in filter_by_layers(iter(SCORES), {"home"})] == [1, 5]
    both = filter_by_layers(iter(SCORES), {"home", "region"})
    assert [r["entity_id"] for r in both] == [3, 1, 5]


def test_cli_layer_repeatable_union_and_entity_pages(tmp_path: Path) -> None:
    ents = tmp_path / "ents"
    result = _run(tmp_path, "--layer", "home", "--layer", "region", "--entity-dir", str(ents))
    assert result.exit_code == 0, result.output
    assert _rows(tmp_path) == ["1", "3", "5"]
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert page.index('data-entity="3"') < page.index('data-entity="1"')
    assert sorted(p.name.split("-")[0] for p in ents.iterdir()) == ["1", "3", "5"]


def test_cli_layer_and_segment_intersect(tmp_path: Path) -> None:
    result = _run(tmp_path, "--layer", "home", "--segment", "agency")
    assert result.exit_code == 0, result.output
    assert _rows(tmp_path) == ["1"]
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert "segment: Agencies" in page and "layer: home" in page


def test_cli_unknown_layer_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    result = _run(tmp_path, "--layer", "nope_layer", "--entity-dir", str(tmp_path / "ents"))
    assert result.exit_code == 2
    assert "nope_layer" in result.output
    assert not (tmp_path / "m.html").exists() and not (tmp_path / "ents").exists()


def test_without_flag_output_unchanged(tmp_path: Path) -> None:
    result = _run(tmp_path, "--entity-dir", str(tmp_path / "ents"))
    assert result.exit_code == 0, result.output
    pack = load_scope_pack(Path(EXAMPLE_SCOPE))
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert "<h2" not in page
    assert _rows(tmp_path) == ["1", "2", "3", "4", "5"]
    assert render_mosaic(pack, iter(SCORES), iter(EXPORT)) == render_mosaic(
        pack, iter(SCORES), iter(EXPORT), heading=None
    )


def test_heading_layer_label_escaped(tmp_path: Path) -> None:
    """Layers carry no ``label`` in the scope pack, so the id is the label; it is escaped."""
    scope = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, scope)
    y = scope / "scope.yaml"
    y.write_text(
        y.read_text(encoding="utf-8").replace("{id: home,", '{id: "<h&o>",'), encoding="utf-8"
    )
    load_scope_pack(scope)
    rows = [{"entity_id": 1, "layers": ["<h&o>"], "topics": []}]
    scores = tmp_path / "s.jsonl"
    exp = tmp_path / "e.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    exp.write_text("", encoding="utf-8")
    out = tmp_path / "m.html"
    args = ["mosaic", "--scope", str(scope), "--scores", str(scores), "--export", str(exp)]
    result = CliRunner().invoke(cli.app, [*args, "--out", str(out), "--layer", "<h&o>"])
    assert result.exit_code == 0, result.output
    page = out.read_text(encoding="utf-8")
    assert "layer: &lt;h&amp;o&gt;" in page and "<h&o>" not in page
    assert '<tr data-entity="1">' in page
