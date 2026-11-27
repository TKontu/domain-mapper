"""Tests for FE-05: optional scope-pack ``Layer.label`` named in the mosaic heading; offline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = "scopes/_example"
ROWS = [
    {"entity_id": 1, "layers": ["home"], "topics": []},
    {"entity_id": 2, "layers": ["region"], "topics": []},
]


def _scope(tmp_path: Path, old: str, new: str) -> Path:
    scope = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, scope)
    y = scope / "scope.yaml"
    text = y.read_text(encoding="utf-8")
    assert old in text
    y.write_text(text.replace(old, new), encoding="utf-8")
    load_scope_pack(scope)
    return scope


def _run(tmp_path: Path, scope: Path | str, *extra: str):
    scores = tmp_path / "s.jsonl"
    exp = tmp_path / "e.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    exp.write_text("", encoding="utf-8")
    args = ["mosaic", "--scope", str(scope), "--scores", str(scores), "--export", str(exp)]
    return CliRunner().invoke(cli.app, [*args, "--out", str(tmp_path / "m.html"), *extra])


def _page(tmp_path: Path) -> str:
    return (tmp_path / "m.html").read_text(encoding="utf-8")


def test_example_layers_have_labels_and_validate() -> None:
    pack = load_scope_pack(Path(EXAMPLE_SCOPE))
    assert all(la.label for la in pack.scope.layers)
    result = CliRunner().invoke(cli.app, ["validate", "--scope", EXAMPLE_SCOPE])
    assert result.exit_code == 0, result.output


def test_heading_uses_layer_label(tmp_path: Path) -> None:
    result = _run(tmp_path, EXAMPLE_SCOPE, "--layer", "home", "--layer", "region")
    assert result.exit_code == 0, result.output
    assert "layer: Home market, Region" in _page(tmp_path)


def test_heading_falls_back_to_id_without_label(tmp_path: Path) -> None:
    scope = _scope(tmp_path, "label: Home market, ", "")
    result = _run(tmp_path, scope, "--layer", "home", "--layer", "region")
    assert result.exit_code == 0, result.output
    assert "layer: home, Region" in _page(tmp_path)


def test_heading_label_is_escaped(tmp_path: Path) -> None:
    scope = _scope(tmp_path, "label: Home market,", 'label: "<H&m>",')
    result = _run(tmp_path, scope, "--layer", "home")
    assert result.exit_code == 0, result.output
    page = _page(tmp_path)
    assert "layer: &lt;H&amp;m&gt;" in page and "<H&m>" not in page


def test_without_layer_flag_no_heading(tmp_path: Path) -> None:
    result = _run(tmp_path, EXAMPLE_SCOPE)
    assert result.exit_code == 0, result.output
    assert "<h2" not in _page(tmp_path)
