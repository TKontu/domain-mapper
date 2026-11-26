"""Tests for the FE-01 static mosaic page (``app.mosaic`` and ``landscape mosaic``); offline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app import cli
from app.mosaic import render_mosaic
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"

SCORES = [
    {
        "entity_id": 7,
        "segments": ["agency"],
        "rubric_version": "0.0.1",
        "topics": [
            {"topic_id": "certification", "score": 1.0, "coverage": 1.0},
            {"topic_id": "terms", "score": None, "coverage": 0.0},
        ],
    },
    {"entity_id": 9, "segments": ["agency"], "rubric_version": "0.0.1", "topics": []},
]
EXPORT = [
    {
        "entity_id": 7,
        "criterion_id": "iso17100",
        "status": "strong",
        "confidence": 0.9,
        "rationale": "register <b>entry</b>",
        "evidence": [
            {
                "quote": "certified <script>alert(1)</script>",
                "url": "https://example.org/cert",
                "fetched_at": "2026-01-02T03:04:05+00:00",
            }
        ],
    }
]


@pytest.fixture
def page() -> str:
    return render_mosaic(load_scope_pack(EXAMPLE_SCOPE), iter(SCORES), iter(EXPORT))


def test_lists_criteria_in_rubric_order_and_every_entity(page: str) -> None:
    pack = load_scope_pack(EXAMPLE_SCOPE)
    expected = [c.id for t in pack.rubric.topics for c in t.criteria]
    assert re.findall(r'data-criterion="([^"]+)"', page) == expected
    assert re.findall(r'data-topic="([^"]+)"', page) == [t.id for t in pack.rubric.topics]
    assert re.findall(r'data-entity="([^"]+)"', page) == ["7", "9"]
    assert "1.00 / 1.00" in page and "- / 0.00" in page


def test_cell_carries_scale_label_color_and_evidence(page: str) -> None:
    cell = re.search(r'<td class="cell"[^>]*data-status="strong".*?</td>', page, re.S)
    assert cell is not None
    text = cell.group(0)
    assert 'data-color="green"' in text and "Strong" in text
    assert '<a href="https://example.org/cert"' in text
    assert "2026-01-02T03:04:05+00:00" in text and "0.9" in text


def test_jsonl_text_is_escaped(page: str) -> None:
    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "register &lt;b&gt;entry&lt;/b&gt;" in page


def test_unassessed_criterion_is_empty_cell(page: str) -> None:
    row9 = page.split('data-entity="9"')[1].split("</tr>")[0]
    assert row9.count('<td class="cell empty"></td>') == 2
    assert "data-status" not in row9


def test_only_http_references_are_evidence_links(page: str) -> None:
    assert re.findall(r"https?://[^\s\"'<>]+", page) == ["https://example.org/cert"] * 2
    assert "<script" not in page and "<link" not in page


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_cli_writes_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scores = _write(tmp_path / "s.jsonl", SCORES)
    export = _write(tmp_path / "e.jsonl", EXPORT)
    out = tmp_path / "page.html"
    replaced: list[tuple[str, str]] = []
    real = cli.os.replace

    def spy(src, dst):  # noqa: ANN001
        replaced.append((str(src), str(dst)))
        real(src, dst)

    monkeypatch.setattr(cli.os, "replace", spy)
    args = ["mosaic", "--scope", str(EXAMPLE_SCOPE), "--scores", str(scores)]
    result = CliRunner().invoke(cli.app, [*args, "--export", str(export), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert replaced == [(f"{out}.tmp", str(out))]
    assert 'data-entity="7"' in out.read_text(encoding="utf-8")
    assert not Path(f"{out}.tmp").exists()


def test_cli_exits_2_on_missing_input_or_bad_scope(tmp_path: Path) -> None:
    scores = _write(tmp_path / "s.jsonl", SCORES)
    out = tmp_path / "page.html"
    runner = CliRunner()
    missing = runner.invoke(
        cli.app,
        ["mosaic", "--scope", str(EXAMPLE_SCOPE), "--scores", str(scores),
         "--export", str(tmp_path / "nope.jsonl"), "--out", str(out)],
    )  # fmt: skip
    assert missing.exit_code == 2
    bad = tmp_path / "badscope"
    bad.mkdir()
    bad_scope = runner.invoke(
        cli.app,
        ["mosaic", "--scope", str(bad), "--scores", str(scores),
         "--export", str(scores), "--out", str(out)],
    )  # fmt: skip
    assert bad_scope.exit_code == 2
    assert not out.exists()
