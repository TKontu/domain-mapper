"""Tests for FE-02 per-entity static pages linked from the mosaic; offline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app import cli
from app.mosaic import entity_slug, render_entity_page, render_mosaic
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"
# sha256 of render_mosaic(SCORES, EXPORT) at base ccc90c0 (FE-01, before FE-02).
BASE_MOSAIC_SHA256 = "f63dbaf84ec4d65a6d2ed33ab4a8370d4654c4b1e23310447e075594678faba5"

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
NASTY = ["../../etc/x", "a b", "<x>", "/abs", "..", ""]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(tmp_path: Path, scores: list[dict], export: list[dict], *extra: str):  # noqa: ANN202
    s = _write(tmp_path / "s.jsonl", scores)
    e = _write(tmp_path / "e.jsonl", export)
    args = ["mosaic", "--scope", str(EXAMPLE_SCOPE), "--scores", str(s), "--export", str(e)]
    return CliRunner().invoke(cli.app, [*args, "--out", str(tmp_path / "page.html"), *extra])


def test_without_flag_mosaic_is_unchanged() -> None:
    page = render_mosaic(load_scope_pack(EXAMPLE_SCOPE), iter(SCORES), iter(EXPORT))
    assert hashlib.sha256(page.encode()).hexdigest() == BASE_MOSAIC_SHA256


def test_entity_page_lists_every_criterion_in_rubric_order() -> None:
    pack = load_scope_pack(EXAMPLE_SCOPE)
    page = render_entity_page(pack, SCORES[0], {"iso17100": EXPORT[0]}, "../page.html")
    expected = [c.id for t in pack.rubric.topics for c in t.criteria]
    assert re.findall(r'<tr data-criterion="([^"]+)"', page) == expected
    assert '<h1 data-entity="7">7</h1>' in page
    assert "1.00 / 1.00" in page
    row = page.split('data-criterion="iso17100"')[1].split("</tr>")[0]
    assert 'data-color="green"' in row and "Strong" in row and "0.9" in row
    assert '<a href="https://example.org/cert"' in row and "2026-01-02T03:04:05+00:00" in row
    assert page.count("not assessed") == len(expected) - 1
    assert '<a href="../page.html">' in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert re.findall(r"https?://[^\s\"'<>]+", page) == ["https://example.org/cert"] * 2


def test_cli_writes_linked_pages_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[str] = []
    real = cli.os.replace

    def spy(src, dst):  # noqa: ANN001
        assert str(src) == f"{dst}.tmp"
        replaced.append(str(dst))
        real(src, dst)

    monkeypatch.setattr(cli.os, "replace", spy)
    result = _run(tmp_path, SCORES, EXPORT, "--entity-dir", str(tmp_path / "ent" / "sub"))
    assert result.exit_code == 0, result.output
    pages = sorted((tmp_path / "ent" / "sub").iterdir())
    assert [p.name for p in pages] == sorted(f"{entity_slug(e)}.html" for e in ("7", "9"))
    assert len(replaced) == 3 and not list(tmp_path.rglob("*.tmp"))
    mosaic = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert f'<a href="ent/sub/{entity_slug("7")}.html">7</a>' in mosaic
    for p in pages:
        assert '<a href="../../page.html">back to mosaic</a>' in p.read_text(encoding="utf-8")
        assert (p.parent / "../../page.html").resolve() == (tmp_path / "page.html").resolve()
    assert re.findall(r"https?://[^\s\"'<>]+", mosaic) == ["https://example.org/cert"] * 2


def test_nasty_entity_ids_yield_safe_files_and_escaped_text(tmp_path: Path) -> None:
    scores = [{"entity_id": e, "topics": []} for e in NASTY]
    out_dir = tmp_path / "ent"
    result = _run(tmp_path, scores, [], "--entity-dir", str(out_dir))
    assert result.exit_code == 0, result.output
    files = list(out_dir.iterdir())
    assert len(files) == len(NASTY)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["e.jsonl", "ent", "page.html", "s.jsonl"]
    for p in files:
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.html", p.name)
        assert p.resolve().parent == out_dir.resolve()
    text = (out_dir / f"{entity_slug('<x>')}.html").read_text(encoding="utf-8")
    assert "<x>" not in text and "&lt;x&gt;" in text


def test_cli_exit_2_with_entity_dir(tmp_path: Path) -> None:
    s = _write(tmp_path / "s.jsonl", SCORES)
    runner = CliRunner()
    base = ["mosaic", "--scores", str(s), "--out", str(tmp_path / "p.html")]
    missing = runner.invoke(
        cli.app,
        [*base, "--scope", str(EXAMPLE_SCOPE), "--export", str(tmp_path / "nope.jsonl"),
         "--entity-dir", str(tmp_path / "ent")],
    )  # fmt: skip
    assert missing.exit_code == 2
    (tmp_path / "bad").mkdir()
    bad = runner.invoke(
        cli.app,
        [*base, "--scope", str(tmp_path / "bad"), "--export", str(s),
         "--entity-dir", str(tmp_path / "ent")],
    )  # fmt: skip
    assert bad.exit_code == 2
    assert not (tmp_path / "ent").exists() and not (tmp_path / "p.html").exists()
