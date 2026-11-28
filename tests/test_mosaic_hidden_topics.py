"""Tests for FE-07: topic score cells mark that they include hidden experimental criteria."""

from __future__ import annotations

from pathlib import Path

from tests.test_mosaic_release import _report, _run, _setup

MARK = 'data-hidden-criteria="1" title="includes 1 hidden experimental criteria"'


def test_mosaic_and_entity_topic_cells_marked(tmp_path: Path) -> None:
    report = _report({"iso17100": "release", "public_terms": "experimental"})
    ents = tmp_path / "ents"
    rc, _ = _run([*_setup(tmp_path, report), "--hide-experimental", "--entity-dir", str(ents)])
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert f'<td class="topic" {MARK}>0.50 / 1.00 *</td>' in page
    assert '<td class="topic">1.00 / 1.00</td>' in page
    assert page.count("data-hidden-criteria") == 1
    assert page.count('<th class="topic">score / coverage</th>') == 2
    (entity_page,) = list(ents.glob("*.html"))
    text = entity_page.read_text(encoding="utf-8")
    assert f'<th class="topic" colspan="6" {MARK}>' in text
    assert text.count("data-hidden-criteria") == 1
    assert "1.00 / 1.00</th>" in text


def test_no_marker_without_hide_flag(tmp_path: Path) -> None:
    report = _report({"iso17100": "release", "public_terms": "experimental"})
    ents = tmp_path / "ents"
    rc, _ = _run([*_setup(tmp_path, report), "--entity-dir", str(ents)])
    assert rc == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "data-hidden-criteria" not in page
    assert " *<" not in page
    for p in ents.glob("*.html"):
        assert "data-hidden-criteria" not in p.read_text(encoding="utf-8")
