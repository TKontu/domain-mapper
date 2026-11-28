"""GS-03: blind review pages generated from a GS-01 cells file."""

from __future__ import annotations

import http.client
import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from core.gold_pages import write_gold_pages
from typer.testing import CliRunner

from app import cli
from app import review_server as rs
from core.scope_pack import load_scope_pack

EXAMPLE = Path(__file__).resolve().parents[1] / "scopes" / "_example"


def _cells(tmp_path: Path) -> Path:
    rows = [
        {
            "cell_id": "c-0000000000000001",
            "primary_domain": "example.com",
            "criterion_id": "iso17100",
            "criterion": "Is the provider certified to ISO 17100?",
            "evidence": [
                {
                    "chunk_text": "We are <script>alert('x')</script> certified & proud.",
                    "url": "https://example.com/quality",
                    "fetched_at": "2026-11-01T00:00:00Z",
                },
                {
                    "chunk_text": "Plain text.",
                    "url": "javascript:alert(1)",
                    "fetched_at": "2026-11-02T00:00:00Z",
                },
            ],
            "status": "red_flag",
            "confidence": 0.123456,
            "assessment_id": 987654,
        },
        {
            "cell_id": "c-0000000000000002",
            "primary_domain": "example.org",
            "criterion_id": "public_terms",
            "criterion": "Are general terms of service publicly available?",
            "evidence": [],
        },
    ]
    path = tmp_path / "cells.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _gen(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    write_gold_pages(load_scope_pack(EXAMPLE), _cells(tmp_path), root)
    return root


def test_pages_are_blind(tmp_path: Path) -> None:
    html = (_gen(tmp_path) / "pages" / "c-0000000000000001.html").read_text(encoding="utf-8")
    for bad in ("confidence", "assessment_id", "0.123456", "987654"):
        assert bad not in html
    # red_flag appears only as a scale option, never marked as preselected
    assert not re.search(r"<input[^>]*\bchecked", html)


def test_escaping_and_safe_urls(tmp_path: Path) -> None:
    html = (_gen(tmp_path) / "pages" / "c-0000000000000001.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
    assert 'href="https://example.com/quality"' in html
    assert 'href="javascript:' not in html
    hrefs = re.findall(r'href="([^"]*)"', html)
    assert all(h.startswith(("http://", "https://", "/p/", "/")) for h in hrefs)
    assert "2026-11-02T00:00:00Z" in html
    assert "http-equiv" not in html and "<link" not in html and " src=" not in html


def test_scale_and_criterion_shown(tmp_path: Path) -> None:
    html = (_gen(tmp_path) / "pages" / "c-0000000000000001.html").read_text(encoding="utf-8")
    assert "Is the provider certified to ISO 17100?" in html
    for sid, label in (("strong", "Strong"), ("red_flag", "Red flag"), ("na", "Does not apply")):
        assert f'value="{sid}"' in html and label in html


def test_pages_json_matches_server_registry(tmp_path: Path) -> None:
    root = _gen(tmp_path)
    pages = json.loads((root / "pages.json").read_text(encoding="utf-8"))
    assert list(pages) == ["c-0000000000000001", "c-0000000000000002"]
    for pid in pages:
        assert rs._PAGE_ID.fullmatch(pid)
    assert rs._registered(root) == pages
    assert (root / "INSTRUCTIONS.md").read_text(encoding="utf-8").startswith("# ")


def test_bad_cell_id_rejected(tmp_path: Path) -> None:
    cells = tmp_path / "cells.jsonl"
    cells.write_text(json.dumps({"cell_id": "../x", "criterion": "q"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_gold_pages(load_scope_pack(EXAMPLE), cells, tmp_path / "root")


def test_submit_script_shape_and_next_link(tmp_path: Path) -> None:
    root = _gen(tmp_path)
    html = (root / "pages" / "c-0000000000000001.html").read_text(encoding="utf-8")
    assert "fetch('/api/answer/' + CELL" in html
    assert 'const CELL = "c-0000000000000001";' in html
    assert "JSON.stringify({cell_id: CELL, status: status, quote: quote, note: note})" in html
    assert 'href="/p/c-0000000000000002"' in html
    last = (root / "pages" / "c-0000000000000002.html").read_text(encoding="utf-8")
    assert 'href="/"' in last


def test_round_trip_through_review_server(tmp_path: Path) -> None:
    root = _gen(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rs._make_handler(root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/p/c-0000000000000001")
        resp = conn.getresponse()
        assert resp.status == 200 and b"ISO 17100" in resp.read()
        body = {"cell_id": "c-0000000000000001", "status": "weak", "quote": "q", "note": "n"}
        conn.request(
            "POST",
            "/api/answer/c-0000000000000001",
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 200
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
    answers = json.loads((root / "answers.json").read_text(encoding="utf-8"))
    assert answers["c-0000000000000001"]["status"] == "weak"


def test_deterministic(tmp_path: Path) -> None:
    a = _gen(tmp_path)
    snap = {p.relative_to(a): p.read_bytes() for p in sorted(a.rglob("*")) if p.is_file()}
    b = tmp_path / "root2"
    write_gold_pages(load_scope_pack(EXAMPLE), tmp_path / "cells.jsonl", b)
    assert {p.relative_to(b): p.read_bytes() for p in sorted(b.rglob("*")) if p.is_file()} == snap


def test_cli_command(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    root = tmp_path / "out"
    res = CliRunner().invoke(
        cli.app,
        ["gold-pages", "--scope", str(EXAMPLE), "--cells", str(cells), "--root", str(root)],
    )
    assert res.exit_code == 0, res.output
    assert (root / "pages" / "c-0000000000000002.html").is_file()
    assert "2 pages" in res.output
