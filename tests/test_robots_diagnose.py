"""Offline tests for scripts/robots_diagnose.py classification (M1-02)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robots_diagnose.py"
_spec = importlib.util.spec_from_file_location("robots_diagnose", _PATH)
assert _spec is not None and _spec.loader is not None
rd = importlib.util.module_from_spec(_spec)
sys.modules["robots_diagnose"] = rd
_spec.loader.exec_module(rd)

URL = "https://example.test/"


def _fetch(status: int, body: str = ""):
    def fetch(_url: str) -> tuple[int, bytes]:
        return status, body.encode()

    return fetch


def test_blanket_star() -> None:
    result = rd.classify(URL, _fetch(200, "User-agent: *\nDisallow: /\n"))
    assert result.category == "blanket"
    assert result.allowed_paths == ()


def test_ua_specific_blanket() -> None:
    text = "User-agent: domain-mapper\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    assert rd.classify(URL, _fetch(200, text)).category == "blanket"


def test_other_ua_blanket_is_allow() -> None:
    text = "User-agent: otherbot\nDisallow: /\n"
    assert rd.classify(URL, _fetch(200, text)).category == "allow"


def test_partial_records_allowed_paths() -> None:
    text = "User-agent: *\nAllow: /about\nAllow: /contact\nDisallow: /\n"
    result = rd.classify(URL, _fetch(200, text))
    assert result.category == "partial"
    assert set(result.allowed_paths) == {"/about", "/contact"}


def test_allow() -> None:
    result = rd.classify(URL, _fetch(200, "User-agent: *\nDisallow: /private\n"))
    assert result.category == "allow"
    assert result.status_class == "2xx"


def test_missing_404_is_unreachable_4xx() -> None:
    result = rd.classify(URL, _fetch(404))
    assert result.category == "unreachable"
    assert result.status_class == "4xx"


def test_server_error_is_unreachable_5xx() -> None:
    assert rd.classify(URL, _fetch(503)).status_class == "5xx"


def test_transport_error_is_unreachable_error() -> None:
    def boom(_url: str) -> tuple[int, bytes]:
        raise OSError("down")

    result = rd.classify(URL, boom)
    assert result.category == "unreachable"
    assert result.status_class == "error"


def test_entity_urls_read_only(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE entity (id INTEGER, primary_domain TEXT)")
    con.executemany("INSERT INTO entity VALUES (?, ?)", [(2, "b.test"), (1, "a.test")])
    con.commit()
    con.close()
    assert rd.entity_urls(db) == ["https://a.test/", "https://b.test/"]
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("DELETE FROM entity")


def test_main_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE entity (id INTEGER, primary_domain TEXT)")
    con.executemany("INSERT INTO entity VALUES (?, ?)", [(1, "a.test"), (2, "b.test")])
    con.commit()
    con.close()
    out = tmp_path / "out"
    rc = rd.main(["--db", str(db), "--out", str(out)], fetch=_fetch(404))
    assert rc == 0
    printed = capsys.readouterr().out
    assert "a.test" not in printed
    assert '"unreachable": 2' in printed
    assert (out / "summary.json").exists() and (out / "hosts.jsonl").exists()
