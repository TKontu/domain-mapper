"""GS-02 review server: offline tests on loopback with ephemeral ports."""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from app import review_server as rs

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "alpha.html").write_text("<p>alpha page</p>", encoding="utf-8")
    (tmp_path / "pages" / "hidden.html").write_text("<p>hidden</p>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    (tmp_path / "pages.json").write_text(json.dumps({"alpha": "Alpha review"}), encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(root: Path) -> Iterator[tuple[Path, int]]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rs._make_handler(root))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield root, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _req(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(
        method,
        path,
        body=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_lists_and_serves_registered_page(server: tuple[Path, int]) -> None:
    _, port = server
    status, body = _req(port, "GET", "/")
    assert status == 200 and b"/p/alpha" in body and b"Alpha review" in body
    assert b"hidden" not in body
    status, body = _req(port, "GET", "/p/alpha")
    assert status == 200 and body == b"<p>alpha page</p>"


@pytest.mark.parametrize(
    "path",
    [
        "/p/hidden",
        "/p/missing",
        "/p/../secret.txt",
        "/p/%2e%2e%2fsecret.txt",
        "/secret.txt",
        "/pages/alpha.html",
    ],
)
def test_unregistered_and_traversal_404(server: tuple[Path, int], path: str) -> None:
    _, port = server
    assert _req(port, "GET", path)[0] == 404


def test_post_writes_answers_and_one_history_file(server: tuple[Path, int]) -> None:
    root, port = server
    status, _ = _req(port, "POST", "/api/answer/alpha", json.dumps({"q1": "yes"}).encode())
    assert status == 200
    assert json.loads((root / "answers.json").read_text()) == {"alpha": {"q1": "yes"}}
    assert len(list((root / "history").iterdir())) == 1
    status, _ = _req(port, "POST", "/api/answer/alpha", json.dumps({"q1": "no"}).encode())
    assert status == 200
    assert json.loads((root / "answers.json").read_text()) == {"alpha": {"q1": "no"}}
    files = sorted((root / "history").iterdir())
    assert len(files) == 2
    assert json.loads(files[0].read_text())["answer"] in ({"q1": "yes"}, {"q1": "no"})


@pytest.mark.parametrize(
    "body", [b"{not json", b"[1, 2]", b'"text"', b"\xff\xfe", b"x" * (64 * 1024 + 1)]
)
def test_bad_body_400_and_no_write(server: tuple[Path, int], body: bytes) -> None:
    root, port = server
    assert _req(port, "POST", "/api/answer/alpha", body)[0] == 400
    assert not (root / "answers.json").exists()
    assert not (root / "history").exists()


def test_post_unregistered_page_404(server: tuple[Path, int]) -> None:
    root, port = server
    assert _req(port, "POST", "/api/answer/hidden", b"{}")[0] == 404
    assert not (root / "answers.json").exists()


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "10.1.2.3", "172.16.5.5", "192.168.1.10", "169.254.1.1", "::1"]
)
def test_private_binds_allowed(host: str) -> None:
    rs._check_bind(host, allow_any_lan=False)


@pytest.mark.parametrize("host", ["8.8.8.8", "172.32.0.1", "2001:4860::1", "example.org"])
def test_public_bind_refused(host: str) -> None:
    with pytest.raises(rs._BindRefused):
        rs._check_bind(host, allow_any_lan=True)


def test_wildcard_needs_flag(root: Path) -> None:
    with pytest.raises(rs._BindRefused):
        rs._check_bind("0.0.0.0", allow_any_lan=False)
    rs._check_bind("0.0.0.0", allow_any_lan=True)
    assert rs.main(["serve", "--root", str(root), "--host", "8.8.8.8", "--port", "0"]) == 2
    assert rs.main(["serve", "--root", str(root), "--host", "0.0.0.0", "--port", "0"]) == 2


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "app.review_server", *args]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30)


def test_start_status_stop_round_trip(root: Path) -> None:
    assert _cli("status", "--root", str(root)).returncode == 1
    started = _cli("start", "--root", str(root), "--host", "127.0.0.1", "--port", "0")
    try:
        assert started.returncode == 0, started.stderr
        info = json.loads((root / ".review_server.pid").read_text())
        assert _req(info["port"], "GET", "/p/alpha")[0] == 200
        status = _cli("status", "--root", str(root))
        assert status.returncode == 0 and str(info["pid"]) in status.stdout
        assert _cli("start", "--root", str(root), "--port", "0").returncode == 1
    finally:
        stopped = _cli("stop", "--root", str(root))
    assert stopped.returncode == 0
    assert not (root / ".review_server.pid").exists()
    assert _cli("status", "--root", str(root)).returncode == 1
