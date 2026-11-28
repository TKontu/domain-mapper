"""LAN-only review server (GS-02): serve registered static pages and collect JSON answers.

``python3 -m app.review_server {start,stop,status,serve} --root DIR [--host H] [--port 8741]``

The root holds ``pages.json`` (a JSON object mapping page id to title; only listed pages are
registered) and ``pages/<page_id>.html``. ``GET /`` lists registered pages, ``GET /p/<page_id>``
serves one, ``POST /api/answer/<page_id>`` with a JSON object body (at most 64 KiB) stores it in
``answers.json`` keyed by page id (atomic tmp + ``os.replace``, last write wins) and appends one
timestamped file under ``history/``. Binding is limited to loopback, RFC1918 and link-local
addresses; ``0.0.0.0``/``::`` require ``--allow-any-lan`` and public addresses are always refused.
``start`` daemonises ``serve`` and records ``{pid, host, port}`` in ``.review_server.pid`` under
the root; ``stop`` and ``status`` read it. Stdlib only, no authentication.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_DEFAULT_PORT = 8741
_MAX_BODY = 64 * 1024
_PID_NAME = ".review_server.pid"
_PAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_LOCK = threading.Lock()


class _BindRefused(ValueError):
    pass


def _check_bind(host: str, allow_any_lan: bool) -> None:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError as exc:
        raise _BindRefused(f"host must be an IP address, got {host!r}") from exc
    if addr.is_unspecified:
        if not allow_any_lan:
            raise _BindRefused(f"refusing wildcard bind {host}; pass --allow-any-lan")
        return
    if not isinstance(addr, ipaddress.IPv4Address):
        raise _BindRefused(f"refusing non-IPv4 bind address {host}; server is IPv4 only")
    if addr.is_loopback or addr.is_link_local:
        return
    if any(
        addr in ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    ):
        return
    raise _BindRefused(f"refusing non-private bind address {host}")


def _registered(root: Path) -> dict[str, str]:
    try:
        data = json.loads((root / "pages.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in data.items()
        if isinstance(k, str) and _PAGE_ID.fullmatch(k) and (root / "pages" / f"{k}.html").is_file()
    }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _store_answer(root: Path, page_id: str, answer: dict[str, Any]) -> None:
    with _LOCK:
        path = root / "answers.json"
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        except (OSError, ValueError):
            current = {}
        current[page_id] = answer
        _atomic_write(path, json.dumps(current, indent=2, sort_keys=True) + "\n")
        history = root / "history"
        history.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        n = 0
        while (target := history / f"{stamp}-{page_id}-{n}.json").exists():
            n += 1
        record = {"page_id": page_id, "received_at": stamp, "answer": answer}
        _atomic_write(target, json.dumps(record, indent=2, sort_keys=True) + "\n")


def _make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj: object) -> None:
            self._send(code, json.dumps(obj), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            pages = _registered(root)
            if path == "/":
                items = "".join(
                    f'<li><a href="/p/{escape(k)}">{escape(v)}</a></li>'
                    for k, v in sorted(pages.items())
                )
                head = "<!doctype html><title>Review pages</title><h1>Review pages</h1>"
                html = f"{head}<ul>{items}</ul>"
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path.startswith("/p/") and path[3:] in pages:
                text = (root / "pages" / f"{path[3:]}.html").read_text(encoding="utf-8")
                self._send(200, text, "text/html; charset=utf-8")
                return
            self._send(404, "not found\n")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            prefix = "/api/answer/"
            page_id = path[len(prefix) :] if path.startswith(prefix) else ""
            if page_id not in _registered(root):
                self._json(404, {"error": "unknown page"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(411, {"error": "content length required"})
                return
            if length < 0 or length > _MAX_BODY:
                self.close_connection = True
                self._json(400, {"error": "body too large"})
                return
            raw = self.rfile.read(length)
            try:
                answer = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self._json(400, {"error": "invalid JSON"})
                return
            if not isinstance(answer, dict):
                self._json(400, {"error": "body must be a JSON object"})
                return
            _store_answer(root, page_id, answer)
            self._json(200, {"ok": True, "page_id": page_id})

    return Handler


def _pid_path(root: Path) -> Path:
    return root / _PID_NAME


def _read_pid(root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(_pid_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("pid"), int) else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return stat.rpartition(")")[2].split()[:1] != ["Z"]  # a zombie has exited


def _is_server(pid: int) -> bool:
    """True only if pid is alive and its cmdline shows it is this review server."""
    if not _alive(pid):
        return False
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return "app.review_server" in raw.replace(b"\0", b" ").decode("utf-8", "replace")


def _serve(root: Path, host: str, port: int) -> int:
    server = ThreadingHTTPServer((host, port), _make_handler(root))
    actual = server.server_address[1]
    _atomic_write(_pid_path(root), json.dumps({"pid": os.getpid(), "host": host, "port": actual}))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"serving {root} on http://{host}:{actual}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        info = _read_pid(root)
        if info and info["pid"] == os.getpid():
            _pid_path(root).unlink(missing_ok=True)
    return 0


def _start(root: Path, args: argparse.Namespace) -> int:
    info = _read_pid(root)
    if info and _is_server(info["pid"]):
        print(f"already running pid {info['pid']} on {info.get('host')}:{info.get('port')}")
        return 1
    _pid_path(root).unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "app.review_server", "serve", "--root", str(root)]
    cmd += ["--host", args.host, "--port", str(args.port)]
    if args.allow_any_lan:
        cmd.append("--allow-any-lan")
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        info = _read_pid(root)
        if info and info["pid"] == proc.pid:
            print(f"started pid {proc.pid} on http://{info['host']}:{info['port']}/")
            return 0
        if proc.poll() is not None:
            print(f"server exited with code {proc.returncode}", file=sys.stderr)
            return 1
        time.sleep(0.05)
    proc.terminate()
    print("server did not start in time", file=sys.stderr)
    return 1


def _status(root: Path) -> int:
    info = _read_pid(root)
    if info and _is_server(info["pid"]):
        print(f"running pid {info['pid']} on http://{info.get('host')}:{info.get('port')}/")
        return 0
    print("not running")
    return 1


def _stop(root: Path) -> int:
    info = _read_pid(root)
    if not info or not _is_server(info["pid"]):
        _pid_path(root).unlink(missing_ok=True)
        print("not running")
        return 1
    os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(info["pid"]):
        time.sleep(0.05)
    if _alive(info["pid"]):
        msg = f"pid {info['pid']} still alive after SIGTERM; not removing pid file"
        print(msg, file=sys.stderr)
        return 1
    _pid_path(root).unlink(missing_ok=True)
    print(f"stopped pid {info['pid']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m app.review_server", description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "serve"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--allow-any-lan", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command in ("start", "serve"):
        try:
            _check_bind(args.host, args.allow_any_lan)
        except _BindRefused as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not root.is_dir():
            print(f"error: root {root} is not a directory", file=sys.stderr)
            return 2
        return _serve(root, args.host, args.port) if args.command == "serve" else _start(root, args)
    return _status(root) if args.command == "status" else _stop(root)


if __name__ == "__main__":
    sys.exit(main())
