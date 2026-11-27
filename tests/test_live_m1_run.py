"""LL-01: scripts/live_m1_run.py against a fake OpenAI-compatible server on 127.0.0.1."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_m1_run.py"
FAKE_KEY = "sk-fake-ll01-key-7f3a9c"
FAKE_MODEL = "fake-ll01-model-q8z"
FAKE_VERIFIER = "fake-ll01-verifier-k2w"


def _answer(payload: dict[str, object]) -> dict[str, object]:
    fmt = payload.get("response_format")
    schema = fmt["json_schema"]["schema"] if isinstance(fmt, dict) else {}  # type: ignore[index]
    props = schema.get("properties", {})
    prompt = payload["messages"][0]["content"]  # type: ignore[index]
    if "in_scope" in props:
        answer: dict[str, object] = {
            "in_scope": "furniture" not in prompt,
            "segments": [] if "furniture" in prompt else ["agency"],
            "confidence": 0.9,
            "reason": "fake screen",
        }
    elif "agree" in props:
        answer = {"agree": True, "reason": "fake verify"}
    else:
        m = re.search(r"\[chunk (\d+)\]\n([^\n]+)", prompt)
        assert m is not None
        quote = m.group(2)[:30]
        answer = {
            "status": "standard",
            "quotes": [{"chunk_id": int(m.group(1)), "quote": quote}],
            "rationale": "fake extract",
        }
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(answer)}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


class _Handler(BaseHTTPRequestHandler):
    fail = False

    def do_POST(self) -> None:  # noqa: N802
        assert self.headers.get("Authorization") == f"Bearer {FAKE_KEY}"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.fail:
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps(_answer(body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _env_file(tmp_path: Path, url: str) -> Path:
    path = tmp_path / "fake.env"
    path.write_text(
        f"DM_LLM_GATEWAY_URL={url}\nDM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n"
        f"DM_LOCAL_MODEL={FAKE_MODEL}\nDM_VERIFIER_MODEL={FAKE_VERIFIER}\n",
        encoding="utf-8",
    )
    return path


def _run(out: Path, env_file: Path) -> subprocess.CompletedProcess[str]:
    import os

    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--env-file", str(env_file)]
        + ["--entities", "3", "--timeout", "30"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=300,
    )


def _assert_no_secrets(proc: subprocess.CompletedProcess[str], out: Path, url: str) -> None:
    secrets = [FAKE_KEY, FAKE_MODEL, FAKE_VERIFIER, url, url.split("//")[1]]
    for value in secrets:
        assert value not in proc.stdout
        assert value not in proc.stderr
    for path in out.rglob("*"):
        if path.is_file() and path.suffix != ".sqlite":
            text = path.read_bytes().decode("utf-8", errors="replace")
            for value in secrets:
                assert value not in text, path


def test_live_m1_run_fake_gateway(tmp_path: Path, server: str) -> None:
    out = tmp_path / "run"
    proc = _run(out, _env_file(tmp_path, server))
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary == json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["entities"] == 3
    assert summary["failed_step"] is None
    assert all(step["exit_code"] == 0 for step in summary["steps"])
    names = [step["command"] for step in summary["steps"]]
    assert names[0] == "screen" and "presence-extract" in names and "mosaic all" in names
    assert any(n.startswith("assess ") for n in names)
    assert summary["screened"]["in_scope"] == 2 and summary["screened"]["out_of_scope"] == 1
    assert summary["assessments"]["by_route"].get("accept", 0) > 0
    assert summary["llm_calls"]["ok"] > 0 and summary["llm_calls"]["error"] == 0
    assert summary["tokens"]["in"] > 0
    assert summary["mosaic"]["all"]["rows"] > 0
    _assert_no_secrets(proc, out, server)


def test_live_m1_run_step_failure(tmp_path: Path, server: str) -> None:
    _Handler.fail = True
    try:
        out = tmp_path / "run"
        proc = _run(out, _env_file(tmp_path, server))
    finally:
        _Handler.fail = False
    # every screen call fails -> screen exits 1 (tolerated), nothing in scope; later steps run.
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["steps"][0] == {**summary["steps"][0], "command": "screen", "exit_code": 1}
    assert summary["llm_calls"]["error"] > 0
    assert "Traceback" not in proc.stderr
    _assert_no_secrets(proc, out, server)


def test_live_m1_run_fatal_step_names_step(tmp_path: Path) -> None:
    # a malformed setting exits 2 with the raw value in the child's error line: fatal + redacted
    env_file = tmp_path / "bad.env"
    env_file.write_text(f"DM_LLM_BUDGET_USD=x{FAKE_KEY}\n", encoding="utf-8")
    out = tmp_path / "run"
    proc = _run(out, env_file)
    assert proc.returncode == 1
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["failed_step"] == "screen"
    assert "Traceback" not in proc.stderr and FAKE_KEY not in proc.stderr + proc.stdout
