"""Offline test for scripts/llm_smoke.py: fake transport, fake key never leaks."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from adapters.llm import HttpResponse

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-fake-smoke-key-7f3a9c"
FAKE_URL = "http://gateway.invalid:9999/v1"


def _load():
    spec = importlib.util.spec_from_file_location("llm_smoke", ROOT / "scripts" / "llm_smoke.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in ("DM_LLM_GATEWAY_URL", "DM_LLM_GATEWAY_API_KEY", "DM_LOCAL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "smoke.env"
    path.write_text(
        f"DM_LLM_GATEWAY_URL={FAKE_URL}\nDM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n"
        "DM_LOCAL_MODEL=fake-model\n",
        encoding="utf-8",
    )
    return path


def _ok(url: str, *, headers: Mapping[str, str], body: bytes, timeout: float) -> HttpResponse:
    assert headers.get("Authorization") == f"Bearer {FAKE_KEY}"
    assert timeout == 600.0
    payload = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    }
    return HttpResponse(status=200, body=json.dumps(payload).encode())


def _fail(url: str, *, headers: Mapping[str, str], body: bytes, timeout: float) -> HttpResponse:
    return HttpResponse(status=503)


def _check_no_leak(capsys, out: Path) -> str:
    captured = capsys.readouterr()
    blob = captured.out + captured.err + out.read_text(encoding="utf-8")
    assert FAKE_KEY not in blob
    assert "gateway.invalid" not in blob
    return captured.out


def test_smoke_ok_hides_secrets(env_file, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = _load().main(
        ["--env-file", str(env_file), "--db", str(tmp_path / "d.sqlite"), "--out", str(out)],
        transport=_ok,
    )
    assert rc == 0
    _check_no_leak(capsys, out)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "ok" and result["model"] == "fake-model"
    assert (result["tokens_in"], result["tokens_out"]) == (7, 1)
    assert set(result) == {"model", "latency_s", "status", "error_class", "tokens_in", "tokens_out"}


def test_smoke_error_records_class(env_file, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = _load().main(["--env-file", str(env_file), "--out", str(out)], transport=_fail)
    assert rc == 1
    _check_no_leak(capsys, out)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "error" and result["error_class"] == "LlmError"
