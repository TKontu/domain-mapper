"""Offline test for scripts/llm_smoke.py: fake transport, fake key never leaks."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
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


@pytest.mark.parametrize(
    "flags", [["--parallel", "0"], ["--parallel", "9"], ["--rounds", "0"], ["--rounds", "6"]]
)
def test_parallel_rounds_bounds_exit_2(env_file, flags):
    with pytest.raises(SystemExit) as exc:
        _load().main(["--env-file", str(env_file), *flags], transport=_ok)
    assert exc.value.code == 2


class _Slow:
    def __init__(self, delay: float, fail_on: int | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.count = 0

    def __call__(
        self, url: str, *, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> HttpResponse:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.count += 1
            n = self.count
        try:
            time.sleep(self.delay)
        finally:
            with self.lock:
                self.active -= 1
        if n == self.fail_on:
            return HttpResponse(status=503)
        return _ok(url, headers=headers, body=body, timeout=timeout)


def test_parallel_runs_concurrently(env_file, tmp_path, capsys):
    out = tmp_path / "r.json"
    fake = _Slow(0.3)
    rc = _load().main(
        ["--env-file", str(env_file), "--out", str(out), "--parallel", "3", "--rounds", "2"],
        transport=fake,
    )
    assert rc == 0
    _check_no_leak(capsys, out)
    assert fake.peak == 3
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["parallel"] == 3 and summary["rounds"] == 2 and summary["status"] == "ok"
    assert len(summary["round_results"]) == 2
    for rnd in summary["round_results"]:
        assert len(rnd["calls"]) == 3
        lats = [c["latency_s"] for c in rnd["calls"]]
        assert rnd["wall_s"] < sum(lats)
        assert rnd["max_latency_s"] == max(lats) and rnd["min_latency_s"] == min(lats)
        for call in rnd["calls"]:
            assert set(call) >= {"latency_s", "status", "error_class", "tokens_in", "tokens_out"}


def test_parallel_one_failure_exit_1(env_file, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = _load().main(
        ["--env-file", str(env_file), "--out", str(out), "--parallel", "3"],
        transport=_Slow(0.05, fail_on=2),
    )
    assert rc == 1
    _check_no_leak(capsys, out)
    calls = json.loads(out.read_text(encoding="utf-8"))["round_results"][0]["calls"]
    assert sorted(c["status"] for c in calls) == ["error", "ok", "ok"]
    assert [c["error_class"] for c in calls if c["status"] == "error"] == ["LlmError"]
