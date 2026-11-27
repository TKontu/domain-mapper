"""LL-01: scripts/live_m1_run.py against a fake OpenAI-compatible server on 127.0.0.1."""

from __future__ import annotations

import json
import os
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
    assert summary["failed_step"] is None and summary["degraded"] is False
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


def test_live_m1_run_dead_gateway_degraded(tmp_path: Path, server: str) -> None:
    _Handler.fail = True
    try:
        out = tmp_path / "run"
        proc = _run(out, _env_file(tmp_path, server))
    finally:
        _Handler.fail = False
    # every screen call fails -> screen exits 1 (tolerated), nothing in scope; later steps run,
    # but the run is flagged degraded and exits 3.
    assert proc.returncode == 3, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["degraded"] is True and summary["failed_step"] is None
    assert summary["steps"][0] == {**summary["steps"][0], "command": "screen", "exit_code": 1}
    assert summary["llm_calls"]["error"] > 0
    assert "Traceback" not in proc.stderr
    _assert_no_secrets(proc, out, server)


def test_live_m1_run_fatal_step_names_step(tmp_path: Path) -> None:
    # a malformed setting exits 2 with the raw value in the child's error line: fatal + redacted
    env_file = tmp_path / "bad.env"
    env_file.write_text(
        f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\nDM_LLM_BUDGET_USD=x{FAKE_KEY}\n", encoding="utf-8"
    )
    out = tmp_path / "run"
    proc = _run(out, env_file)
    assert proc.returncode == 1
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["failed_step"] == "screen"
    assert "Traceback" not in proc.stderr and FAKE_KEY not in proc.stderr + proc.stdout


def _load_script() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location("live_m1_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_failure_tail_redacts_encoded_forms_and_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import urllib.parse

    mod = _load_script()
    host = "gw-fake-ll03.internal"
    url = f"https://{host}:8443/v1"
    key = 'sk-fake/ll03 key+"q&z'
    secrets = mod.secret_values(  # type: ignore[attr-defined]
        {"DM_LLM_GATEWAY_URL": url, "DM_LLM_GATEWAY_API_KEY": key, "DM_FOO": "true"}
    )
    forms = [
        urllib.parse.quote(key),
        urllib.parse.quote(key, safe=""),
        urllib.parse.quote_plus(key),
        json.dumps(key)[1:-1],
        host,
        f"{host}:8443",
    ]
    code = "import sys; print(' '.join(sys.argv[1:]), 'true'); sys.exit(2)"
    monkeypatch.setattr(mod, "LANDSCAPE", [sys.executable, "-c", code])
    runner = mod.Runner(dict(os.environ), tmp_path, secrets)  # type: ignore[attr-defined]
    with pytest.raises(mod.StepFailed):  # type: ignore[attr-defined]
        runner.run(["presence-extract", *forms])
    err = capsys.readouterr().err
    assert "[redacted]" in err and "true" in err
    for form in forms:
        assert form not in err, form


def test_benign_values_not_scrubbed() -> None:
    mod = _load_script()
    values = {"DM_FOO": "true", "DM_LLM_RETRIES": "1", "DM_BAR_KEY": "abc", "DM_X_TOKEN": "tok-9"}
    secrets = mod.secret_values(values)  # type: ignore[attr-defined]
    assert "true" not in secrets and "1" not in secrets and "abc" not in secrets
    assert "tok-9" in secrets
    assert mod.redact("enabled=true retries=1", secrets) == "enabled=true retries=1"  # type: ignore[attr-defined]


def test_scrub_skips_sqlite_journal_files(tmp_path: Path) -> None:
    mod = _load_script()
    secret = "fake-ll03-model-zz"
    names = ["db.sqlite-journal", "db.sqlite-wal", "db.sqlite-shm", "x-journal", "y-wal", "z-shm"]
    for name in names:
        (tmp_path / name).write_text(secret, encoding="utf-8")
    (tmp_path / "export.jsonl").write_text(secret, encoding="utf-8")
    assert mod.scrub_artifacts(tmp_path, [secret]) == 1  # type: ignore[attr-defined]
    for name in names:
        assert (tmp_path / name).read_text(encoding="utf-8") == secret
    assert secret not in (tmp_path / "export.jsonl").read_text(encoding="utf-8")


def test_no_screen_calls_with_entities_is_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # every child command "succeeds" without any llm_call: entities present, zero screen rows
    mod = _load_script()
    code = (
        "import sys, pathlib; a = sys.argv[1:]\n"
        "if '--out' in a: pathlib.Path(a[a.index('--out') + 1]).write_text('')"
    )
    monkeypatch.setattr(mod, "LANDSCAPE", [sys.executable, "-c", code])
    env_file = tmp_path / "fake.env"
    env_file.write_text(f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    out = tmp_path / "run"
    argv = ["--out", str(out), "--env-file", str(env_file), "--entities", "2"]
    assert mod.main(argv) == 3  # type: ignore[attr-defined]
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["degraded"] is True and summary["failed_step"] is None
    assert summary["entities"] == 2 and summary["llm_calls"] == {"ok": 0, "error": 0}


def test_no_entities_is_not_degraded(tmp_path: Path) -> None:
    mod = _load_script()
    db_url = f"sqlite:///{tmp_path / 'empty.sqlite'}"
    mod.seed(db_url, "scope-x", 0)  # type: ignore[attr-defined]
    assert mod.db_counts(db_url, "scope-x")["degraded"] is False  # type: ignore[attr-defined]


class _FakeRun:
    """Thread-safe stand-in for ``subprocess.run``: tracks concurrent assess calls."""

    def __init__(self, delay: float = 0.0, fatal: str | None = None) -> None:
        self.delay, self.fatal = delay, fatal
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        import time

        args = [str(a) for a in cmd]
        with self.lock:
            self.calls.append(args)
        if "--out" in args:
            Path(args[args.index("--out") + 1]).write_text("", encoding="utf-8")
        code = 0
        if "assess" in args:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(self.delay)
            with self.lock:
                self.active -= 1
            label = f"assess {args[args.index('--entity') + 1]} {args[-1]}"
            code = 2 if label == self.fatal else 1 if label.endswith("c2") else 0
        return subprocess.CompletedProcess(args, code, "", "")


PAIRS = [(e, c) for e in (1, 2, 3) for c in ("c1", "c2")]
LABELS = [f"assess {e} {c}" for e, c in PAIRS]


def _main_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: _FakeRun, *extra: str
) -> tuple[int, dict[str, object]]:
    mod = _load_script()
    monkeypatch.setattr(mod.subprocess, "run", fake)  # type: ignore[attr-defined]
    monkeypatch.setattr(mod, "assess_pairs", lambda *a: list(PAIRS))
    env_file = tmp_path / "fake.env"
    env_file.write_text(f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    out = tmp_path / "run"
    argv = ["--out", str(out), "--env-file", str(env_file), "--entities", "2", *extra]
    code = mod.main(argv)  # type: ignore[attr-defined]
    return code, json.loads((out / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("jobs", ["0", "9", "x"])
def test_jobs_out_of_range_exits_2(tmp_path: Path, jobs: str) -> None:
    mod = _load_script()
    env_file = tmp_path / "fake.env"
    env_file.write_text("DM_X=1\n", encoding="utf-8")
    argv = ["--out", str(tmp_path / "o"), "--env-file", str(env_file), "--jobs", jobs]
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)  # type: ignore[attr-defined]
    assert exc.value.code == 2


def test_jobs_parallel_bounded_and_pair_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun(delay=0.2)
    code, summary = _main_jobs(tmp_path, monkeypatch, fake, "--jobs", "3")
    assert code == 3  # no screen calls in the DB -> degraded, as in the serial stub test
    assert 1 < fake.peak <= 3
    names = [s["command"] for s in summary["steps"]]  # type: ignore[index]
    assert names[1:7] == LABELS and names[0] == "screen" and "mosaic all" in names
    assert summary["jobs"] == 3 and summary["failed_step"] is None


def test_jobs_default_serial_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRun()
    _, summary = _main_jobs(tmp_path, monkeypatch, fake)
    assert fake.peak == 1 and summary["jobs"] == 1
    names = [s["command"] for s in summary["steps"]]  # type: ignore[index]
    assert names[:8] == ["screen", *LABELS, "presence-extract"]
    codes = [s["exit_code"] for s in summary["steps"]][1:7]  # type: ignore[index]
    assert codes == [0, 1] * 3
    assessed = [c for c in fake.calls if "assess" in c]
    assert [f"assess {c[c.index('--entity') + 1]} {c[-1]}" for c in assessed] == LABELS


@pytest.mark.parametrize("value", ["0", "8193", "x"])
def test_max_tokens_out_of_range_exits_2(tmp_path: Path, value: str) -> None:
    mod = _load_script()
    env_file = tmp_path / "fake.env"
    env_file.write_text("DM_X=1\n", encoding="utf-8")
    argv = ["--out", str(tmp_path / "o"), "--env-file", str(env_file)]
    with pytest.raises(SystemExit) as exc:
        mod.main([*argv, "--max-tokens", value])  # type: ignore[attr-defined]
    assert exc.value.code == 2


def test_max_tokens_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\nDM_LOCAL_MAX_TOKENS=4096\n"
    fake, summary = _main_env(tmp_path, monkeypatch, text, "--max-tokens", "512")
    assert fake.envs and all(e["DM_LOCAL_MAX_TOKENS"] == "512" for e in fake.envs)
    assert summary["max_tokens"] == 512


@pytest.mark.parametrize("line", ["", "DM_LOCAL_MAX_TOKENS=4096\n"])
def test_max_tokens_omitted_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str
) -> None:
    text = f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n{line}"
    fake, summary = _main_env(tmp_path, monkeypatch, text)
    expected = "4096" if line else None
    assert fake.envs and all(e.get("DM_LOCAL_MAX_TOKENS") == expected for e in fake.envs)
    assert summary["max_tokens"] is None


def test_max_tokens_and_verifier_max_tokens_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n"
    extra = ("--max-tokens", "512", "--verifier-max-tokens", "256")
    fake, summary = _main_env(tmp_path, monkeypatch, text, *extra)
    assert fake.envs
    assert all(e["DM_LOCAL_MAX_TOKENS"] == "512" for e in fake.envs)
    assert all(e["DM_VERIFIER_MAX_TOKENS"] == "256" for e in fake.envs)
    assert summary["max_tokens"] == 512 and summary["verifier_max_tokens"] == 256


@pytest.mark.parametrize("jobs", ["1", "3"])
def test_jobs_fatal_assess_stops_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: str
) -> None:
    fake = _FakeRun(delay=0.05, fatal="assess 2 c1")
    code, summary = _main_jobs(tmp_path, monkeypatch, fake, "--jobs", jobs)
    assert code == 1 and summary["failed_step"] == "assess 2 c1"
    names = [s["command"] for s in summary["steps"]]  # type: ignore[index]
    assert "presence-extract" not in names and not any(n.startswith("mosaic") for n in names)
    assessed = [n for n in names if n.startswith("assess")]
    assert assessed == [lab for lab in LABELS if lab in assessed]  # pair order
    assert not any("presence-extract" in c for c in fake.calls)


class _EnvRun(_FakeRun):
    """Records the child environment of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.envs: list[dict[str, str]] = []

    def __call__(self, cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        self.envs.append(dict(kw["env"]))  # type: ignore[call-overload]
        return super().__call__(cmd)


def _main_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_text: str, *extra: str
) -> tuple[_EnvRun, dict[str, object]]:
    mod = _load_script()
    fake = _EnvRun()
    monkeypatch.setattr(mod.subprocess, "run", fake)  # type: ignore[attr-defined]
    monkeypatch.setattr(mod, "assess_pairs", lambda *a: list(PAIRS))
    monkeypatch.delenv("DM_VERIFIER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("DM_LOCAL_MAX_TOKENS", raising=False)
    env_file = tmp_path / "fake.env"
    env_file.write_text(env_text, encoding="utf-8")
    out = tmp_path / "run"
    argv = ["--out", str(out), "--env-file", str(env_file), "--entities", "2", *extra]
    mod.main(argv)  # type: ignore[attr-defined]
    return fake, json.loads((out / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("value", ["0", "8193", "x"])
def test_verifier_max_tokens_out_of_range_exits_2(tmp_path: Path, value: str) -> None:
    mod = _load_script()
    env_file = tmp_path / "fake.env"
    env_file.write_text("DM_X=1\n", encoding="utf-8")
    argv = ["--out", str(tmp_path / "o"), "--env-file", str(env_file)]
    with pytest.raises(SystemExit) as exc:
        mod.main([*argv, "--verifier-max-tokens", value])  # type: ignore[attr-defined]
    assert exc.value.code == 2


def test_verifier_max_tokens_overrides_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\nDM_VERIFIER_MAX_TOKENS=999\n"
    fake, summary = _main_env(tmp_path, monkeypatch, text, "--verifier-max-tokens", "256")
    assert fake.envs and all(e["DM_VERIFIER_MAX_TOKENS"] == "256" for e in fake.envs)
    assert summary["verifier_max_tokens"] == 256


@pytest.mark.parametrize("line", ["", "DM_VERIFIER_MAX_TOKENS=999\n"])
def test_verifier_max_tokens_omitted_passes_env_file_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str
) -> None:
    text = f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n{line}"
    fake, summary = _main_env(tmp_path, monkeypatch, text)
    expected = "999" if line else None
    assert fake.envs and all(e.get("DM_VERIFIER_MAX_TOKENS") == expected for e in fake.envs)
    assert summary["verifier_max_tokens"] is None


@pytest.mark.parametrize("jobs", ["1", "3"])
def test_jobs_passed_to_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: str) -> None:
    fake = _FakeRun()
    _main_jobs(tmp_path, monkeypatch, fake, "--jobs", jobs)
    screen = [c for c in fake.calls if "screen" in c]
    assert len(screen) == 1
    assert screen[0][screen[0].index("--jobs") + 1] == jobs
