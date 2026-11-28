"""Offline tests for scripts/live_real_m1_run.py (M1-01); no network, no real .env."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "fake-key-not-real-1234"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "live_real_m1_run", ROOT / "scripts" / "live_real_m1_run.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_queries_skips_blanks_comments_and_reads_language(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "q.txt"
    f.write_text("# note\n\nalpha services\nxx\tbeta services\n", encoding="utf-8")
    assert mod.parse_queries(f) == [(None, "alpha services"), ("xx", "beta services")]


def test_child_env_passes_service_urls_and_scrubs_them(tmp_path: Path) -> None:
    mod = _load()
    values = {
        "DM_LLM_GATEWAY_API_KEY": FAKE_KEY,
        "SEARXNG_URL": "http://search-host.invalid:8888",
        "FIRECRAWL_URL": "http://crawl-host.invalid:3002",
        "RAW_STORE_DIR": "/elsewhere/raw",
        "OTHER": "ignored",
    }
    env, secrets = mod.child_env(values, tmp_path / "db.sqlite", tmp_path / "raw")
    assert env["SEARXNG_URL"] == values["SEARXNG_URL"]
    assert env["FIRECRAWL_URL"] == values["FIRECRAWL_URL"]
    assert env["RAW_STORE_DIR"] == str(tmp_path / "raw")  # never the env-file store
    assert env["DM_DATABASE_URL"].startswith("sqlite:///")
    assert "OTHER" not in env or env.get("OTHER") != "ignored"
    for v in (FAKE_KEY, "search-host.invalid", "crawl-host.invalid"):
        assert any(v in s for s in secrets)


def test_plan_is_serial_and_ordered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    env_file = tmp_path / "e.env"
    env_file.write_text(f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    queries = tmp_path / "q.txt"
    queries.write_text("alpha\nyy\tbeta\n", encoding="utf-8")
    seen: list[list[str]] = []

    def fake_run(self: object, args: list[str], *, label: str | None = None) -> str:
        seen.append(args)
        return ""

    monkeypatch.setattr(mod.Runner, "run", fake_run)
    out = tmp_path / "run"
    rc = mod.main(
        [
            "--scope", str(ROOT / "scopes" / "_example"),
            "--queries", str(queries),
            "--per-query", "5",
            "--out", str(out),
            "--env-file", str(env_file),
        ]
    )  # fmt: skip
    names = [a[0] for a in seen]
    assert names == ["web-search", "web-search", "resolve", "crawl", "screen"]
    assert seen[0][seen[0].index("--limit") + 1] == "5" and "--save" in seen[0]
    assert seen[1][seen[1].index("--language") + 1] == "yy"
    assert "--apply" in seen[2]
    assert seen[4][seen[4].index("--jobs") + 1] == "1"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"] == 0 and summary["entities"] == 0
    assert rc == 0
    assert FAKE_KEY not in (out / "summary.json").read_text(encoding="utf-8")


def test_steps_written_as_they_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    env_file = tmp_path / "e.env"
    env_file.write_text(f"DM_LLM_GATEWAY_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    queries = tmp_path / "q.txt"
    queries.write_text("alpha\n", encoding="utf-8")

    def fake_execute(
        self: object, args: list[str], label: str | None
    ) -> tuple[dict[str, object], str, str | None]:
        step = {"command": label or args[0], "exit_code": 0, "elapsed_s": 0.0}
        return step, "", None

    monkeypatch.setattr(mod.Runner, "_execute", fake_execute)
    out = tmp_path / "run"
    args = ["--scope", str(ROOT / "scopes" / "_example"), "--queries", str(queries)]
    assert mod.main([*args, "--out", str(out), "--env-file", str(env_file)]) == 0
    lines = (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["command"] for x in lines] == [
        "web-search q0", "resolve", "crawl", "screen"
    ]  # fmt: skip
