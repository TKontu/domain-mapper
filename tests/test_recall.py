"""Tests for the LLM-recall Discover channel (DS-03); offline, fake LLM, SQLite temp dir."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from core.recall import RECALL_SCHEMA, RecallError, build_prompt, recall_candidates
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.llm import LlmResult, LlmSchemaError
from app import cli
from core.candidate_store import Candidate
from core.runs import Run, create_all
from core.scope_pack import load_scope_pack

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


class FakeLLM:
    model = "fake-recall"

    def __init__(self, parsed: object = None, *, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        self.calls.append(
            {"prompt": prompt, "purpose": purpose, "run_id": run_id, "schema": json_schema}
        )
        if self.error is not None:
            raise self.error
        return LlmResult(
            text="{}",
            parsed=self.parsed,  # type: ignore[arg-type]
            tokens_in=1,
            tokens_out=1,
            cost=0.0,
            cache_hit=False,
        )


def proposals(*pairs: tuple[str, str]) -> dict[str, object]:
    return {"candidates": [{"name": n, "url": u} for n, u in pairs]}


@pytest.fixture
def pack():
    return load_scope_pack(Path(EXAMPLE_SCOPE))


# --- core -------------------------------------------------------------------


def test_valid_output_becomes_llm_recall_candidates(pack) -> None:
    llm = FakeLLM(
        proposals(
            ("Acme", "https://acme.example/"),
            ("Ftp", "ftp://bad.example/"),
            ("Dup", "https://acme.example/"),
            ("Beta", "http://beta.example"),
            ("Blank", "  "),
            ("Gamma", "https://gamma.example"),
        )
    )
    rows = recall_candidates(llm, pack, segment="agency", country="XA", limit=2, run_id=7)
    assert [(r.name, r.url) for r in rows] == [
        ("Acme", "https://acme.example/"),
        ("Beta", "http://beta.example"),
    ]
    for r in rows:
        assert r.channel == "llm_recall"
        assert r.segment_hint == "agency"
        assert r.country_hint == "XA"
        assert r.query and "agency" in r.query and "XA" in r.query
        assert r.found_at.tzinfo is not None
        assert r.found_at.utcoffset() == UTC.utcoffset(None)
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["purpose"] == "recall"
    assert call["run_id"] == 7
    assert call["schema"] == RECALL_SCHEMA


@pytest.mark.parametrize(
    ("parsed", "field"),
    [
        (None, "response"),
        ({}, "candidates"),
        ({"candidates": "x"}, "candidates"),
        ({"candidates": ["x"]}, "candidates[0]"),
        ({"candidates": [{"name": "A"}]}, "candidates[0].url"),
        ({"candidates": [{"name": 3, "url": "https://a.example"}]}, "candidates[0].name"),
    ],
)
def test_schema_violation_raises_recall_error_naming_field(pack, parsed, field) -> None:
    llm = FakeLLM(parsed)
    secret = "SECRET-OUTPUT"
    if isinstance(parsed, dict) and parsed.get("candidates") == "x":
        parsed["candidates"] = secret
    with pytest.raises(RecallError) as info:
        recall_candidates(llm, pack, segment="agency", country="XA", limit=5, run_id=1)
    assert field in str(info.value)
    assert secret not in str(info.value)


def test_prompt_marks_output_unverified_and_forbids_invented_urls(pack) -> None:
    prompt = build_prompt(pack, segment="agency", country="XA", limit=5)
    lowered = prompt.lower()
    assert "unverified" in lowered
    assert "do not invent" in lowered and "url" in lowered
    assert "agency" in prompt and "XA" in prompt and "5" in prompt


@pytest.mark.parametrize(
    ("segment", "country", "limit"),
    [
        ("nope", "XA", 5),
        ("agency", "ZZ", 5),
        (" ", "XA", 5),
        ("agency", " ", 5),
        ("agency", "XA", 0),
    ],
)
def test_core_rejects_bad_inputs_before_llm(pack, segment, country, limit) -> None:
    llm = FakeLLM(proposals())
    with pytest.raises(ValueError):
        recall_candidates(llm, pack, segment=segment, country=country, limit=limit, run_id=1)
    assert llm.calls == []


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def db_url(tmp_path, monkeypatch) -> str:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no stray .env
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    return url


@pytest.fixture
def engine(db_url: str):
    engine = sa.create_engine(db_url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def use_llm(monkeypatch, llm: FakeLLM) -> None:
    monkeypatch.setattr(cli, "_make_recall_llm", lambda settings, db: llm)


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["recall", "--scope", EXAMPLE_SCOPE, *args])


def candidates(engine) -> list[Candidate]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(Candidate).order_by(Candidate.id)))


def runs(engine) -> list[tuple[int, str, str]]:
    with Session(engine) as session:
        return [(r.id, r.stage, r.status) for r in session.scalars(sa.select(Run))]


def test_cli_prints_and_saves(engine, monkeypatch) -> None:
    llm = FakeLLM(proposals(("Acme", "https://acme.example/"), ("Beta", "https://beta.example")))
    use_llm(monkeypatch, llm)
    result = invoke("--segment", "agency", "--country", "XA", "--save")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "Acme\thttps://acme.example/",
        "Beta\thttps://beta.example",
        "saved 2 candidates",
    ]
    rows = candidates(engine)
    assert [(r.name, r.channel, r.segment_hint, r.country_hint) for r in rows] == [
        ("Acme", "llm_recall", "agency", "XA"),
        ("Beta", "llm_recall", "agency", "XA"),
    ]
    assert all(r.found_at.tzinfo is not None for r in rows)
    (run,) = runs(engine)
    assert run[1:] == ("recall", "finished")
    assert llm.calls[0]["run_id"] == run[0]


def test_cli_without_save_writes_nothing(engine, monkeypatch) -> None:
    use_llm(monkeypatch, FakeLLM(proposals(("Acme", "https://acme.example/"))))
    result = invoke("--segment", "agency", "--country", "XA")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["Acme\thttps://acme.example/"]
    assert candidates(engine) == []


def test_cli_limit_caps_output(engine, monkeypatch) -> None:
    pairs = [(f"N{i}", f"https://n{i}.example") for i in range(5)]
    use_llm(monkeypatch, FakeLLM(proposals(*pairs)))
    result = invoke("--segment", "agency", "--country", "XA", "--limit", "2")
    assert result.exit_code == 0, result.output
    assert len(result.stdout.splitlines()) == 2


@pytest.mark.parametrize(
    "error", [None, LlmSchemaError("model output is not valid JSON")], ids=["schema", "json"]
)
def test_cli_invalid_output_exits_1_and_saves_nothing(engine, monkeypatch, error) -> None:
    llm = FakeLLM({"candidates": [{"name": "A"}]}, error=error)
    use_llm(monkeypatch, llm)
    result = invoke("--segment", "agency", "--country", "XA", "--save")
    assert result.exit_code == 1, result.output
    assert candidates(engine) == []
    (run,) = runs(engine)
    assert run[1:] == ("recall", "failed")


@pytest.mark.parametrize(
    "args",
    [
        ["--segment", "nope", "--country", "XA"],
        ["--segment", "agency", "--country", "ZZ"],
        ["--segment", " ", "--country", "XA"],
        ["--segment", "agency", "--country", " "],
        ["--segment", "agency", "--country", "XA", "--limit", "0"],
        ["--segment", "agency", "--country", "XA", "--limit", "101"],
    ],
)
def test_cli_bad_inputs_exit_2_before_llm(engine, monkeypatch, args) -> None:
    llm = FakeLLM(proposals())
    use_llm(monkeypatch, llm)
    result = invoke(*args)
    assert result.exit_code == 2, result.output
    assert llm.calls == []
    assert runs(engine) == []


# --- name sanitising ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Acme\tCorp",
        "Acme\nCorp",
        "Acme\r\nCorp",
        "Acme\x00Corp",
        "Acme Corp",
        "  Acme \t​  Corp\x7f ",
    ],
)
def test_recalled_name_is_single_line_single_space(pack, raw) -> None:
    llm = FakeLLM(proposals((raw, "https://acme.example/")))
    (row,) = recall_candidates(llm, pack, segment="agency", country="XA", limit=5, run_id=1)
    assert row.name == "Acme Corp"


def test_recalled_name_capped_at_200(pack) -> None:
    llm = FakeLLM(proposals(("x" * 500, "https://acme.example/")))
    (row,) = recall_candidates(llm, pack, segment="agency", country="XA", limit=5, run_id=1)
    assert row.name == "x" * 200


def test_control_only_name_falls_back_to_url(pack) -> None:
    llm = FakeLLM(proposals((" \t\n\x00  ", "https://acme.example/")))
    (row,) = recall_candidates(llm, pack, segment="agency", country="XA", limit=5, run_id=1)
    assert row.name == "https://acme.example/"


def test_cli_prints_and_saves_sanitised_name(engine, monkeypatch) -> None:
    use_llm(monkeypatch, FakeLLM(proposals(("Acme\t\r\nCorp\x00", "https://acme.example/"))))
    result = invoke("--segment", "agency", "--country", "XA", "--save")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines == ["Acme Corp\thttps://acme.example/", "saved 1 candidates"]
    assert lines[0].count("\t") == 1
    assert [r.name for r in candidates(engine)] == ["Acme Corp"]
