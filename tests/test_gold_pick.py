"""Tests for ``landscape gold-pick`` (GS-01); temp SQLite, neutral ``scopes/_example`` pack."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.llm import HttpResponse
from app import cli
from core.assess import Extraction, Quote
from core.assessment_store import record_assessment
from core.chunk_store import ChunkRow
from core.entity_store import Entity
from core.gold_pick import is_readable, trim_to_passage
from core.llm_calls import LlmCall
from core.runs import Run, create_all
from core.snapshot_store import SnapshotRow
from core.verify import Route

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
CRITERIA = ("iso17100", "public_terms")
READABLE = (
    "The provider publishes its general terms of service on the website. "
    "They cover liability, confidentiality and payment, and are dated."
)
FETCHED = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine(tmp_path, monkeypatch) -> Engine:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", "sqlite:///unused-default.sqlite")
    engine = create_engine(url)
    create_all(engine)
    engine.url_text = url  # type: ignore[attr-defined]
    try:
        yield engine
    finally:
        engine.dispose()


def _run(engine) -> int:
    with Session(engine) as session, session.begin():
        row = Run(stage="assess", status="finished", started_at=datetime.now(UTC))
        session.add(row)
        session.flush()
        return row.id


def seed(engine, domain: str, statuses: dict[str, str], text: str = READABLE) -> int:
    with Session(engine) as session, session.begin():
        ent = Entity(scope_id="example", name=domain, primary_domain=domain, status="in_scope")
        session.add(ent)
        session.flush()
        snap = SnapshotRow(
            entity_id=ent.id,
            url=f"https://{domain}/",
            final_url=f"https://{domain}/terms",
            content_type="text/html",
            sha256="0" * 64,
            fetched_at=FETCHED,
            source_kind="homepage",
        )
        session.add(snap)
        session.flush()
        chunk = ChunkRow(
            snapshot_id=snap.id,
            seq=0,
            text=text,
            lang="en",
            page=None,
            heading_path=[],
            char_start=0,
            char_end=len(text),
        )
        session.add(chunk)
        session.flush()
        entity_id, chunk_id = ent.id, chunk.id
    run_id = _run(engine)
    for criterion, status in statuses.items():
        record_assessment(
            engine,
            entity_id=entity_id,
            criterion_id=criterion,
            extraction=Extraction(
                status=status,
                quotes=(
                    Quote(
                        chunk_id=chunk_id,
                        text=text[:20],
                        grounded=True,
                        method="exact",
                        span=(0, 20),
                    ),
                ),
                rationale="x",
                checks_passed=True,
                failures=(),
            ),
            route=Route("accept", None, "r"),
            rubric_version="1",
            extractor_model="e",
            verifier_model="v",
            run_id=run_id,
        )
    return entity_id


def pick(engine, out: Path, *extra: str):
    return CliRunner().invoke(
        cli.app,
        [
            "gold-pick",
            "--scope",
            EXAMPLE_SCOPE,
            "--db",
            engine.url_text,
            "--out",
            str(out),
            *extra,
        ],
    )


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def standard_db(engine) -> None:
    seed(engine, "a.example", {"iso17100": "strong", "public_terms": "strong"})
    seed(engine, "b.example", {"iso17100": "weak", "public_terms": "strong"})
    seed(engine, "c.example", {"iso17100": "red_flag", "public_terms": "standard"})
    seed(engine, "d.example", {"iso17100": "weak", "public_terms": "standard"})


def test_cells_are_blind_and_key_is_separate(engine, tmp_path):
    standard_db(engine)
    result = pick(engine, tmp_path / "out")
    assert result.exit_code == 0, result.output
    cells = lines(tmp_path / "out" / "cells.jsonl")
    key = lines(tmp_path / "out" / "cells.key.jsonl")
    assert cells and len(cells) == len(key)
    forbidden = {"status", "confidence", "score", "assessment_id", "rationale"}
    for cell in cells:
        assert set(cell) == {
            "cell_id",
            "primary_domain",
            "criterion_id",
            "criterion",
            "evidence",
        }
        assert not forbidden & set(json.dumps(cell).replace('"', " ").split())
    assert {k["cell_id"] for k in key} == {c["cell_id"] for c in cells}
    for k in key:
        assert set(k) == {"cell_id", "status", "confidence", "assessment_id"}


def test_deterministic_for_same_db(engine, tmp_path):
    standard_db(engine)
    assert pick(engine, tmp_path / "o1").exit_code == 0
    assert pick(engine, tmp_path / "o2").exit_code == 0
    for name in ("cells.jsonl", "cells.key.jsonl"):
        assert (tmp_path / "o1" / name).read_bytes() == (tmp_path / "o2" / name).read_bytes()


def test_status_spread(engine, tmp_path):
    standard_db(engine)
    result = pick(engine, tmp_path / "out", "--entities", "4", "--per-entity", "1")
    assert result.exit_code == 0, result.output
    key = lines(tmp_path / "out" / "cells.key.jsonl")
    assert len(key) == 4
    assert {k["status"] for k in key} == {"strong", "standard", "weak", "red_flag"}


def test_evidence_frozen_with_url_and_fetched_at(engine, tmp_path):
    standard_db(engine)
    assert pick(engine, tmp_path / "out").exit_code == 0
    for cell in lines(tmp_path / "out" / "cells.jsonl"):
        assert cell["criterion"].startswith(("Is the provider", "Are general terms"))
        (ev,) = cell["evidence"]
        assert ev == {
            "chunk_text": READABLE,
            "url": f"https://{cell['primary_domain']}/terms",
            "fetched_at": "2026-05-01T12:00:00+00:00",
        }


def test_unreadable_evidence_is_skipped(engine, tmp_path):
    seed(engine, "junk.example", {"iso17100": "weak"}, text="\x00\x01\x02 %%%% ####" * 5)
    seed(engine, "a.example", {"iso17100": "strong"})
    assert pick(engine, tmp_path / "out").exit_code == 0
    assert [c["primary_domain"] for c in lines(tmp_path / "out" / "cells.jsonl")] == ["a.example"]


def test_readability_and_trim_helpers():
    assert is_readable(READABLE)
    assert not is_readable("short")
    assert not is_readable("�" * 80)
    assert trim_to_passage(READABLE, "They cover liability") == "They cover liability"
    assert trim_to_passage(READABLE, "not in the text") == READABLE
    assert trim_to_passage(READABLE, "") == READABLE


# --- --clean ------------------------------------------------------------------


class FakeTransport:
    """Anthropic-shaped responses; rejects cells whose prompt mentions ``reject_domain``."""

    def __init__(self, reject: str | None = None, passage: str = "") -> None:
        self.reject = reject
        self.passage = passage
        self.calls = 0

    def __call__(self, url, *, headers, body, timeout):
        self.calls += 1
        payload = json.loads(body)
        prompt = payload["messages"][0]["content"]
        assert "temperature" not in payload and "top_p" not in payload
        accept = self.reject is None or self.reject not in prompt
        out = {"accept": accept, "reason": "r", "passages": [self.passage]}
        resp = {
            "content": [{"type": "tool_use", "name": "emit_result", "input": out}],
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
        return HttpResponse(200, json.dumps(resp).encode())


@pytest.fixture
def cloud_env(monkeypatch):
    monkeypatch.setenv("DM_CLOUD_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DM_CLOUD_LLM_MODEL", "cloud-test")
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "25")
    monkeypatch.setenv("DM_CLOUD_COST_PER_MTOK_IN", "3")
    monkeypatch.setenv("DM_CLOUD_COST_PER_MTOK_OUT", "15")


def use_transport(monkeypatch, transport) -> None:
    monkeypatch.setattr(cli, "_gold_pick_transport", lambda: transport)


def llm_rows(engine) -> list[LlmCall]:
    with Session(engine) as session:
        return list(session.query(LlmCall).order_by(LlmCall.id))


def test_clean_replaces_rejected_and_logs_each_call(engine, tmp_path, cloud_env, monkeypatch):
    standard_db(engine)
    args = ("--entities", "2", "--per-entity", "1")
    assert pick(engine, tmp_path / "plain", *args).exit_code == 0
    first = lines(tmp_path / "plain" / "cells.jsonl")[0]["primary_domain"]
    transport = FakeTransport(reject=first, passage="They cover liability")
    use_transport(monkeypatch, transport)
    result = pick(engine, tmp_path / "out", *args, "--clean")
    assert result.exit_code == 0, result.output
    cells = lines(tmp_path / "out" / "cells.jsonl")
    assert len(cells) == 2
    assert first not in {c["primary_domain"] for c in cells}
    assert all(c["evidence"][0]["chunk_text"] == "They cover liability" for c in cells)
    rows = llm_rows(engine)
    assert len(rows) == transport.calls >= 3
    assert all(r.model == "cloud-test" and r.purpose == "gold_clean" for r in rows)
    with Session(engine) as session:
        assert session.get(Run, rows[0].run_id).stage == "gold-pick"
    report = CliRunner().invoke(cli.app, ["cost-report"], env={"DM_DATABASE_URL": engine.url_text})
    assert "stage gold-pick purpose gold_clean model cloud-test" in report.output


def test_clean_call_count_is_bounded(engine, tmp_path, cloud_env, monkeypatch):
    for i in range(6):
        seed(engine, f"e{i}.example", {"iso17100": "weak", "public_terms": "strong"})
    transport = FakeTransport(reject=".example")  # rejects everything
    use_transport(monkeypatch, transport)
    result = pick(engine, tmp_path / "out", "--entities", "2", "--per-entity", "1", "--clean")
    assert transport.calls == 4  # 2 x target
    assert result.exit_code == 0, result.output
    assert lines(tmp_path / "out" / "cells.jsonl") == []


def test_budget_is_cumulative_across_runs(engine, tmp_path, cloud_env, monkeypatch):
    standard_db(engine)
    with Session(engine) as session, session.begin():
        prior = Run(stage="gold-pick", status="finished", started_at=datetime.now(UTC))
        session.add(prior)
        session.flush()
        session.add(
            LlmCall(
                run_id=prior.id,
                model="cloud-test",
                purpose="gold_clean",
                prompt_hash="0" * 64,
                cost=24.999,
            )
        )
    transport = FakeTransport()
    use_transport(monkeypatch, transport)
    result = pick(engine, tmp_path / "out", "--clean")
    assert result.exit_code == 2
    assert "budget" in result.output
    assert transport.calls == 0
    assert len(llm_rows(engine)) == 1


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("DM_LLM_BUDGET_USD", "0"),
        ("DM_CLOUD_COST_PER_MTOK_IN", "0"),
        ("DM_CLOUD_COST_PER_MTOK_OUT", "0"),
    ],
)
def test_clean_refuses_without_live_cap(engine, tmp_path, cloud_env, monkeypatch, var, value):
    standard_db(engine)
    monkeypatch.setenv(var, value)
    transport = FakeTransport()
    use_transport(monkeypatch, transport)
    result = pick(engine, tmp_path / "out", "--clean")
    assert result.exit_code == 2
    assert transport.calls == 0
    assert not (tmp_path / "out" / "cells.jsonl").exists()


def test_clean_requires_cloud_backend(engine, tmp_path, cloud_env):
    result = pick(engine, tmp_path / "out", "--clean", "--backend", "local")
    assert result.exit_code == 2
