"""Tests for core.retrieve (AS-02): keyword-only evidence retrieval. Offline, SQLite."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from core.assess import extract_criterion
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from adapters.llm import LlmResult
from core.chunk_store import ChunkRow, store_chunks
from core.chunking import Chunk
from core.keyword_search import search_chunks
from core.retrieve import build_query, retrieve_evidence
from core.runs import create_all
from core.scope_pack import Criterion, Rubric
from core.snapshot_store import SnapshotRow


def make_engine(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'rt.db'}")
    create_all(engine)
    return engine


def add_snapshot(engine: Engine, entity_id: int | None) -> int:
    row = SnapshotRow(
        entity_id=entity_id,
        url="https://example.test/page",
        final_url="https://example.test/page",
        content_type="text/html",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_kind="web",
    )
    with Session(engine) as session, session.begin():
        session.add(row)
        session.flush()
        return row.id


def add_chunks(engine: Engine, snapshot_id: int, texts: list[str]) -> None:
    chunks = [
        Chunk(seq=i, text=t, page=None, heading_path=("S",), char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]
    store_chunks(engine, snapshot_id, chunks)


def stored_texts(engine: Engine) -> dict[int, str]:
    with Session(engine) as session:
        return {r.id: r.text for r in session.query(ChunkRow).all()}


def _rubric(hints: list[str] | None = None) -> tuple[Rubric, Criterion]:
    rubric = Rubric.model_validate(
        {
            "status_scale": [
                {"id": "strong", "label": "Strong", "value": 3, "color": "green"},
                {"id": "silent", "label": "Not stated", "value": None, "color": "white"},
            ],
            "topics": [
                {
                    "id": "terms",
                    "label": "Terms",
                    "criteria": [
                        {
                            "id": "public_terms",
                            "question": "Are general terms published?",
                            "applies_to": ["agency"],
                            "statuses": {"strong": "Terms published on the site."},
                            "retrieval_hints": hints if hints is not None else ["conditions"],
                        }
                    ],
                }
            ],
        }
    )
    return rubric, rubric.topics[0].criteria[0]


def test_build_query_rule() -> None:
    _, crit = _rubric(["alpha beta", "gamma"])
    assert build_query(crit) == "alpha beta gamma Are general terms published?"
    _, crit = _rubric([])
    assert build_query(crit) == "Are general terms published?"


def test_entity_filter(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    a, b = add_snapshot(engine, 1), add_snapshot(engine, 2)
    add_chunks(engine, a, ["general terms here"])
    add_chunks(engine, b, ["general terms there", "published terms conditions"])
    _, crit = _rubric()
    out = retrieve_evidence(engine, crit, entity_id=1)
    with Session(engine) as session:
        own = {r.id for r in session.query(ChunkRow).filter(ChunkRow.snapshot_id == a)}
    assert out and {cid for cid, _ in out} <= own


def test_at_most_n_in_score_order_with_stored_text(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(
        engine,
        snap,
        [f"terms {'published ' * i} filler" for i in range(8)] + ["unrelated words"],
    )
    _, crit = _rubric()
    out = retrieve_evidence(engine, crit, entity_id=1, n=3, k=6)
    hits = search_chunks(engine, build_query(crit), limit=6, entity_id=1)
    assert [cid for cid, _ in out] == [h.chunk_id for h in hits[:3]]
    texts = stored_texts(engine)
    assert all(text == texts[cid] for cid, text in out)


def test_hint_only_and_question_only_matches(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["the conditions apply", "nothing else", "general rules"])
    _, crit = _rubric(["conditions"])
    texts = [t for _, t in retrieve_evidence(engine, crit, entity_id=1)]
    assert "the conditions apply" in texts
    assert "general rules" in texts
    assert "nothing else" not in texts


def test_no_match_returns_empty(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["unrelated words only"])
    _, crit = _rubric()
    assert retrieve_evidence(engine, crit, entity_id=1) == []
    assert retrieve_evidence(engine, crit, entity_id=7) == []


def test_invalid_arguments(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    _, crit = _rubric()
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, n=0)
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, n=5, k=4)


class FakeLLM:
    model = "fake-local"

    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        self.prompts.append(prompt)
        return LlmResult(
            text="{}",
            parsed=self.parsed,  # type: ignore[arg-type]
            tokens_in=1,
            tokens_out=1,
            cost=0.0,
            cache_hit=False,
        )


def test_output_feeds_extract_criterion(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["Our general terms are published on this page."])
    rubric, crit = _rubric()
    chunks = retrieve_evidence(engine, crit, entity_id=1)
    cid = chunks[0][0]
    llm = FakeLLM(
        {
            "status": "strong",
            "quotes": [{"chunk_id": cid, "quote": "general terms are published"}],
            "rationale": "ok",
        }
    )
    result = extract_criterion(llm, rubric, crit, chunks, run_id=1)
    assert result.status == "strong" and result.checks_passed
    assert "Our general terms are published on this page." in llm.prompts[0]
