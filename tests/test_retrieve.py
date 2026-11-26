"""Tests for core.retrieve (AS-02, AS-06, AS-08): keyword/hybrid retrieval and reranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from adapters.embedding import RerankHit
from adapters.llm import LlmResult
from core.assess import extract_criterion
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


def ids_by_text(engine: Engine) -> dict[str, int]:
    with Session(engine) as session:
        return {r.text: r.id for r in session.scalars(sa.select(ChunkRow))}


def set_embedding(
    engine: Engine, chunk_id: int, vector: list[float], model: str = "fake-embed"
) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            sa.update(ChunkRow)
            .where(ChunkRow.id == chunk_id)
            .values(embedding=vector, embedding_model=model)
        )


class FakeEmbedder:
    def __init__(self, vector: list[float], model: str = "fake-embed") -> None:
        self.model = model
        self.vector = vector
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


class BadShapeEmbedder:
    model = "fake-embed"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.vectors


class FakeReranker:
    """Score documents by position (descending unless ``scores`` is given)."""

    model = "fake-rerank"

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.scores = None if scores is None else list(scores)
        self.queries: list[str] = []
        self.documents: list[list[str]] = []
        self.top_ns: list[int] = []

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]:
        docs = list(documents)
        self.queries.append(query)
        self.documents.append(docs)
        self.top_ns.append(top_n)
        if self.scores is None:
            scores = [float(len(docs) - position) for position in range(len(docs))]
        else:
            scores = (self.scores + [0.0] * len(docs))[: len(docs)]
        hits = [RerankHit(index=index, score=score) for index, score in enumerate(scores)]
        hits.sort(key=lambda hit: (-hit.score, hit.index))
        return hits[:top_n]


class FixedReranker:
    """Return a fixed hit list, ignoring ``top_n``, to exercise the caller's checks."""

    model = "fake-rerank"

    def __init__(self, hits: list[RerankHit]) -> None:
        self.hits = hits

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]:
        return list(self.hits)


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


def test_hybrid_fuses_both_legs_with_rrf(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    own = add_snapshot(engine, 1)
    other = add_snapshot(engine, 2)
    add_chunks(engine, own, ["hint only", "nothing relevant", "hint relevant", "other words"])
    add_chunks(engine, other, ["other entity"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["hint only"], [-1.0, 0.0])
    set_embedding(engine, ids["nothing relevant"], [1.0, 0.0])
    set_embedding(engine, ids["hint relevant"], [0.0, 1.0])
    set_embedding(engine, ids["other words"], [0.6, 0.8])
    set_embedding(engine, ids["other entity"], [1.0, 0.0])
    _, crit = _rubric(["hint"])
    fake = FakeEmbedder([1.0, 0.0])

    out = retrieve_evidence(engine, crit, entity_id=1, n=3, k=3, embedder=fake)

    assert fake.calls == [[crit.question]]
    # "hint relevant" is found by both legs and wins; "hint only" (keyword
    # only) and "nothing relevant" (vector only) tie on 1/(60+1) and break by id.
    assert [cid for cid, _ in out] == [
        ids["hint relevant"],
        ids["hint only"],
        ids["nothing relevant"],
    ]
    assert [text for _, text in out] == ["hint relevant", "hint only", "nothing relevant"]
    texts = stored_texts(engine)
    assert all(text == texts[cid] for cid, text in out)


def test_hybrid_model_selects_stored_embedding_group(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["hint only", "nothing relevant"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["hint only"], [-1.0, 0.0], "other-model")
    set_embedding(engine, ids["nothing relevant"], [1.0, 0.0], "other-model")
    _, crit = _rubric(["hint"])
    fake = FakeEmbedder([1.0, 0.0])

    # No stored vectors for the embedder's own model: keyword leg only.
    out = retrieve_evidence(engine, crit, entity_id=1, n=2, k=2, embedder=fake)
    assert [cid for cid, _ in out] == [ids["hint only"]]

    out = retrieve_evidence(engine, crit, entity_id=1, n=2, k=2, embedder=fake, model="other-model")
    assert [cid for cid, _ in out] == [ids["hint only"], ids["nothing relevant"]]


def test_hybrid_honours_entity_filter(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    own = add_snapshot(engine, 1)
    other = add_snapshot(engine, 2)
    add_chunks(engine, own, ["own page words"])
    add_chunks(engine, other, ["other page words"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["own page words"], [1.0, 0.0])
    set_embedding(engine, ids["other page words"], [1.0, 0.0])
    _, crit = _rubric([])
    fake = FakeEmbedder([1.0, 0.0])

    out = retrieve_evidence(engine, crit, entity_id=1, n=6, k=6, embedder=fake)
    assert [text for _, text in out] == ["own page words"]


def test_embedder_none_path_ignores_model(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["general terms here", "nothing else"])
    _, crit = _rubric()

    baseline = retrieve_evidence(engine, crit, entity_id=1, n=3, k=6)
    assert retrieve_evidence(engine, crit, entity_id=1, n=3, k=6, model="ignored") == baseline
    assert baseline


def test_hybrid_argument_validation_before_embedding(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    _, crit = _rubric()
    fake = FakeEmbedder([1.0])
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, n=0, embedder=fake)
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, n=5, k=4, embedder=fake)
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, embedder=FakeEmbedder([1.0], model=""))
    assert fake.calls == []


def test_hybrid_requires_one_well_formed_vector(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    _, crit = _rubric()
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, embedder=BadShapeEmbedder([]))
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, embedder=BadShapeEmbedder([[1.0], [2.0]]))


def test_hybrid_propagates_dimension_mismatch(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, ["unrelated words"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["unrelated words"], [1.0, 0.0])
    _, crit = _rubric([])
    with pytest.raises(ValueError):
        retrieve_evidence(engine, crit, entity_id=1, embedder=FakeEmbedder([1.0, 0.0, 0.0]))


def test_reranker_reorders_keyword_pool_and_receives_question(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, [f"terms passage {i}" for i in range(5)])
    _, crit = _rubric(["terms"])
    pool_ids = [
        hit.chunk_id for hit in search_chunks(engine, build_query(crit), limit=4, entity_id=1)
    ]
    assert len(pool_ids) == 4
    texts = stored_texts(engine)
    reranker = FakeReranker(scores=[0.1, 0.2, 0.3, 0.4])

    out = retrieve_evidence(engine, crit, entity_id=1, n=2, k=4, reranker=reranker)

    assert reranker.queries == [crit.question]
    assert reranker.top_ns == [2]
    assert reranker.documents == [[texts[cid] for cid in pool_ids]]
    assert len(reranker.documents[0]) <= 4
    assert [cid for cid, _ in out] == [pool_ids[3], pool_ids[2]]
    assert [text for _, text in out] == [texts[cid] for cid in (pool_ids[3], pool_ids[2])]


def test_reranker_reorders_hybrid_fused_pool(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    own = add_snapshot(engine, 1)
    add_chunks(engine, own, ["hint only", "nothing relevant", "hint relevant", "other words"])
    ids = ids_by_text(engine)
    set_embedding(engine, ids["hint only"], [-1.0, 0.0])
    set_embedding(engine, ids["nothing relevant"], [1.0, 0.0])
    set_embedding(engine, ids["hint relevant"], [0.0, 1.0])
    set_embedding(engine, ids["other words"], [0.6, 0.8])
    _, crit = _rubric(["hint"])
    embedder = FakeEmbedder([1.0, 0.0])
    fused_pool = retrieve_evidence(engine, crit, entity_id=1, n=4, k=4, embedder=embedder)
    pool_ids = [cid for cid, _ in fused_pool]
    assert len(pool_ids) == 4
    reranker = FakeReranker(scores=[0.1, 0.2, 0.3, 0.4])

    out = retrieve_evidence(
        engine, crit, entity_id=1, n=2, k=4, embedder=embedder, reranker=reranker
    )

    assert reranker.queries == [crit.question]
    assert reranker.top_ns == [2]
    assert reranker.documents == [[text for _, text in fused_pool]]
    assert [cid for cid, _ in out] == [pool_ids[3], pool_ids[2]]


def test_reranker_orders_distinct_scores_descending(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, [f"terms passage {i}" for i in range(4)])
    _, crit = _rubric(["terms"])
    pool_ids = [
        hit.chunk_id for hit in search_chunks(engine, build_query(crit), limit=4, entity_id=1)
    ]
    reranker = FixedReranker(
        [
            RerankHit(index=0, score=0.2),
            RerankHit(index=2, score=0.9),
            RerankHit(index=1, score=0.5),
        ]
    )

    out = retrieve_evidence(engine, crit, entity_id=1, n=3, k=4, reranker=reranker)

    assert [cid for cid, _ in out] == [pool_ids[2], pool_ids[1], pool_ids[0]]


def test_reranker_ties_break_by_chunk_id_ascending(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, [f"terms passage {i}" for i in range(4)])
    _, crit = _rubric(["terms"])
    pool_ids = [
        hit.chunk_id for hit in search_chunks(engine, build_query(crit), limit=4, entity_id=1)
    ]
    reranker = FixedReranker(
        [
            RerankHit(index=3, score=0.9),
            RerankHit(index=1, score=0.9),
            RerankHit(index=2, score=0.5),
            RerankHit(index=0, score=0.5),
        ]
    )

    out = retrieve_evidence(engine, crit, entity_id=1, n=2, k=4, reranker=reranker)

    assert [cid for cid, _ in out] == [pool_ids[1], pool_ids[3]]


def test_reranker_output_is_capped_at_n(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, [f"terms passage {i}" for i in range(4)])
    _, crit = _rubric(["terms"])
    pool_ids = [
        hit.chunk_id for hit in search_chunks(engine, build_query(crit), limit=4, entity_id=1)
    ]
    reranker = FixedReranker(
        [
            RerankHit(index=0, score=0.9),
            RerankHit(index=1, score=0.8),
            RerankHit(index=2, score=0.7),
        ]
    )

    out = retrieve_evidence(engine, crit, entity_id=1, n=1, k=4, reranker=reranker)

    assert [cid for cid, _ in out] == [pool_ids[0]]


def test_reranker_rejects_out_of_range_index(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    snap = add_snapshot(engine, 1)
    add_chunks(engine, snap, [f"terms passage {i}" for i in range(4)])
    _, crit = _rubric(["terms"])

    for bad_index in (4, -1):
        reranker = FixedReranker([RerankHit(index=bad_index, score=0.9)])
        with pytest.raises(ValueError):
            retrieve_evidence(engine, crit, entity_id=1, n=2, k=4, reranker=reranker)
