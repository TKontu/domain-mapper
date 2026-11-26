"""Tests for ``landscape assess`` (AS-03); SQLite under a temp dir, fake local LLM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.llm import LlmError, LlmResult
from core import cli
from core.assessment_store import AssessmentRow, EvidenceRow
from core.chunk_store import store_chunks
from core.chunking import Chunk
from core.entity_store import Entity
from core.runs import FAILED, FINISHED, Run, create_all
from core.snapshot_store import SnapshotRow

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")
CRIT = "public_terms"
OTHER_MARKER = "OTHER-ENTITY-MARKER-zz9"


class FakeLLM:
    model = "fake-local"

    def __init__(self, parsed: object = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
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


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    engine = sa.create_engine(url)
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def seed(engine, name: str, texts: list[str]) -> tuple[int, list[int]]:
    with Session(engine) as session, session.begin():
        entity = Entity(
            scope_id="example", name=name, primary_domain=f"{name}.example", status="in_scope"
        )
        session.add(entity)
        session.flush()
        entity_id = entity.id
        snap = SnapshotRow(
            entity_id=entity_id,
            url=f"https://{name}.example/",
            final_url=f"https://{name}.example/",
            content_type="text/html",
            sha256="0" * 64,
            fetched_at=datetime.now(UTC),
            source_kind="web",
        )
        session.add(snap)
        session.flush()
        snap_id = snap.id
    chunks = [
        Chunk(seq=i, text=t, page=None, heading_path=("S",), char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]
    store_chunks(engine, snap_id, chunks)
    with engine.connect() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                sa.text("SELECT id FROM chunk WHERE snapshot_id = :s ORDER BY seq"), {"s": snap_id}
            )
        ]
    return entity_id, ids


def runs(engine) -> list[tuple[str, str]]:
    with Session(engine) as session:
        return [(r.stage, r.status) for r in session.query(Run).all()]


def assessments(engine) -> list[AssessmentRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(AssessmentRow).order_by(AssessmentRow.id)))


def evidence(engine) -> list[EvidenceRow]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(EvidenceRow).order_by(EvidenceRow.id)))


def run_id(engine) -> int:
    with Session(engine) as session:
        return session.scalar(sa.select(Run.id))


AGREE = {"agree": True, "reason": "Quote supports the status."}


def invoke(llm: FakeLLM | None, *args: str, verifier: FakeLLM | None = None) -> Any:
    fake = llm or FakeLLM()
    vfake = verifier or FakeLLM(AGREE)
    with (
        mock.patch.object(cli, "_make_assess_llm", return_value=fake),
        mock.patch.object(cli, "_make_verify_llm", return_value=vfake),
    ):
        return CliRunner().invoke(cli.app, ["assess", "--scope", EXAMPLE_SCOPE, *args])


def test_happy_path_grounded(engine) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    seed(engine, "other", [f"Terms and conditions {OTHER_MARKER}."])
    llm = FakeLLM(
        {
            "status": "strong",
            "quotes": [{"chunk_id": ids[0], "quote": "general terms and conditions"}],
            "rationale": "Published.",
        }
    )
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 0, result.output
    [row] = assessments(engine)
    lines = result.output.splitlines()
    assert lines[0] == "status strong checks pass chunks 1"
    assert lines[1] == f'quote chunk {ids[0]} grounded exact "general terms and conditions"'
    assert lines[2] == "rationale Published."
    assert lines[3] == "route accept confidence high reason Quote supports the status."
    assert len(lines) == 5
    assert lines[4] == f"assessment {row.id}"
    assert runs(engine) == [("assess", FINISHED)]
    assert (row.entity_id, row.criterion_id) == (eid, CRIT)
    assert (row.status, row.route_kind, row.decided_by) == ("strong", "accept", "auto")
    assert row.confidence is None and row.run_id == run_id(engine)
    assert row.rubric_version == "0.0.1"
    assert (row.extractor_model, row.verifier_model) == ("fake-local", "fake-local")
    assert [(e.chunk_id, e.quote, e.grounded) for e in evidence(engine)] == [
        (ids[0], "general terms and conditions", True)
    ]
    # (f) only this entity's chunks reach the prompt
    assert len(llm.prompts) == 1 and OTHER_MARKER not in llm.prompts[0]


def test_ungrounded_fails_checks(engine) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    llm = FakeLLM(
        {
            "status": "strong",
            "quotes": [{"chunk_id": ids[0], "quote": "invented liability text"}],
            "rationale": "r",
        }
    )
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert "status strong checks fail chunks 1" in result.output
    assert f'quote chunk {ids[0]} ungrounded - "invented liability text"' in result.output
    assert "failure quote 0: not found verbatim" in result.output
    assert "failure evidence_min:" in result.output
    assert runs(engine) == [("assess", FINISHED)]
    # A failed-checks outcome is still an outcome: one adjudicate row, evidence ungrounded.
    [row] = assessments(engine)
    assert (row.status, row.route_kind) == ("strong", "adjudicate")
    assert result.output.splitlines()[-1] == f"assessment {row.id}"
    assert [(e.chunk_id, e.grounded) for e in evidence(engine)] == [(ids[0], False)]


def test_no_chunks_is_silent(engine) -> None:
    eid, _ = seed(engine, "acme", ["Nothing relevant at all."])
    llm = FakeLLM()
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "status silent checks pass chunks 0"
    assert llm.prompts == []
    [row] = assessments(engine)
    assert (row.status, row.route_kind) == ("silent", "silent")
    assert evidence(engine) == []


def test_silent_makes_no_verifier_call(engine) -> None:
    eid, _ = seed(engine, "acme", ["Nothing relevant at all."])
    verifier = FakeLLM(AGREE)
    result = invoke(FakeLLM(), "--entity", str(eid), "--criterion", CRIT, verifier=verifier)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[-2].startswith("route silent confidence - reason ")
    assert lines[-1] == f"assessment {assessments(engine)[0].id}"
    assert verifier.prompts == []


GROUNDED = {"status": "strong", "quotes": [], "rationale": "Published."}


def _grounded(ids: list[int]) -> dict[str, object]:
    return {**GROUNDED, "quotes": [{"chunk_id": ids[0], "quote": "general terms and conditions"}]}


def test_verifier_disagree_adjudicates_exit_1(engine) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    verifier = FakeLLM({"agree": False, "reason": "Quote is too generic."})
    result = invoke(
        FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT, verifier=verifier
    )
    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    assert lines[-2] == (
        "route adjudicate confidence - reason verifier disagrees: Quote is too generic."
    )
    assert lines[-1] == f"assessment {assessments(engine)[0].id}"
    assert len(verifier.prompts) == 1
    assert runs(engine) == [("assess", FINISHED)]
    [row] = assessments(engine)
    assert (row.status, row.route_kind) == ("strong", "adjudicate")
    assert len(evidence(engine)) == 1


@pytest.mark.parametrize(
    "verifier", [FakeLLM({"agree": "yes", "reason": "r"}), FakeLLM(error=LlmError("vboom"))]
)
def test_verifier_failure_exit_1_run_failed(engine, verifier: FakeLLM) -> None:
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    result = invoke(
        FakeLLM(_grounded(ids)), "--entity", str(eid), "--criterion", CRIT, verifier=verifier
    )
    assert result.exit_code == 1, result.output
    assert "error AssessError: verify output" in result.output or "error LlmError: vboom" in (
        result.output
    )
    assert "route " not in result.output
    assert "assessment " not in result.output
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []


def test_verifier_built_from_verifier_model(engine, monkeypatch) -> None:
    monkeypatch.setenv("DM_VERIFIER_MODEL", "verifier-x")
    eid, ids = seed(engine, "acme", ["Our general terms and conditions are published here."])
    built: list[tuple[str, str]] = []
    verifier = FakeLLM(AGREE)

    def fake_build(settings, backend, *, db, budget_usd=None):  # type: ignore[no-untyped-def]
        built.append((backend, settings.local_model))
        return verifier

    with (
        mock.patch.object(cli, "_make_assess_llm", return_value=FakeLLM(_grounded(ids))),
        mock.patch.object(cli, "build_llm", side_effect=fake_build),
    ):
        result = CliRunner().invoke(
            cli.app,
            ["assess", "--scope", EXAMPLE_SCOPE, "--entity", str(eid), "--criterion", CRIT],
        )
    assert result.exit_code == 0, result.output
    assert built == [("local", "verifier-x")]
    assert len(verifier.prompts) == 1


def test_verifier_model_setting_default(monkeypatch, tmp_path) -> None:
    from core.config import DEFAULT_VERIFIER_MODEL, load_settings

    monkeypatch.delenv("DM_VERIFIER_MODEL", raising=False)
    assert load_settings(tmp_path / "none.env").verifier_model == DEFAULT_VERIFIER_MODEL
    monkeypatch.setenv("DM_VERIFIER_MODEL", "v2")
    assert load_settings(tmp_path / "none.env").verifier_model == "v2"


@pytest.mark.parametrize(
    "extra",
    [
        ["--criterion", "nope"],
        ["--criterion", CRIT, "--n", "0"],
        ["--criterion", CRIT, "--n", "5", "--k", "4"],
        [],
    ],
)
def test_bad_args_exit_2_no_run(engine, extra: list[str]) -> None:
    result = invoke(None, "--entity", "1", *extra)
    assert result.exit_code == 2, result.output
    assert runs(engine) == []
    assert assessments(engine) == []
    if extra[:2] == ["--criterion", "nope"]:
        assert "error: unknown criterion nope" in result.output


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(error=LlmError("boom")), FakeLLM({"status": "bogus", "quotes": [], "rationale": ""})],
)
def test_llm_failure_exit_1_run_failed(engine, llm: FakeLLM) -> None:
    eid, _ = seed(engine, "acme", ["Our general terms and conditions are published here."])
    result = invoke(llm, "--entity", str(eid), "--criterion", CRIT)
    assert result.exit_code == 1, result.output
    assert "error LlmError: boom" in result.output or "error AssessError:" in result.output
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []


def test_database_unavailable(tmp_path, monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/missing/dir/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    result = invoke(None, "--entity", "1", "--criterion", CRIT)
    assert result.exit_code == 2, result.output
    assert "database unavailable" in result.output
    assert url not in result.output


def test_dry_run_still_works(engine) -> None:
    result = CliRunner().invoke(cli.app, ["assess", "--scope", EXAMPLE_SCOPE, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "stage: assess" in result.output


# AS-07: --embedding-model wires a hybrid Embedder into retrieval.

VECTOR_ONLY = "Zebra quokka nimbus lattice."


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.model = "fake-embed"
        self.error = error
        self.calls: list[list[str]] = []

    def embed(self, texts: Any) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


def _set_embedding(engine, chunk_id: int, vector: list[float], model: str) -> None:
    from core.chunk_store import ChunkRow

    with Session(engine) as session, session.begin():
        session.execute(
            sa.update(ChunkRow)
            .where(ChunkRow.id == chunk_id)
            .values(embedding=vector, embedding_model=model)
        )


def invoke_embed(embedder: Any, *args: str, llm: FakeLLM | None = None) -> tuple[Any, Any]:
    built = mock.MagicMock(side_effect=embedder) if isinstance(embedder, Exception) else None
    seam = built or mock.MagicMock(return_value=embedder)
    with mock.patch.object(cli, "_make_embedder", seam):
        result = invoke(llm, *args)
    return result, seam


def test_embedding_model_hybrid_reaches_extractor(engine) -> None:
    eid, ids = seed(engine, "acme", [VECTOR_ONLY])
    _set_embedding(engine, ids[0], [1.0, 0.0], "stored-embed")
    llm = FakeLLM({"status": "silent", "quotes": [], "rationale": "none"})
    fake = FakeEmbedder()
    result, seam = invoke_embed(
        fake,
        "--entity",
        str(eid),
        "--criterion",
        CRIT,
        "--embedding-model",
        "stored-embed",
        llm=llm,
    )
    assert result.exit_code in (0, 1), result.output
    assert seam.call_args.args[1] == "stored-embed"
    assert len(fake.calls) == 1
    assert "chunks 1" in result.output
    assert any(VECTOR_ONLY in p for p in llm.prompts)


def test_without_embedding_model_no_embedder(engine) -> None:
    eid, ids = seed(engine, "acme", [VECTOR_ONLY])
    _set_embedding(engine, ids[0], [1.0, 0.0], "stored-embed")
    result, seam = invoke_embed(FakeEmbedder(), "--entity", str(eid), "--criterion", CRIT)
    assert seam.call_count == 0
    assert "chunks 0" in result.output


def test_embedding_model_bad_backend_exit_2_no_run(engine) -> None:
    eid, _ = seed(engine, "acme", ["text"])
    result, _ = invoke_embed(
        ValueError("unknown embedding backend"),
        "--entity", str(eid), "--criterion", CRIT, "--embedding-model", "m",
    )  # fmt: skip
    assert result.exit_code == 2, result.output
    assert "error: unknown embedding backend" in result.output
    assert runs(engine) == []


def test_embedding_error_exit_1_run_failed(engine) -> None:
    from adapters.embedding import EmbeddingError

    eid, _ = seed(engine, "acme", ["text"])
    result, _ = invoke_embed(
        FakeEmbedder(EmbeddingError("embed down")),
        "--entity", str(eid), "--criterion", CRIT, "--embedding-model", "m",
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "error EmbeddingError: embed down" in result.output
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []


@pytest.mark.parametrize("value", ["", "   "])
def test_embedding_model_blank_exit_2(engine, value: str) -> None:
    result, seam = invoke_embed(
        FakeEmbedder(), "--entity", "1", "--criterion", CRIT, "--embedding-model", value
    )
    assert result.exit_code == 2, result.output
    assert "error: --embedding-model must be non-empty" in result.output
    assert seam.call_count == 0
    assert runs(engine) == []


# AS-09: --rerank-model wires a Reranker into retrieval.

TERMS_A = "Alpha general terms and conditions apply."
TERMS_B = "Beta general terms and conditions apply."


class FakeReranker:
    def __init__(self, error: Exception | None = None, bad_index: bool = False) -> None:
        self.model = "fake-rerank"
        self.error = error
        self.bad_index = bad_index
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Any, top_n: int) -> list[Any]:
        from adapters.embedding import RerankHit

        docs = list(documents)
        self.calls.append((query, docs, top_n))
        if self.error is not None:
            raise self.error
        if self.bad_index:
            return [RerankHit(index=len(docs) + 5, score=1.0)]
        # Prefer the Beta chunk: reverse of any keyword order tie-break.
        return [RerankHit(index=i, score=1.0 if "Beta" in d else 0.1) for i, d in enumerate(docs)]


def invoke_rerank(
    reranker: Any, *args: str, llm: FakeLLM | None = None, embedder: Any = None
) -> tuple[Any, Any]:
    built = mock.MagicMock(side_effect=reranker) if isinstance(reranker, Exception) else None
    seam = built or mock.MagicMock(return_value=reranker)
    with (
        mock.patch.object(cli, "_make_reranker", seam),
        mock.patch.object(cli, "_make_embedder", mock.MagicMock(return_value=embedder)),
    ):
        result = invoke(llm, *args)
    return result, seam


def _assert_beta_first(llm: FakeLLM) -> None:
    prompt = llm.prompts[0]
    assert prompt.index(TERMS_B) < prompt.index(TERMS_A)


def test_rerank_model_order_reaches_extractor(engine) -> None:
    eid, _ = seed(engine, "acme", [TERMS_A, TERMS_B])
    llm = FakeLLM({"status": "silent", "quotes": [], "rationale": "none"})
    fake = FakeReranker()
    result, seam = invoke_rerank(
        fake, "--entity", str(eid), "--criterion", CRIT, "--n", "1", "--k", "4",
        "--rerank-model", "rr", llm=llm,
    )  # fmt: skip
    assert result.exit_code in (0, 1), result.output
    assert seam.call_args.args[1] == "rr"
    assert len(fake.calls) == 1 and fake.calls[0][2] == 1
    assert "chunks 1" in result.output
    assert TERMS_B in llm.prompts[0] and TERMS_A not in llm.prompts[0]


def test_rerank_model_with_embedding_model(engine) -> None:
    eid, ids = seed(engine, "acme", [TERMS_A, TERMS_B])
    for cid in ids:
        _set_embedding(engine, cid, [1.0, 0.0], "stored-embed")
    llm = FakeLLM({"status": "silent", "quotes": [], "rationale": "none"})
    fake = FakeReranker()
    embedder = FakeEmbedder()
    result, _ = invoke_rerank(
        fake, "--entity", str(eid), "--criterion", CRIT, "--embedding-model", "stored-embed",
        "--rerank-model", "rr", llm=llm, embedder=embedder,
    )  # fmt: skip
    assert result.exit_code in (0, 1), result.output
    assert len(embedder.calls) == 1 and len(fake.calls) == 1
    assert "chunks 2" in result.output
    _assert_beta_first(llm)


def test_without_rerank_model_no_reranker(engine) -> None:
    eid, _ = seed(engine, "acme", [TERMS_A, TERMS_B])
    llm = FakeLLM({"status": "silent", "quotes": [], "rationale": "none"})
    result, seam = invoke_rerank(FakeReranker(), "--entity", str(eid), "--criterion", CRIT, llm=llm)
    assert seam.call_count == 0
    assert "chunks 2" in result.output


def test_rerank_model_bad_backend_exit_2_no_run(engine) -> None:
    eid, _ = seed(engine, "acme", [TERMS_A])
    result, _ = invoke_rerank(
        ValueError("bad reranker url"),
        "--entity", str(eid), "--criterion", CRIT, "--rerank-model", "rr",
    )  # fmt: skip
    assert result.exit_code == 2, result.output
    assert "error: bad reranker url" in result.output
    assert runs(engine) == []


def test_rerank_error_exit_1_run_failed(engine) -> None:
    from adapters.embedding import EmbeddingError

    eid, _ = seed(engine, "acme", [TERMS_A])
    result, _ = invoke_rerank(
        FakeReranker(EmbeddingError("rerank down")),
        "--entity", str(eid), "--criterion", CRIT, "--rerank-model", "rr",
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "error EmbeddingError: rerank down" in result.output
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []


def test_rerank_out_of_range_exit_1_run_failed(engine) -> None:
    eid, _ = seed(engine, "acme", [TERMS_A])
    result, _ = invoke_rerank(
        FakeReranker(bad_index=True),
        "--entity", str(eid), "--criterion", CRIT, "--rerank-model", "rr",
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "error ValueError:" in result.output
    assert runs(engine) == [("assess", FAILED)]
    assert assessments(engine) == []


@pytest.mark.parametrize("value", ["", "   "])
def test_rerank_model_blank_exit_2(engine, value: str) -> None:
    result, seam = invoke_rerank(
        FakeReranker(), "--entity", "1", "--criterion", CRIT, "--rerank-model", value
    )
    assert result.exit_code == 2, result.output
    assert "error: --rerank-model must be non-empty" in result.output
    assert seam.call_count == 0
    assert runs(engine) == []
