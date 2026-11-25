"""Tests for ``landscape index`` (IX-02); SQLite under a temp dir, fake embedder."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from adapters.embedding import EmbeddingError
from core import cli
from core.chunk_store import ChunkRow
from core.embed_store import EmbedStoreError
from core.runs import Run, create_all


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


def seed_chunks(engine, texts: list[str]) -> None:
    """Insert one chunk row per text (no snapshot FK is created)."""
    with Session(engine) as session, session.begin():
        for seq, text in enumerate(texts):
            session.add(
                ChunkRow(
                    snapshot_id=1,
                    seq=seq,
                    text=text,
                    lang=None,
                    page=None,
                    heading_path=[],
                    char_start=0,
                    char_end=len(text),
                )
            )


def run_rows(engine) -> list[tuple[str, str]]:
    with Session(engine) as session:
        return [(r.stage, r.status) for r in session.scalars(sa.select(Run).order_by(Run.id))]


def run_records(engine) -> list[Run]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(Run).order_by(Run.id)))


def chunk_states(engine) -> list[tuple[str, str | None]]:
    with Session(engine) as session:
        return [
            (r.text, r.embedding_model)
            for r in session.scalars(sa.select(ChunkRow).order_by(ChunkRow.id))
        ]


class FakeEmbedder:
    """Fake Embedder; ``fail_on_call`` raises ``error`` on that 1-based call number."""

    model = "fake-index-model"

    def __init__(self, *, fail_on_call: int | None = None, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call
        self.error = error or EmbeddingError("fake service down")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise self.error
        return [[float(len(t)), 1.0, 2.0] for t in texts]


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ["index", *args])


def use_embedder(monkeypatch, embedder: FakeEmbedder) -> None:
    monkeypatch.setattr(cli, "_make_embedder", lambda settings, model: embedder)


def test_success_prints_counts_and_closes_run_finished(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta", "gamma"])
    fake = FakeEmbedder()
    use_embedder(monkeypatch, fake)

    result = invoke("--model", "m1", "--batch-size", "2")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 3 batches 2"]
    assert [len(c) for c in fake.calls] == [2, 1]
    assert run_rows(engine) == [("index", "finished")]


def test_rerun_embeds_zero(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta"])
    fake = FakeEmbedder()
    use_embedder(monkeypatch, fake)

    first = invoke("--model", "m1")
    second = invoke("--model", "m1")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert second.stdout.splitlines() == ["embedded 0 batches 0"]
    assert len(fake.calls) == 1  # the rerun made no embed call


def test_limit_zero_embeds_nothing(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    fake = FakeEmbedder()
    use_embedder(monkeypatch, fake)

    result = invoke("--model", "m1", "--limit", "0")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 0 batches 0"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ([], "--model"),
        (["--model", ""], "--model"),
        (["--model", "m", "--batch-size", "0"], "--batch-size"),
        (["--model", "m", "--limit=-1"], "--limit"),
    ],
)
def test_invalid_values_exit_2_with_no_run_and_no_embed(
    engine, monkeypatch, args: list[str], fragment: str
) -> None:
    fake = FakeEmbedder()
    use_embedder(monkeypatch, fake)

    result = invoke(*args)

    assert result.exit_code == 2, result.output
    assert fragment in result.stderr
    assert run_rows(engine) == []
    assert fake.calls == []


def test_embedding_error_fails_run_and_exits_1(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta"])
    use_embedder(monkeypatch, FakeEmbedder(fail_on_call=1))

    result = invoke("--model", "m1")

    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == "error EmbeddingError: fake service down"
    assert run_rows(engine) == [("index", "failed")]


def test_embed_store_error_fails_run_and_exits_1(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    use_embedder(monkeypatch, FakeEmbedder(fail_on_call=1, error=EmbedStoreError("bad vectors")))

    result = invoke("--model", "m1")

    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == "error EmbedStoreError: bad vectors"
    assert run_rows(engine) == [("index", "failed")]


def test_earlier_batches_stay_committed_when_a_later_batch_fails(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta", "gamma"])
    use_embedder(monkeypatch, FakeEmbedder(fail_on_call=2))

    result = invoke("--model", "m1", "--batch-size", "1")

    assert result.exit_code == 1, result.output
    assert "error EmbeddingError: fake service down" in result.stderr
    assert chunk_states(engine) == [("alpha", "m1"), ("beta", None), ("gamma", None)]
    assert run_rows(engine) == [("index", "failed")]


def test_seam_receives_settings_and_model(engine, monkeypatch, db_url) -> None:
    seed_chunks(engine, ["alpha"])
    seen: list[tuple[object, str]] = []

    def seam(settings, model):
        seen.append((settings, model))
        return FakeEmbedder()

    monkeypatch.setattr(cli, "_make_embedder", seam)
    result = invoke("--model", "m1")

    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    settings, model = seen[0]
    assert model == "m1"
    assert settings.database_url == db_url


def test_unexpected_embedder_error_fails_run_and_propagates(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta"])
    use_embedder(monkeypatch, FakeEmbedder(fail_on_call=1, error=RuntimeError("boom")))

    result = invoke("--model", "m1")

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, RuntimeError)
    records = run_records(engine)
    assert [(r.stage, r.status) for r in records] == [("index", "failed")]
    assert records[0].finished_at is not None


class InterruptingEmbedder(FakeEmbedder):
    """Embedder whose first call is interrupted (a BaseException, not an Exception)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise KeyboardInterrupt


def test_keyboard_interrupt_fails_run_and_exits_130(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    use_embedder(monkeypatch, InterruptingEmbedder())

    result = invoke("--model", "m1")

    # typer converts the re-raised KeyboardInterrupt into exit 130.
    assert result.exit_code == 130, result.output
    records = run_records(engine)
    assert [(r.stage, r.status) for r in records] == [("index", "failed")]
    assert records[0].finished_at is not None


def test_unbuildable_embedder_exits_2_without_run(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    fake = FakeEmbedder()

    def seam(settings, model):
        raise ValueError("embedding service URL is not configured")

    monkeypatch.setattr(cli, "_make_embedder", seam)

    result = invoke("--model", "m1")

    assert result.exit_code == 2, result.output
    assert result.stderr.strip() == "error: embedding service URL is not configured"
    assert run_rows(engine) == []
    assert fake.calls == []
