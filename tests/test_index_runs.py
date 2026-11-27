"""RT-01 (135/A1): ``landscape index`` runs one StageRunner unit under its ``index`` run row."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app import cli
from core.chunk_store import ChunkRow
from core.runs import WorkUnit
from tests.test_cli_index import (  # noqa: F401  (fixtures)
    FakeEmbedder,
    db_url,
    engine,
    invoke,
    run_rows,
    seed_chunks,
    use_embedder,
)


def units(engine) -> list[tuple[str, str, str]]:
    with Session(engine) as session:
        return [
            (u.stage, u.unit_key, u.status)
            for u in session.scalars(sa.select(WorkUnit).order_by(WorkUnit.id))
        ]


def count_embed_calls(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = cli.embed_pending_chunks

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "embed_pending_chunks", spy)
    return calls


def test_first_index_creates_one_run_and_one_done_unit(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta", "gamma"])
    use_embedder(monkeypatch, FakeEmbedder())

    result = invoke("--model", "m1", "--batch-size", "2")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 3 batches 2"]
    assert run_rows(engine) == [("index", "finished")]
    assert units(engine) == [("index", "m1", "done")]


def test_unchanged_rerun_skips_unit_without_embedding(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    use_embedder(monkeypatch, FakeEmbedder())
    assert invoke("--model", "m1").exit_code == 0  # embeds
    calls = count_embed_calls(monkeypatch)

    result = invoke("--model", "m1")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 0 batches 0", "skipped 1"]
    assert calls == []
    assert run_rows(engine) == [("index", "finished")] * 2


def test_first_run_with_zero_pending_still_runs_unit(engine, monkeypatch) -> None:
    use_embedder(monkeypatch, FakeEmbedder())
    calls = count_embed_calls(monkeypatch)

    result = invoke("--model", "m1")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 0 batches 0"]
    assert calls == [1]
    assert units(engine) == [("index", "m1", "done")]


def test_new_pending_chunk_reruns_unit(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    fake = FakeEmbedder()
    use_embedder(monkeypatch, fake)
    invoke("--model", "m1")
    with Session(engine) as session, session.begin():
        session.add(
            ChunkRow(
                snapshot_id=2,
                seq=0,
                text="beta",
                lang=None,
                page=None,
                heading_path=[],
                char_start=0,
                char_end=4,
            )
        )

    result = invoke("--model", "m1")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["embedded 1 batches 1"]
    assert fake.calls[-1] == ["beta"]


def test_changed_limit_or_batch_size_reruns_unit(engine, monkeypatch) -> None:
    use_embedder(monkeypatch, FakeEmbedder())
    invoke("--model", "m1")
    calls = count_embed_calls(monkeypatch)

    assert invoke("--model", "m1", "--batch-size", "7").stdout.splitlines() == [
        "embedded 0 batches 0"
    ]
    assert invoke("--model", "m1", "--batch-size", "7", "--limit", "3").exit_code == 0
    assert calls == [1, 1]


def test_embedding_error_fails_unit_and_run(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha"])
    use_embedder(monkeypatch, FakeEmbedder(fail_on_call=1))

    result = invoke("--model", "m1")

    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == "error EmbeddingError: fake service down"
    assert run_rows(engine) == [("index", "failed")]
    assert units(engine) == [("index", "m1", "failed")]


def test_unbuildable_embedder_leaves_no_run_or_unit(engine, monkeypatch) -> None:
    def seam(settings, model):
        raise ValueError("no backend")

    monkeypatch.setattr(cli, "_make_embedder", seam)

    result = invoke("--model", "m1")

    assert result.exit_code == 2, result.output
    assert run_rows(engine) == []
    assert units(engine) == []


def test_bounded_run_continues_on_next_invocation(engine, monkeypatch) -> None:
    seed_chunks(engine, ["alpha", "beta"])
    use_embedder(monkeypatch, FakeEmbedder())
    assert invoke("--model", "m1", "--limit", "1").stdout.splitlines() == ["embedded 1 batches 1"]

    result = invoke("--model", "m1", "--limit", "1")

    assert result.stdout.splitlines() == ["embedded 1 batches 1"]
    again = invoke("--model", "m1", "--limit", "1")
    assert again.stdout.splitlines() == ["embedded 0 batches 0"]  # pending 1 -> 0 changes hash
    assert invoke("--model", "m1", "--limit", "1").stdout.splitlines()[-1] == "skipped 1"
