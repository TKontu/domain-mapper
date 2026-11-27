"""RT-01: ``landscape sources`` runs through StageRunner (assignment 136/A1); temp SQLite."""

from __future__ import annotations

from unittest import mock

import sqlalchemy as sa

from core.runs import Run, WorkUnit
from core.source_catalogue import Source
from tests.test_cli_sources import db_url, invoke, scope  # noqa: F401 - fixtures


def _rows(url: str, stmt) -> list[tuple]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return [tuple(r) for r in conn.execute(stmt)]
    finally:
        engine.dispose()


def runs(url: str) -> list[tuple]:
    return _rows(url, sa.select(Run.stage, Run.status).order_by(Run.id))


def units(url: str) -> list[tuple]:
    return _rows(
        url,
        sa.select(WorkUnit.stage, WorkUnit.unit_key, WorkUnit.status, WorkUnit.attempts),
    )


def test_first_run_one_finished_run_and_done_unit(db_url, scope):  # noqa: F811
    result = invoke(scope)
    assert result.exit_code == 0, result.output
    assert result.stdout == "inserted 3 updated 0 unchanged 0\n"
    assert runs(db_url) == [("sources", "finished")]
    assert units(db_url) == [("sources", "example", "done", 1)]


def test_unchanged_rerun_skips_unit(db_url, scope):  # noqa: F811
    assert invoke(scope).exit_code == 0
    # The pre-seed hash saw an empty table: one no-op pass, then reruns skip.
    assert invoke(scope).stdout == "inserted 0 updated 0 unchanged 3\n"
    with mock.patch("app.cli.seed_sources") as seeded:
        result = invoke(scope)
    seeded.assert_not_called()
    assert result.exit_code == 0, result.output
    assert result.stdout == "inserted 0 updated 0 unchanged 0\nskipped 1\n"
    assert runs(db_url) == [("sources", "finished")] * 3
    assert [u[2] for u in units(db_url)] == ["done", "done"]


def test_changed_pack_source_reruns_unit(db_url, scope):  # noqa: F811
    assert invoke(scope).exit_code == 0
    path = scope / "sources.yaml"
    path.write_text(path.read_text().replace("licence: open (fictional)", "licence: changed"))
    result = invoke(scope)
    assert result.exit_code == 0, result.output
    assert result.stdout == "inserted 0 updated 1 unchanged 2\n"
    assert [u[2] for u in units(db_url)].count("done") == 2


def test_deleted_source_row_reruns_unit(db_url, scope):  # noqa: F811
    assert invoke(scope).exit_code == 0
    engine = sa.create_engine(db_url)
    try:
        with engine.begin() as conn:
            top = conn.execute(sa.select(sa.func.max(Source.id))).scalar_one()
            conn.execute(sa.delete(Source).where(Source.id == top))
    finally:
        engine.dispose()
    result = invoke(scope)
    assert result.exit_code == 0, result.output
    assert result.stdout == "inserted 1 updated 0 unchanged 2\n"


def test_seed_failure_fails_unit_and_run(db_url, scope):  # noqa: F811
    with mock.patch("app.cli.seed_sources", side_effect=RuntimeError(db_url)):
        result = invoke(scope)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "error: database unavailable (RuntimeError)\n"
    assert db_url not in result.output
    assert runs(db_url) == [("sources", "failed")]
    assert units(db_url) == [("sources", "example", "failed", 1)]


def test_pack_error_leaves_no_rows(db_url, scope):  # noqa: F811
    (scope / "sources.yaml").write_text("sources: [{id: 1}]\n")
    result = invoke(scope)
    assert result.exit_code == 2
    engine = sa.create_engine(db_url)
    try:
        names = sa.inspect(engine).get_table_names()
    finally:
        engine.dispose()
    if "run" in names:
        assert runs(db_url) == []
        assert units(db_url) == []
