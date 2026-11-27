"""Tests for ``landscape sources`` (AD-15); temp SQLite, CliRunner."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

from app import cli

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


@pytest.fixture
def db_url(tmp_path, monkeypatch) -> str:
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path}/dm.sqlite"
    monkeypatch.setenv("DM_DATABASE_URL", url)
    return url


@pytest.fixture
def scope(tmp_path) -> Path:
    target = tmp_path / "scope"
    shutil.copytree(EXAMPLE_SCOPE, target)
    return target


def invoke(scope: Path):
    return CliRunner().invoke(cli.app, ["sources", "--scope", str(scope)])


def test_first_run_inserts_then_rerun_unchanged(db_url, scope):
    first = invoke(scope)
    assert first.exit_code == 0, first.output
    assert first.stdout == "inserted 3 updated 0 unchanged 0\n"
    second = invoke(scope)
    assert second.exit_code == 0, second.output
    assert second.stdout == "inserted 0 updated 0 unchanged 3\n"


def test_changed_pack_updates(db_url, scope):
    assert invoke(scope).exit_code == 0
    path = scope / "sources.yaml"
    path.write_text(path.read_text().replace("licence: open (fictional)", "licence: changed"))
    result = invoke(scope)
    assert result.exit_code == 0, result.output
    assert result.stdout == "inserted 0 updated 1 unchanged 2\n"


def test_bad_pack_exits_2(db_url, scope):
    (scope / "sources.yaml").write_text("sources: [{id: 1}]\n")
    result = invoke(scope)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error:")


def test_db_failure_exits_2_without_url(db_url, scope, monkeypatch):
    def boom(engine, pack):
        raise OperationalError("stmt", {}, Exception(db_url))

    monkeypatch.setattr(cli, "seed_sources", boom)
    result = invoke(scope)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "error: database unavailable (OperationalError)\n"
    assert db_url not in result.output
