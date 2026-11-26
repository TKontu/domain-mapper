"""The ``landscape`` CLI configures structured logging once per invocation (RT-02)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from app.cli import app
from core.log import StructuredFormatter

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def _structured_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if isinstance(h.formatter, StructuredFormatter)]


def test_command_installs_one_structured_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DM_LOG_LEVEL", raising=False)
    result = runner.invoke(app, ["sources", "--help"])
    assert result.exit_code == 0
    assert len(_structured_handlers()) == 1
    assert logging.getLogger().level == logging.INFO


def test_repeated_invocation_does_not_duplicate_handlers() -> None:
    runner.invoke(app, ["sources", "--help"])
    runner.invoke(app, ["sources", "--help"])
    assert len(_structured_handlers()) == 1


def test_log_level_option_sets_root_level() -> None:
    result = runner.invoke(app, ["--log-level", "DEBUG", "sources", "--help"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.DEBUG


def test_env_default_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_LOG_LEVEL", "WARNING")
    result = runner.invoke(app, ["sources", "--help"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.WARNING


def test_invalid_level_exits_2() -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "sources", "--help"])
    assert result.exit_code == 2
    assert "error: invalid log level LOUD" in result.output
    assert _structured_handlers() == []
