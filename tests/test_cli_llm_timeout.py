"""LL-02: the CLI LLM seams pass ``settings.llm_timeout_s`` to ``build_llm``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

from app import cli
from core.config import Settings

SEAMS: list[Callable[[Settings, Any], Any]] = [
    cli._make_screen_llm,
    cli._make_assess_llm,
    cli._make_verify_llm,
]


@pytest.mark.parametrize("seam", SEAMS, ids=lambda f: f.__name__)
def test_seam_passes_the_timeout_setting(seam: Callable[[Settings, Any], Any]) -> None:
    settings = Settings(llm_timeout_s=900.0, verifier_model="fake-verifier")
    sentinel = object()
    with mock.patch.object(cli, "build_llm", return_value=sentinel) as spy:
        assert seam(settings, "sqlite://") is sentinel
    spy.assert_called_once()
    assert spy.call_args.kwargs["timeout"] == 900.0
    assert spy.call_args.kwargs["budget_usd"] == settings.llm_budget_usd
