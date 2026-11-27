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


@pytest.mark.parametrize(
    "seam", [cli._make_screen_llm, cli._make_assess_llm], ids=lambda f: f.__name__
)
def test_screen_and_assess_pass_the_local_cap(seam: Callable[[Settings, Any], Any]) -> None:
    """LM-01: local seams cap output with ``settings.local_max_tokens``."""
    settings = Settings(local_max_tokens=512, verifier_max_tokens=64)
    with mock.patch.object(cli, "build_llm") as spy:
        seam(settings, "sqlite://")
    assert spy.call_args.kwargs["max_tokens"] == 512


@pytest.mark.parametrize(
    ("local_cap", "verifier_cap", "expected"),
    [(512, 64, 64), (512, None, 512), (None, None, None)],
)
def test_verify_prefers_the_verifier_cap(
    local_cap: int | None, verifier_cap: int | None, expected: int | None
) -> None:
    """LM-01: verify uses ``verifier_max_tokens``, else falls back to the local cap."""
    settings = Settings(
        verifier_model="fake-verifier",
        local_max_tokens=local_cap,
        verifier_max_tokens=verifier_cap,
    )
    with mock.patch.object(cli, "build_llm") as spy:
        cli._make_verify_llm(settings, "sqlite://")
    assert spy.call_args.kwargs["max_tokens"] == expected
