"""Tests for the per-run LLM spend cap (LB-01); fake inner LLMs, no network."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
import sqlalchemy as sa

from adapters.llm import (
    BudgetedLLM,
    LlmBudgetExceeded,
    LlmError,
    LlmResult,
    OpenAICompatibleLLM,
    build_llm,
)
from core.config import Settings

PROMPT = "classify the neutral widget page"


class CostLLM:
    model = "fake-cost"

    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        self.calls += 1
        return LlmResult(
            text="{}", parsed=None, tokens_in=1, tokens_out=1, cost=self.cost, cache_hit=False
        )


def call(llm: BudgetedLLM) -> LlmResult:
    return llm.complete(PROMPT, purpose="screen", run_id=1)


def test_refuses_once_spent_reaches_budget_without_calling_inner() -> None:
    inner = CostLLM(0.4)
    llm = BudgetedLLM(inner, 1.0)
    for _ in range(3):
        call(llm)
    assert inner.calls == 3
    assert llm.spent == pytest.approx(1.2)
    with pytest.raises(LlmBudgetExceeded) as info:
        call(llm)
    assert inner.calls == 3
    assert isinstance(info.value, LlmError)
    message = str(info.value)
    assert "1.2" in message and "1.0" in message
    assert PROMPT not in message and "widget" not in message


def test_zero_cost_calls_are_never_refused_even_at_zero_budget() -> None:
    inner = CostLLM(0.0)
    llm = BudgetedLLM(inner, 0.0)
    for _ in range(50):
        call(llm)
    assert inner.calls == 50


def test_model_is_the_inner_model() -> None:
    assert BudgetedLLM(CostLLM(0.0), 1.0).model == "fake-cost"


@pytest.mark.parametrize("budget", [-0.01, math.nan, math.inf])
def test_invalid_budget_is_rejected(budget: float) -> None:
    with pytest.raises(ValueError):
        BudgetedLLM(CostLLM(0.0), budget)


def test_build_llm_wraps_only_when_budget_given() -> None:
    engine = sa.create_engine("sqlite://")
    settings = Settings()
    plain = build_llm(settings, "local", db=engine)
    assert isinstance(plain, OpenAICompatibleLLM)
    budgeted = build_llm(settings, "local", db=engine, budget_usd=2.5)
    assert isinstance(budgeted, BudgetedLLM)
    assert budgeted.budget_usd == 2.5
    assert isinstance(budgeted.inner, OpenAICompatibleLLM)
