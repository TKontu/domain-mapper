"""Tests for the per-run LLM spend cap (LB-01); fake inner LLMs, no network."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.llm import (
    AnthropicLLM,
    BudgetedLLM,
    HttpResponse,
    LlmBudgetExceeded,
    LlmError,
    LlmResult,
    OpenAICompatibleLLM,
    build_llm,
)
from core.config import Settings
from core.llm_calls import LlmCall
from core.runs import Run, create_all

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


class FakeTransport:
    """Records each request and returns queued responses in order."""

    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append({"url": url, "headers": dict(headers), "body": body, "timeout": timeout})
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


def openai_response(tokens_in: int, tokens_out: int) -> HttpResponse:
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "{}"}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }
    return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"))


def anthropic_response(tokens_in: int, tokens_out: int) -> HttpResponse:
    payload = {
        "content": [{"type": "text", "text": "{}"}],
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }
    return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"))


@pytest.fixture()
def engine() -> Iterator[Engine]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(Run(stage="screen"))
    yield engine
    engine.dispose()


def call_rows(engine: Engine) -> list[LlmCall]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(LlmCall).order_by(LlmCall.id)).all())


def test_local_rates_reach_the_result_and_the_llm_call_row(engine: Engine) -> None:
    transport = FakeTransport(openai_response(1_000_000, 500_000))
    settings = Settings(local_cost_per_mtok_in=2.0, local_cost_per_mtok_out=4.0)

    llm = build_llm(settings, "local", db=engine, transport=transport)
    result = llm.complete(PROMPT, purpose="screen", run_id=1)

    assert result.cost == pytest.approx(4.0)
    rows = call_rows(engine)
    assert len(rows) == 1
    assert rows[0].tokens_in == 1_000_000
    assert rows[0].tokens_out == 500_000
    assert rows[0].cost == pytest.approx(4.0)


def test_cloud_uses_its_own_rate_pair_not_the_local_one(engine: Engine) -> None:
    transport = FakeTransport(anthropic_response(1_000_000, 500_000))
    settings = Settings(
        cloud_api_key="test-key",
        local_cost_per_mtok_in=9.0,
        local_cost_per_mtok_out=9.0,
        cloud_cost_per_mtok_in=3.0,
        cloud_cost_per_mtok_out=1.0,
    )

    llm = build_llm(settings, "cloud", db=engine, transport=transport)
    assert isinstance(llm, AnthropicLLM)
    result = llm.complete(PROMPT, purpose="screen", run_id=1)

    assert result.cost == pytest.approx(3.5)
    assert call_rows(engine)[0].cost == pytest.approx(3.5)


def test_budget_refuses_the_second_call_now_that_cost_is_known(engine: Engine) -> None:
    # 2M in / 1M out at 2.0/4.0 USD per million tokens is 8.0 USD for one call,
    # so a 5.0 USD budget is already spent when the second call is attempted.
    transport = FakeTransport(openai_response(2_000_000, 1_000_000))
    settings = Settings(local_cost_per_mtok_in=2.0, local_cost_per_mtok_out=4.0)

    llm = build_llm(settings, "local", db=engine, transport=transport, budget_usd=5.0)
    assert isinstance(llm, BudgetedLLM)
    first = llm.complete(PROMPT, purpose="screen", run_id=1)
    assert first.cost == pytest.approx(8.0)
    assert llm.spent == pytest.approx(8.0)

    with pytest.raises(LlmBudgetExceeded):
        llm.complete(PROMPT, purpose="screen", run_id=1)

    assert len(transport.calls) == 1
    assert len(call_rows(engine)) == 1


def test_concurrent_calls_sum_spent_exactly() -> None:  # LM-04
    import threading
    import time

    class SlowCost(CostLLM):
        def complete(self, prompt: str, **kwargs: Any) -> LlmResult:  # type: ignore[override]
            time.sleep(0.001)
            return super().complete(prompt, **kwargs)

    wrapper = BudgetedLLM(SlowCost(0.001), budget_usd=1000.0)
    threads = [
        threading.Thread(
            target=lambda: [wrapper.complete(PROMPT, purpose="screen", run_id=1) for _ in range(50)]
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert math.isclose(wrapper.spent, 400 * 0.001)
    assert hasattr(wrapper, "_lock")
