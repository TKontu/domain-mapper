"""Tests for core.screen (SC-01): one-entity relevance screen via the LLM protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from core.screen import (
    BORDERLINE_HIGH,
    BORDERLINE_LOW,
    ScreenError,
    ScreenResult,
    screen_entity,
)

from adapters.llm import LlmError, LlmResult
from core.scope_pack import ScopePack, load_scope_pack

EXAMPLE_SCOPE = Path(__file__).resolve().parent.parent / "scopes" / "_example"
SECRET_TEXT = "PAGE-TEXT-MARKER-xyz"


class FakeLLM:
    model = "fake-small"

    def __init__(self, parsed: object = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        self.calls.append(
            {"prompt": prompt, "purpose": purpose, "run_id": run_id, "schema": json_schema}
        )
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


@pytest.fixture(scope="module")
def scope() -> ScopePack:
    return load_scope_pack(EXAMPLE_SCOPE)


def ids(scope: ScopePack) -> list[str]:
    return [s.id for s in scope.scope.segments]


def run(llm: FakeLLM, scope: ScopePack, text: str = SECRET_TEXT, **kw: object) -> ScreenResult:
    return screen_entity(
        llm,
        scope,
        name="Acme",
        domain="acme.example",
        text=text,
        run_id=7,
        **kw,  # type: ignore[arg-type]
    )


def valid(scope: ScopePack, **over: object) -> dict[str, object]:
    out: dict[str, object] = {
        "in_scope": True,
        "segments": list(reversed(ids(scope))) + [ids(scope)[0]],
        "confidence": 0.9,
        "reason": "  makes the thing  ",
    }
    out.update(over)
    return out


def test_valid_output(scope: ScopePack) -> None:
    llm = FakeLLM(valid(scope))
    r = run(llm, scope)
    assert r == ScreenResult(
        in_scope=True,
        segments=tuple(ids(scope)),
        confidence=0.9,
        reason="makes the thing",
        borderline=False,
    )
    call = llm.calls[0]
    assert call["purpose"] == "screen"
    assert call["run_id"] == 7
    schema = call["schema"]
    assert isinstance(schema, Mapping)
    assert schema["properties"]["segments"]["items"]["enum"] == ids(scope)  # type: ignore[index]
    assert set(schema["required"]) >= {"in_scope", "segments", "confidence", "reason"}  # type: ignore[arg-type]
    prompt = str(call["prompt"])
    assert scope.scope.title in prompt and scope.scope.perspective in prompt
    for seg in scope.scope.segments:
        assert seg.id in prompt and seg.label in prompt and seg.definition in prompt
    assert "Acme" in prompt and "acme.example" in prompt and SECRET_TEXT in prompt


def test_truncates_text(scope: ScopePack) -> None:
    llm = FakeLLM(valid(scope))
    run(llm, scope, text="a" * 50 + "TAIL", max_chars=50)
    prompt = str(llm.calls[0]["prompt"])
    assert "a" * 50 in prompt and "TAIL" not in prompt


def test_max_chars_validated(scope: ScopePack) -> None:
    with pytest.raises(ValueError):
        run(FakeLLM(valid(scope)), scope, max_chars=0)


@pytest.mark.parametrize(
    "bad",
    [
        {"segments": ["nope"]},
        {"segments": "x"},
        {"segments": [1]},
        {"in_scope": "yes"},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"confidence": "0.5"},
        {"confidence": True},
        {"reason": "   "},
        {"reason": 3},
    ],
)
def test_invalid_output(scope: ScopePack, bad: dict[str, object]) -> None:
    with pytest.raises(ScreenError) as e:
        run(FakeLLM(valid(scope, **bad)), scope)
    assert SECRET_TEXT not in str(e.value)
    assert next(iter(bad)) in str(e.value)


@pytest.mark.parametrize("key", ["in_scope", "segments", "confidence", "reason"])
def test_missing_key(scope: ScopePack, key: str) -> None:
    out = valid(scope)
    del out[key]
    with pytest.raises(ScreenError, match=key):
        run(FakeLLM(out), scope)


def test_missing_parsed(scope: ScopePack) -> None:
    with pytest.raises(ScreenError):
        run(FakeLLM(None), scope)


def test_screen_error_is_value_error() -> None:
    assert issubclass(ScreenError, ValueError)


def test_long_reason_truncated(scope: ScopePack) -> None:
    r = run(FakeLLM(valid(scope, reason="r" * 900)), scope)
    assert len(r.reason) == 500


def test_out_of_scope_drops_segments(scope: ScopePack) -> None:
    r = run(FakeLLM(valid(scope, in_scope=False, confidence=0.95)), scope)
    assert r.segments == () and not r.in_scope and r.reason == "makes the thing"
    assert not r.borderline


@pytest.mark.parametrize(
    ("conf", "border"),
    [
        (0.3499, False),
        (BORDERLINE_LOW, True),
        (0.5, True),
        (0.6499, True),
        (BORDERLINE_HIGH, False),
    ],
)
def test_borderline_bounds(scope: ScopePack, conf: float, border: bool) -> None:
    assert (BORDERLINE_LOW, BORDERLINE_HIGH) == (0.35, 0.65)
    assert run(FakeLLM(valid(scope, confidence=conf)), scope).borderline is border


def test_in_scope_without_segments_is_borderline(scope: ScopePack) -> None:
    r = run(FakeLLM(valid(scope, segments=[], confidence=0.99)), scope)
    assert r.borderline and r.segments == ()


def test_llm_error_propagates(scope: ScopePack) -> None:
    err = LlmError("boom")
    with pytest.raises(LlmError) as e:
        run(FakeLLM(error=err), scope)
    assert e.value is err
