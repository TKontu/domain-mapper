"""LLM adapter: one interface over an OpenAI-compatible gateway and Anthropic (AD-01).

ARCHITECTURE.md §6 routes every model task through one interface: an
OpenAI-compatible client for the local vLLM gateway and an Anthropic client for
the cloud model. Both speak JSON over HTTP and expose the same
:meth:`LLM.complete` method.

* :class:`OpenAICompatibleLLM` posts to ``{base}/chat/completions``. Structured
  output rides on ``response_format`` (``json_schema``) or, for vLLM guided
  decoding, on the ``guided_json`` field.
* :class:`AnthropicLLM` posts to ``/v1/messages`` and forces a single tool whose
  ``input_schema`` is the caller's JSON schema.

Every call is wrapped in :func:`core.llm_calls.log_llm_call`, so exactly one
``llm_call`` row is written per call, on success and on failure. Prompt and
response text are never stored or logged, and no exception message contains
either: provider failures surface as a status code plus a canned reason.
Transport is an injectable callable; the default uses only the standard
library's :mod:`urllib.request`, so this adds no dependency.

The module is **unwired by design**: the first production callers are the
Screen and Assess stage rows (Phase 1), which will build it with
:func:`build_llm`.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from http.client import responses as _HTTP_REASONS
from typing import Any, Final, Literal, Protocol, cast

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.config import Settings
from core.llm_calls import LlmCallUsage, log_llm_call

DEFAULT_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_ANTHROPIC_BASE_URL: Final = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS: Final = 1024
ANTHROPIC_VERSION: Final = "2023-06-01"
# The forced tool's name; the response's ``tool_use`` block carries the object.
RESULT_TOOL_NAME: Final = "emit_result"

SchemaMode = Literal["response_format", "guided_json"]
DbTarget = Session | Engine


@dataclass(frozen=True)
class HttpResponse:
    """A minimal HTTP response: status and raw body, nothing else."""

    status: int
    body: bytes = b""


class Transport(Protocol):
    """Injected HTTP transport; the default is :func:`urllib_transport`."""

    def __call__(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse: ...


class LlmError(RuntimeError):
    """A failed model call; the message never contains prompt/response text."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LlmSchemaError(LlmError):
    """The model's structured output was missing or not a JSON object."""


class LlmBudgetExceeded(LlmError):
    """The run's accumulated LLM cost reached its budget; no further call was made."""


@dataclass(frozen=True)
class LlmResult:
    """One completed call; ``parsed`` is set only when a schema was given."""

    text: str
    parsed: dict[str, object] | None
    tokens_in: int
    tokens_out: int
    cost: float
    cache_hit: bool


class LLM(Protocol):
    """The single interface every backend implements (ARCHITECTURE.md §6)."""

    model: str

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult: ...


def urllib_transport(
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> HttpResponse:
    """Default transport: one stdlib POST, no redirects beyond urllib's own."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:
        # The error body is deliberately not read: it can echo the prompt.
        return HttpResponse(status=exc.code)
    except OSError as exc:
        # Only the exception type: provider/transport text is not trusted here.
        raise LlmError(f"transport failed: {type(exc).__name__}") from None


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_timeout(timeout: float) -> float:
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite number of seconds > 0")
    return float(timeout)


def _validate_rate(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return float(value)


def _require_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def _reason_for(status: int) -> str:
    return _HTTP_REASONS.get(status, "error")


def _raise_for_status(response: HttpResponse) -> None:
    if 200 <= response.status < 300:
        return
    # A canned reason from the status line; never the response body.
    raise LlmError(f"HTTP {response.status} {_reason_for(response.status)}", status=response.status)


def _decode_body(response: HttpResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body)
    except ValueError:
        raise LlmError("response body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise LlmError("response body is not a JSON object")
    return cast("dict[str, Any]", payload)


def _usage_int(payload: Mapping[str, Any], *keys: str) -> int:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise LlmError("response has no usage counters")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LlmError("response usage counters are malformed")
    return value


def _parse_structured(text: str, *, source: str) -> dict[str, object]:
    try:
        value = json.loads(text)
    except ValueError:
        raise LlmSchemaError(f"{source} is not valid JSON") from None
    if not isinstance(value, dict):
        raise LlmSchemaError(f"{source} is not a JSON object")
    return cast("dict[str, object]", value)


class _BaseHttpLLM:
    """Shared configuration and cost arithmetic for the two HTTP backends."""

    model: str

    def __init__(
        self,
        model: str,
        *,
        db: DbTarget,
        transport: Transport | None,
        timeout: float,
        cost_per_mtok_in: float,
        cost_per_mtok_out: float,
    ) -> None:
        self.model = _require_text(model, "model")
        self._db = db
        self.transport: Transport = urllib_transport if transport is None else transport
        self._timeout = _validate_timeout(timeout)
        self._cost_in = _validate_rate(cost_per_mtok_in, "cost_per_mtok_in")
        self._cost_out = _validate_rate(cost_per_mtok_out, "cost_per_mtok_out")

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1_000_000 * self._cost_in + tokens_out / 1_000_000 * self._cost_out

    def _post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        response = self.transport(url, headers=headers, body=body, timeout=self._timeout)
        _raise_for_status(response)
        return _decode_body(response)

    def _log(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
    ) -> AbstractContextManager[LlmCallUsage]:
        return log_llm_call(
            self._db,
            run_id=run_id,
            model=self.model,
            purpose=purpose,
            prompt=prompt,
        )


class OpenAICompatibleLLM(_BaseHttpLLM):
    """vLLM / OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        db: DbTarget,
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        schema_mode: SchemaMode = "response_format",
        cost_per_mtok_in: float = 0.0,
        cost_per_mtok_out: float = 0.0,
    ) -> None:
        super().__init__(
            model,
            db=db,
            transport=transport,
            timeout=timeout,
            cost_per_mtok_in=cost_per_mtok_in,
            cost_per_mtok_out=cost_per_mtok_out,
        )
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        if schema_mode not in ("response_format", "guided_json"):
            raise ValueError("schema_mode must be 'response_format' or 'guided_json'")
        self._schema_mode: SchemaMode = schema_mode
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None and self._api_key.strip():
            # The key travels only in this header; it is never logged.
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, prompt: str, json_schema: Mapping[str, object] | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_schema is not None:
            if self._schema_mode == "guided_json":
                payload["guided_json"] = dict(json_schema)
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": dict(json_schema),
                        "strict": True,
                    },
                }
        return payload

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        with self._log(prompt, purpose=purpose, run_id=run_id) as usage:
            payload = self._post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                payload=self._payload(prompt, json_schema),
            )
            tokens_in = _usage_int(payload, "usage", "prompt_tokens")
            tokens_out = _usage_int(payload, "usage", "completion_tokens")
            cost = self._cost(tokens_in, tokens_out)
            usage.tokens_in = tokens_in
            usage.tokens_out = tokens_out
            usage.cost = cost

            text = self._text(payload)
            parsed = (
                _parse_structured(text, source="model output") if json_schema is not None else None
            )
            return LlmResult(
                text=text,
                parsed=parsed,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                cache_hit=False,
            )

    @staticmethod
    def _text(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmError("response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LlmError("response choice is malformed")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise LlmError("response has no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise LlmError("response has no text content")
        return content


class AnthropicLLM(_BaseHttpLLM):
    """Anthropic Messages client; structured output uses a forced tool."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str,
        db: DbTarget,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cost_per_mtok_in: float = 0.0,
        cost_per_mtok_out: float = 0.0,
    ) -> None:
        super().__init__(
            model,
            db=db,
            transport=transport,
            timeout=timeout,
            cost_per_mtok_in=cost_per_mtok_in,
            cost_per_mtok_out=cost_per_mtok_out,
        )
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        self._api_key = _require_text(api_key, "api_key")
        self._max_tokens = _require_positive_int(max_tokens, "max_tokens")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _payload(self, prompt: str, json_schema: Mapping[str, object] | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_schema is not None:
            payload["tools"] = [
                {
                    "name": RESULT_TOOL_NAME,
                    "description": "Return the result as a JSON object.",
                    "input_schema": dict(json_schema),
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": RESULT_TOOL_NAME}
        return payload

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        with self._log(prompt, purpose=purpose, run_id=run_id) as usage:
            payload = self._post(
                f"{self._base_url}/v1/messages",
                headers=self._headers(),
                payload=self._payload(prompt, json_schema),
            )
            tokens_in = _usage_int(payload, "usage", "input_tokens")
            tokens_out = _usage_int(payload, "usage", "output_tokens")
            cost = self._cost(tokens_in, tokens_out)
            usage.tokens_in = tokens_in
            usage.tokens_out = tokens_out
            usage.cost = cost

            blocks = payload.get("content")
            if not isinstance(blocks, list):
                raise LlmError("response has no content blocks")
            text = self._text(blocks)
            parsed = self._tool_input(blocks) if json_schema is not None else None
            return LlmResult(
                text=text,
                parsed=parsed,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                cache_hit=False,
            )

    @staticmethod
    def _text(blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, Mapping):
                raise LlmError("response content block is malformed")
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(cast("str", block["text"]))
        return "\n".join(parts)

    @staticmethod
    def _tool_input(blocks: list[Any]) -> dict[str, object]:
        for block in blocks:
            if not isinstance(block, Mapping):
                raise LlmError("response content block is malformed")
            if block.get("type") == "tool_use":
                value = block.get("input")
                if isinstance(value, Mapping):
                    return dict(value)
                # The API returns parsed tool input; a string is raw JSON.
                if isinstance(value, str):
                    return _parse_structured(value, source="tool input")
        raise LlmSchemaError("response has no tool_use block")


class BudgetedLLM:
    """Wrap an :class:`LLM` and refuse calls once a run's spend reaches ``budget_usd``.

    ``spent`` sums :attr:`LlmResult.cost` of calls completed through this wrapper. Costs are
    known only after a call, so the last allowed call may overshoot the budget once. Zero-cost
    calls are never refused. Built by :func:`build_llm` (``budget_usd``) for ``landscape
    screen``; one instance per run.
    """

    def __init__(self, inner: LLM, budget_usd: float) -> None:
        self.inner = inner
        self.budget_usd = _validate_rate(budget_usd, "budget_usd")
        self.spent = 0.0
        self.model: str = inner.model

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        run_id: int,
        json_schema: Mapping[str, object] | None = None,
    ) -> LlmResult:
        if self.spent > 0 and self.spent >= self.budget_usd:
            # Numbers only: the prompt never reaches an exception message.
            raise LlmBudgetExceeded(
                f"run spent {self.spent:.4f} USD of budget {self.budget_usd:.4f} USD"
            )
        result = self.inner.complete(
            prompt, purpose=purpose, run_id=run_id, json_schema=json_schema
        )
        self.spent += result.cost
        return result


_LOCAL_BACKENDS: Final = frozenset({"local", "vllm", "openai", "openai-compatible"})
_CLOUD_BACKENDS: Final = frozenset({"cloud", "anthropic", "claude"})


def build_llm(
    settings: Settings,
    backend: str,
    *,
    db: DbTarget,
    transport: Transport | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    budget_usd: float | None = None,
) -> LLM:
    """Build the adapter named by ``backend`` from :class:`core.config.Settings`.

    ``local``/``vllm``/``openai`` select the OpenAI-compatible gateway at
    ``settings.llm_gateway_url`` with ``settings.local_model``.
    ``cloud``/``anthropic``/``claude`` select Anthropic with
    ``settings.cloud_llm_model`` and refuse to build without a key.
    With ``budget_usd`` the adapter is wrapped in :class:`BudgetedLLM` (one budget per run).
    """
    inner = _build_inner(
        settings,
        backend,
        db=db,
        transport=transport,
        timeout=timeout,
        anthropic_base_url=anthropic_base_url,
    )
    return inner if budget_usd is None else BudgetedLLM(inner, budget_usd)


def _build_inner(
    settings: Settings,
    backend: str,
    *,
    db: DbTarget,
    transport: Transport | None,
    timeout: float,
    anthropic_base_url: str,
) -> LLM:
    name = backend.strip().lower()
    if name in _LOCAL_BACKENDS:
        return OpenAICompatibleLLM(
            settings.llm_gateway_url,
            settings.local_model,
            db=db,
            api_key=settings.llm_gateway_api_key,
            transport=transport,
            timeout=timeout,
            cost_per_mtok_in=settings.local_cost_per_mtok_in,
            cost_per_mtok_out=settings.local_cost_per_mtok_out,
        )
    if name in _CLOUD_BACKENDS:
        key = settings.cloud_api_key
        if key is None or not key.strip():
            raise ValueError("cloud LLM backend needs DM_CLOUD_LLM_API_KEY")
        return AnthropicLLM(
            anthropic_base_url,
            settings.cloud_llm_model,
            api_key=key,
            db=db,
            transport=transport,
            timeout=timeout,
            cost_per_mtok_in=settings.cloud_cost_per_mtok_in,
            cost_per_mtok_out=settings.cloud_cost_per_mtok_out,
        )
    allowed = ", ".join(sorted(_LOCAL_BACKENDS | _CLOUD_BACKENDS))
    raise ValueError(f"unknown LLM backend {backend!r}; expected one of {allowed}")
