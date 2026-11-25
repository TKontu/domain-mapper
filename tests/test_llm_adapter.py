"""Tests for the ``LLM`` adapter (AD-01).

Both backends run against a fake transport and an in-memory SQLite database, so
the suite is offline: no model endpoint is contacted and no key is real.
"""

from __future__ import annotations

import io
import json
import logging
import math
import unittest
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adapters.llm import (
    DEFAULT_TIMEOUT_SECONDS,
    AnthropicLLM,
    HttpResponse,
    LlmError,
    LlmSchemaError,
    OpenAICompatibleLLM,
    build_llm,
    urllib_transport,
)
from core.config import Settings
from core.llm_calls import ERROR, OK, LlmCall, hash_prompt, log_llm_call
from core.log import configure_logging
from core.runs import Base, Run, create_all

PROMPT = "Private prompt text that must never be stored or logged"
RESPONSE_BODY = "Private model text that must never reach an error"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"status": {"type": "string"}},
    "required": ["status"],
}


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


@dataclass
class Call:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float


class FakeTransport:
    """Records each request and returns queued responses in order."""

    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[Call] = []

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append(Call(url=url, headers=dict(headers), body=body, timeout=timeout))
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


def json_response(payload: dict[str, Any]) -> HttpResponse:
    return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"))


def openai_payload(
    content: str,
    *,
    tokens_in: int = 11,
    tokens_out: int = 7,
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


def anthropic_payload(
    content: list[dict[str, Any]],
    *,
    tokens_in: int = 13,
    tokens_out: int = 5,
) -> dict[str, Any]:
    return {
        "content": content,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


class LlmAdapterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = memory_engine()
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)
        with Session(self.engine) as session, session.begin():
            run = Run(stage="assess")
            session.add(run)
            session.flush()
            self.run_id = run.id

    def rows(self) -> list[LlmCall]:
        with Session(self.engine) as session:
            return list(session.scalars(sa.select(LlmCall).order_by(LlmCall.id)).all())

    def configure_logging_for_test(self) -> io.StringIO:
        stream = io.StringIO()
        handler = configure_logging("INFO", stream=stream)

        def cleanup() -> None:
            logging.getLogger().removeHandler(handler)
            handler.close()

        self.addCleanup(cleanup)
        return stream

    def openai(self, transport: FakeTransport, **kwargs: Any) -> OpenAICompatibleLLM:
        return OpenAICompatibleLLM(
            "http://gateway:8000/v1",
            "local-small",
            db=self.engine,
            transport=transport,
            **kwargs,
        )

    def anthropic(self, transport: FakeTransport, **kwargs: Any) -> AnthropicLLM:
        return AnthropicLLM(
            "https://api.anthropic.test",
            "cloud-large",
            api_key="sk-ant-secret",
            db=self.engine,
            transport=transport,
            **kwargs,
        )


class OpenAITests(LlmAdapterTestCase):
    def test_sends_url_headers_and_parses_text_and_usage(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hello")))
        llm = self.openai(transport, api_key="sk-local-secret")

        result = llm.complete(PROMPT, purpose="screen", run_id=self.run_id)

        self.assertEqual(len(transport.calls), 1)
        call = transport.calls[0]
        self.assertEqual(call.url, "http://gateway:8000/v1/chat/completions")
        self.assertEqual(call.headers["Authorization"], "Bearer sk-local-secret")
        self.assertEqual(call.headers["Content-Type"], "application/json")
        body = json.loads(call.body)
        self.assertEqual(body["model"], "local-small")
        self.assertEqual(body["messages"], [{"role": "user", "content": PROMPT}])
        self.assertNotIn("response_format", body)
        self.assertNotIn("guided_json", body)
        self.assertNotIn("sk-local-secret", call.body.decode("utf-8"))

        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.parsed)
        self.assertEqual(result.tokens_in, 11)
        self.assertEqual(result.tokens_out, 7)
        self.assertFalse(result.cache_hit)

    def test_no_authorization_header_without_a_key(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hi")))
        self.openai(transport).complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertNotIn("Authorization", transport.calls[0].headers)

    def test_schema_uses_response_format_json_schema(self) -> None:
        transport = FakeTransport(
            json_response(
                openai_payload('{"status": "in"}'),
            )
        )
        llm = self.openai(transport)

        result = llm.complete(PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA)

        body = json.loads(transport.calls[0].body)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertIs(body["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(result.parsed, {"status": "in"})
        self.assertEqual(result.text, '{"status": "in"}')

    def test_schema_can_use_vllm_guided_json(self) -> None:
        transport = FakeTransport(
            json_response(
                openai_payload('{"status": "in"}'),
            )
        )
        llm = self.openai(transport, schema_mode="guided_json")

        result = llm.complete(PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA)

        body = json.loads(transport.calls[0].body)
        self.assertEqual(body["guided_json"], SCHEMA)
        self.assertNotIn("response_format", body)
        self.assertEqual(result.parsed, {"status": "in"})

    def test_invalid_json_with_a_schema_raises_typed_error(self) -> None:
        transport = FakeTransport(json_response(openai_payload("not json")))
        llm = self.openai(transport)

        with self.assertRaises(LlmSchemaError):
            llm.complete(PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, ERROR)
        self.assertEqual(rows[0].tokens_in, 11)
        self.assertEqual(rows[0].tokens_out, 7)

    def test_non_object_json_with_a_schema_raises_typed_error(self) -> None:
        transport = FakeTransport(json_response(openai_payload("[1, 2]")))
        llm = self.openai(transport)

        with self.assertRaises(LlmSchemaError):
            llm.complete(PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA)

    def test_missing_content_raises_prompt_free_error(self) -> None:
        payload = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        transport = FakeTransport(json_response(payload))

        with self.assertRaises(LlmError) as caught:
            self.openai(transport).complete(PROMPT, purpose="screen", run_id=self.run_id)

        self.assertNotIn(PROMPT, str(caught.exception))
        self.assertNotIn("None", str(caught.exception))

    def test_http_failure_raises_status_and_writes_one_error_row(self) -> None:
        stream = self.configure_logging_for_test()
        transport = FakeTransport(HttpResponse(status=500, body=RESPONSE_BODY.encode("utf-8")))
        llm = self.openai(transport, api_key="sk-local-secret")

        with self.assertRaises(LlmError) as caught:
            llm.complete(PROMPT, purpose="screen", run_id=self.run_id)

        error = str(caught.exception)
        self.assertIn("500", error)
        self.assertNotIn(PROMPT, error)
        self.assertNotIn(RESPONSE_BODY, error)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, ERROR)
        self.assertIsNotNone(rows[0].error)
        assert rows[0].error is not None
        self.assertIn("500", rows[0].error)
        self.assertNotIn(PROMPT, rows[0].error)
        self.assertNotIn(RESPONSE_BODY, rows[0].error)
        output = stream.getvalue()
        self.assertNotIn(PROMPT, output)
        self.assertNotIn("sk-local-secret", output)

    def test_success_writes_one_row_with_run_usage_and_latency(self) -> None:
        stream = self.configure_logging_for_test()
        transport = FakeTransport(json_response(openai_payload("hello")))
        llm = self.openai(transport, api_key="sk-local-secret")

        llm.complete(PROMPT, purpose="screen", run_id=self.run_id)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.run_id, self.run_id)
        self.assertEqual(row.model, "local-small")
        self.assertEqual(row.purpose, "screen")
        self.assertEqual(row.prompt_hash, hash_prompt(PROMPT))
        self.assertEqual(row.tokens_in, 11)
        self.assertEqual(row.tokens_out, 7)
        self.assertGreaterEqual(row.latency_ms, 0)
        self.assertEqual(row.status, OK)
        self.assertIsNone(row.error)
        self.assertFalse(row.cache_hit)
        output = stream.getvalue()
        self.assertNotIn(PROMPT, output)
        self.assertNotIn("sk-local-secret", output)
        self.assertIn("prompt_hash=" + hash_prompt(PROMPT), output)

    def test_prompt_absent_from_every_column(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hello")))
        self.openai(transport).complete(PROMPT, purpose="screen", run_id=self.run_id)
        row = self.rows()[0]
        for column in LlmCall.__table__.columns:
            self.assertNotIn(PROMPT, str(getattr(row, column.key)), column.key)

    def test_usage_computes_cost_from_optional_rates(self) -> None:
        transport = FakeTransport(
            json_response(openai_payload("hi", tokens_in=1000, tokens_out=500))
        )
        llm = self.openai(transport, cost_per_mtok_in=2.0, cost_per_mtok_out=8.0)

        result = llm.complete(PROMPT, purpose="screen", run_id=self.run_id)

        self.assertAlmostEqual(result.cost, 0.002 + 0.004)
        self.assertAlmostEqual(self.rows()[0].cost, 0.006)

    def test_cost_defaults_to_zero(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hi")))
        result = self.openai(transport).complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(result.cost, 0.0)

    def test_two_calls_write_two_rows(self) -> None:
        transport = FakeTransport(
            json_response(openai_payload("one")), json_response(openai_payload("two"))
        )
        llm = self.openai(transport)
        for _ in range(2):
            llm.complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(len(self.rows()), 2)

    def test_timeout_is_passed_to_the_transport(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hi")))
        llm = self.openai(transport, timeout=7.5)
        llm.complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(transport.calls[0].timeout, 7.5)

    def test_trailing_slash_in_base_url_is_handled(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hi")))
        llm = OpenAICompatibleLLM(
            "http://gateway:8000/v1/", "local-small", db=self.engine, transport=transport
        )
        llm.complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(transport.calls[0].url, "http://gateway:8000/v1/chat/completions")

    def test_invalid_timeout_is_rejected(self) -> None:
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                self.openai(FakeTransport(), timeout=bad)

    def test_invalid_schema_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.openai(FakeTransport(), schema_mode="yaml")

    def test_empty_base_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleLLM(" ", "local-small", db=self.engine, transport=FakeTransport())

    def test_default_timeout_is_finite(self) -> None:
        self.assertTrue(math.isfinite(DEFAULT_TIMEOUT_SECONDS))
        self.assertGreater(DEFAULT_TIMEOUT_SECONDS, 0)


class AnthropicTests(LlmAdapterTestCase):
    def test_sends_url_headers_and_parses_text_and_usage(self) -> None:
        transport = FakeTransport(
            json_response(
                anthropic_payload(
                    [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ]
                )
            )
        )
        llm = self.anthropic(transport)

        result = llm.complete(PROMPT, purpose="profile", run_id=self.run_id)

        call = transport.calls[0]
        self.assertEqual(call.url, "https://api.anthropic.test/v1/messages")
        self.assertEqual(call.headers["x-api-key"], "sk-ant-secret")
        self.assertEqual(call.headers["anthropic-version"], "2023-06-01")
        self.assertEqual(call.headers["Content-Type"], "application/json")
        self.assertNotIn("Authorization", call.headers)
        body = json.loads(call.body)
        self.assertEqual(body["model"], "cloud-large")
        self.assertEqual(body["messages"], [{"role": "user", "content": PROMPT}])
        self.assertGreater(body["max_tokens"], 0)
        self.assertNotIn("tools", body)
        self.assertNotIn("sk-ant-secret", call.body.decode("utf-8"))

        self.assertEqual(result.text, "first\nsecond")
        self.assertIsNone(result.parsed)
        self.assertEqual(result.tokens_in, 13)
        self.assertEqual(result.tokens_out, 5)
        self.assertFalse(result.cache_hit)

    def test_schema_uses_a_forced_tool(self) -> None:
        transport = FakeTransport(
            json_response(
                anthropic_payload(
                    [
                        {"type": "text", "text": "ok"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "emit_result",
                            "input": {"status": "in"},
                        },
                    ]
                )
            )
        )
        llm = self.anthropic(transport)

        result = llm.complete(PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA)

        body = json.loads(transport.calls[0].body)
        self.assertEqual(body["tools"][0]["name"], "emit_result")
        self.assertEqual(body["tools"][0]["input_schema"], SCHEMA)
        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "emit_result"})
        self.assertEqual(result.parsed, {"status": "in"})
        self.assertEqual(result.text, "ok")

    def test_schema_without_a_tool_use_block_raises_typed_error(self) -> None:
        transport = FakeTransport(
            json_response(anthropic_payload([{"type": "text", "text": "no tool"}]))
        )

        with self.assertRaises(LlmSchemaError) as caught:
            self.anthropic(transport).complete(
                PROMPT, purpose="classify", run_id=self.run_id, json_schema=SCHEMA
            )
        self.assertNotIn(PROMPT, str(caught.exception))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0].status, ERROR)

    def test_http_failure_is_prompt_free_and_logs_one_error_row(self) -> None:
        stream = self.configure_logging_for_test()
        transport = FakeTransport(HttpResponse(status=429, body=RESPONSE_BODY.encode("utf-8")))

        with self.assertRaises(LlmError) as caught:
            self.anthropic(transport).complete(PROMPT, purpose="profile", run_id=self.run_id)

        error = str(caught.exception)
        self.assertIn("429", error)
        self.assertNotIn(PROMPT, error)
        self.assertNotIn(RESPONSE_BODY, error)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, ERROR)
        assert rows[0].error is not None
        self.assertNotIn(PROMPT, rows[0].error)
        self.assertNotIn(RESPONSE_BODY, rows[0].error)
        self.assertNotIn(PROMPT, stream.getvalue())
        self.assertNotIn("sk-ant-secret", stream.getvalue())

    def test_success_writes_one_row_with_usage(self) -> None:
        transport = FakeTransport(
            json_response(anthropic_payload([{"type": "text", "text": "hello"}]))
        )
        self.anthropic(transport).complete(PROMPT, purpose="profile", run_id=self.run_id)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model, "cloud-large")
        self.assertEqual(rows[0].tokens_in, 13)
        self.assertEqual(rows[0].tokens_out, 5)
        self.assertEqual(rows[0].status, OK)
        self.assertEqual(rows[0].prompt_hash, hash_prompt(PROMPT))

    def test_missing_api_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnthropicLLM("https://api.anthropic.test", "cloud-large", api_key="", db=self.engine)

    def test_invalid_timeout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.anthropic(FakeTransport(), timeout=0.0)

    def test_invalid_max_tokens_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.anthropic(FakeTransport(), max_tokens=0)

    def test_malformed_json_body_raises_prompt_free_error(self) -> None:
        transport = FakeTransport(HttpResponse(status=200, body=RESPONSE_BODY.encode("utf-8")))
        with self.assertRaises(LlmError) as caught:
            self.anthropic(transport).complete(PROMPT, purpose="profile", run_id=self.run_id)
        self.assertNotIn(RESPONSE_BODY, str(caught.exception))
        self.assertNotIn(PROMPT, str(caught.exception))


class FactoryTests(LlmAdapterTestCase):
    def settings(self, **kwargs: Any) -> Settings:
        values: dict[str, Any] = {
            "llm_gateway_url": "http://gateway:8000/v1",
            "local_model": "local-small",
            "cloud_llm_model": "cloud-large",
        }
        values.update(kwargs)
        return Settings(**values)

    def test_local_backend_uses_gateway_and_local_model(self) -> None:
        transport = FakeTransport(json_response(openai_payload("hi")))
        llm = build_llm(self.settings(), "local", db=self.engine, transport=transport)
        self.assertIsInstance(llm, OpenAICompatibleLLM)
        self.assertEqual(llm.model, "local-small")
        llm.complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(transport.calls[0].url, "http://gateway:8000/v1/chat/completions")

    def test_vllm_and_openai_aliases_select_the_compatible_client(self) -> None:
        for backend in ("vllm", "openai", "openai-compatible"):
            with self.subTest(backend=backend):
                llm = build_llm(self.settings(), backend, db=self.engine, transport=FakeTransport())
                self.assertIsInstance(llm, OpenAICompatibleLLM)

    def test_cloud_backend_uses_anthropic_with_the_cloud_model(self) -> None:
        transport = FakeTransport(
            json_response(anthropic_payload([{"type": "text", "text": "hi"}]))
        )
        llm = build_llm(
            self.settings(cloud_api_key="sk-cloud"),
            "cloud",
            db=self.engine,
            transport=transport,
        )
        self.assertIsInstance(llm, AnthropicLLM)
        self.assertEqual(llm.model, "cloud-large")
        llm.complete(PROMPT, purpose="profile", run_id=self.run_id)
        self.assertEqual(transport.calls[0].url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(transport.calls[0].headers["x-api-key"], "sk-cloud")

    def test_anthropic_backend_alias(self) -> None:
        llm = build_llm(
            self.settings(cloud_api_key="sk-cloud"),
            "anthropic",
            db=self.engine,
            transport=FakeTransport(),
        )
        self.assertIsInstance(llm, AnthropicLLM)

    def test_cloud_backend_without_a_key_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_llm(self.settings(), "cloud", db=self.engine, transport=FakeTransport())
        self.assertIn("DM_CLOUD_LLM_API_KEY", str(caught.exception))

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_llm(self.settings(), "magic", db=self.engine, transport=FakeTransport())


class DefaultTransportTests(unittest.TestCase):
    def test_default_transport_is_urllib(self) -> None:
        self.assertTrue(callable(urllib_transport))

    def test_transport_choice_defaults_to_urllib(self) -> None:
        engine = memory_engine()
        self.addCleanup(engine.dispose)
        llm = OpenAICompatibleLLM("http://gateway:8000/v1", "local-small", db=engine)
        self.assertIs(llm.transport, urllib_transport)


class LoggingBoundaryTests(LlmAdapterTestCase):
    def test_error_row_uses_the_context_helper_once(self) -> None:
        # The adapter must not write its own row on top of ``log_llm_call``.
        transport = FakeTransport(HttpResponse(status=401))
        with self.assertRaises(LlmError):
            self.openai(transport).complete(PROMPT, purpose="screen", run_id=self.run_id)
        self.assertEqual(len(self.rows()), 1)

    def test_manual_helper_and_adapter_are_distinct_paths(self) -> None:
        # Guards against the adapter bypassing ``log_llm_call``: a direct helper
        # call still produces exactly its own row.
        with log_llm_call(self.engine, run_id=self.run_id, model="x", purpose="test", prompt="p"):
            pass
        self.assertEqual(len(self.rows()), 1)


if __name__ == "__main__":
    unittest.main()
