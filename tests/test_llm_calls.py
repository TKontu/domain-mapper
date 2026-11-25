"""Tests for per-call LLM cost/latency logging (RT-02).

The suite runs offline against in-memory SQLite, on the same declarative ``Base``
as the run-tracking tables. No model endpoint is contacted.
"""

from __future__ import annotations

import io
import logging
import unittest
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from core.llm_calls import (
    ERROR,
    OK,
    LlmCall,
    hash_prompt,
    log_llm_call,
    record_llm_call,
)
from core.log import configure_logging
from core.runs import Base, Run, create_all

PROMPT = "Private prompt text that must never be stored or logged"


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


class LlmCallTestCase(unittest.TestCase):
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

    def test_record_llm_call_writes_one_row(self) -> None:
        call_id = record_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-small",
            purpose="screen",
            prompt=PROMPT,
            tokens_in=12,
            tokens_out=4,
            cost=0.0012,
            latency_ms=42,
            cache_hit=True,
        )

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.id, call_id)
        self.assertEqual(row.run_id, self.run_id)
        self.assertEqual(row.model, "local-small")
        self.assertEqual(row.purpose, "screen")
        self.assertEqual(row.prompt_hash, hash_prompt(PROMPT))
        self.assertEqual(len(row.prompt_hash), 64)
        self.assertEqual(row.tokens_in, 12)
        self.assertEqual(row.tokens_out, 4)
        self.assertAlmostEqual(row.cost, 0.0012)
        self.assertGreaterEqual(row.latency_ms, 0)
        self.assertEqual(row.latency_ms, 42)
        self.assertTrue(row.cache_hit)
        self.assertEqual(row.status, OK)
        self.assertIsNone(row.error)
        self.assertIsInstance(row.created_at, datetime)

    def test_prompt_text_is_absent_from_row_and_logs(self) -> None:
        stream = self.configure_logging_for_test()
        record_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-small",
            purpose="screen",
            prompt=PROMPT,
            tokens_in=3,
            tokens_out=1,
        )

        row = self.rows()[0]
        for column in LlmCall.__table__.columns:
            self.assertNotIn(PROMPT, str(getattr(row, column.key)), column.key)
        output = stream.getvalue()
        self.assertNotIn(PROMPT, output)
        self.assertIn("prompt_hash=" + hash_prompt(PROMPT), output)
        self.assertIn("run_id=" + str(self.run_id), output)

    def test_prompt_hash_may_be_supplied_instead_of_prompt(self) -> None:
        expected = hash_prompt("already hashed")
        record_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-small",
            purpose="classify",
            prompt_hash=expected,
        )
        self.assertEqual(self.rows()[0].prompt_hash, expected)

    def test_missing_prompt_and_hash_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            record_llm_call(
                self.engine,
                run_id=self.run_id,
                model="local-small",
                purpose="screen",
            )
        self.assertEqual(self.rows(), [])

    def test_malformed_prompt_hash_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            record_llm_call(
                self.engine,
                run_id=self.run_id,
                model="local-small",
                purpose="screen",
                prompt_hash=PROMPT,
            )

    def test_cache_hit_defaults_to_false(self) -> None:
        record_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-small",
            purpose="screen",
            prompt="p",
        )
        self.assertFalse(self.rows()[0].cache_hit)

    def test_llm_call_uses_the_run_base_and_foreign_key(self) -> None:
        self.assertIs(LlmCall.metadata, Base.metadata)
        foreign_keys = list(LlmCall.__table__.c.run_id.foreign_keys)
        self.assertEqual([fk.target_fullname for fk in foreign_keys], ["run.id"])

    def test_record_llm_call_accepts_a_session(self) -> None:
        with Session(self.engine) as session:
            call_id = record_llm_call(
                session,
                run_id=self.run_id,
                model="local-small",
                purpose="screen",
                prompt="p",
            )
        rows = self.rows()
        self.assertEqual([row.id for row in rows], [call_id])

    def test_record_llm_call_joins_an_open_session_transaction(self) -> None:
        with Session(self.engine) as session:
            with session.begin():
                record_llm_call(
                    session,
                    run_id=self.run_id,
                    model="local-small",
                    purpose="screen",
                    prompt="p",
                )
                self.assertEqual(session.scalar(sa.select(sa.func.count()).select_from(LlmCall)), 1)
        self.assertEqual(len(self.rows()), 1)

    def test_context_helper_records_latency_and_usage(self) -> None:
        with log_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-large",
            purpose="extract",
            prompt=PROMPT,
        ) as usage:
            usage.tokens_in = 100
            usage.tokens_out = 20
            usage.cost = 0.004
            usage.cache_hit = True

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.status, OK)
        self.assertGreaterEqual(row.latency_ms, 0)
        self.assertEqual(row.tokens_in, 100)
        self.assertEqual(row.tokens_out, 20)
        self.assertAlmostEqual(row.cost, 0.004)
        self.assertTrue(row.cache_hit)
        self.assertIsNone(row.error)
        self.assertEqual(row.prompt_hash, hash_prompt(PROMPT))

    def test_context_helper_writes_row_and_reraises_on_error(self) -> None:
        stream = self.configure_logging_for_test()
        with self.assertRaises(RuntimeError):
            with log_llm_call(
                self.engine,
                run_id=self.run_id,
                model="local-large",
                purpose="extract",
                prompt=PROMPT,
            ) as usage:
                usage.tokens_in = 5
                raise RuntimeError("provider failed")

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.status, ERROR)
        self.assertGreaterEqual(row.latency_ms, 0)
        self.assertEqual(row.tokens_in, 5)
        self.assertIsNotNone(row.error)
        assert row.error is not None
        self.assertIn("RuntimeError: provider failed", row.error)
        self.assertNotIn(PROMPT, row.error)
        self.assertNotIn(PROMPT, stream.getvalue())

    def test_context_helper_writes_one_row_per_call(self) -> None:
        for _ in range(2):
            with log_llm_call(
                self.engine,
                run_id=self.run_id,
                model="local-small",
                purpose="screen",
                prompt="p",
            ):
                pass
        self.assertEqual(len(self.rows()), 2)

    def test_error_text_is_capped(self) -> None:
        record_llm_call(
            self.engine,
            run_id=self.run_id,
            model="local-small",
            purpose="screen",
            prompt="p",
            status=ERROR,
            error="x" * 5000,
        )
        row = self.rows()[0]
        assert row.error is not None
        self.assertLessEqual(len(row.error), 2000)


if __name__ == "__main__":
    unittest.main()
