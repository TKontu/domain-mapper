"""Tests for the structured one-line logging setup (RT-02).

The suite is offline: it configures the root logger with a ``StringIO`` stream
and inspects the rendered lines.
"""

from __future__ import annotations

import io
import logging
import unittest
from collections.abc import Callable

from core.log import configure_logging


class ConfigureLoggingTests(unittest.TestCase):
    def _configure(self, level: str = "INFO") -> io.StringIO:
        stream = io.StringIO()
        handler = configure_logging(level, stream=stream)

        def cleanup() -> None:
            logging.getLogger().removeHandler(handler)
            handler.close()

        self.addCleanup(cleanup)
        return stream

    def _logger(self) -> logging.Logger:
        return logging.getLogger("core.test_log")

    def test_one_line_per_record_with_structured_fields(self) -> None:
        stream = self._configure()
        logger = self._logger()
        logger.info("first record")
        logger.info(
            "llm_call",
            extra={
                "run_id": 7,
                "prompt_hash": "a" * 64,
                "cache_hit": True,
                "cost": 0.25,
                "latency_ms": 12,
            },
        )

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("level=INFO", lines[0])
        self.assertIn("logger=core.test_log", lines[0])
        self.assertIn("message=\"first record\"", lines[0])
        self.assertIn("message=llm_call", lines[1])
        self.assertIn("run_id=7", lines[1])
        self.assertIn("prompt_hash=" + "a" * 64, lines[1])
        self.assertIn("cache_hit=True", lines[1])
        self.assertIn("cost=0.25", lines[1])
        self.assertIn("latency_ms=12", lines[1])

    def test_multiline_message_stays_on_one_line(self) -> None:
        stream = self._configure()
        self._logger().info("first\nsecond")

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("\\n", lines[0])

    def test_exception_traceback_stays_on_one_line(self) -> None:
        stream = self._configure()
        try:
            raise ValueError("boom")
        except ValueError:
            self._logger().exception("call failed")

        output = stream.getvalue()
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("exc=", output)
        self.assertIn("ValueError", output)

    def test_level_filters_lower_records(self) -> None:
        stream = self._configure("WARNING")
        self._logger().info("hidden")
        self._logger().warning("shown")

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("level=WARNING", lines[0])
        self.assertIn("message=shown", lines[0])

    def test_reconfiguring_replaces_the_handler(self) -> None:
        first = self._configure()
        self._logger().info("first")
        second = self._configure()
        self._logger().info("second")

        self.assertEqual(first.getvalue().count("\n"), 1)
        self.assertIn("message=first", first.getvalue())
        self.assertEqual(second.getvalue().count("\n"), 1)
        self.assertIn("message=second", second.getvalue())

    def test_configure_returns_a_handler(self) -> None:
        handler = configure_logging("INFO", stream=io.StringIO())
        self.addCleanup(handler.close)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        self.assertIsInstance(handler, logging.Handler)
        self.assertIsInstance(handler.formatter, logging.Formatter)

    def test_formatter_is_callable_for_any_extra(self) -> None:
        # A stray non-string extra must not break one-line rendering.
        formatter: Callable[[logging.LogRecord], str]
        formatter = configure_logging("INFO", stream=io.StringIO()).formatter.format  # type: ignore[union-attr,assignment]
        record = logging.LogRecord("core.test_log", logging.INFO, __file__, 1, "m", (), None)
        record.extra_value = {"a": 1}  # type: ignore[attr-defined]
        rendered = formatter(record)
        self.assertEqual(len(rendered.splitlines()), 1)
        self.assertIn("extra_value=", rendered)


if __name__ == "__main__":
    unittest.main()
