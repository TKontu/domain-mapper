"""Tests for the ``landscape`` Typer CLI skeleton (PF-04)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from core.cli import STAGES, app

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)  # no stray .env
        self.addCleanup(os.chdir, cwd)

    def invoke(self, *args: str):
        return self.runner.invoke(app, list(args))

    def test_help_lists_every_stage(self) -> None:
        result = self.invoke("--help")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(STAGES), 12)
        for stage in STAGES:
            self.assertIn(stage, result.output)

    def test_dry_run_prints_plan(self) -> None:
        result = self.invoke(
            "discover", "--scope", EXAMPLE_SCOPE, "--layer", "home", "--limit", "5", "--dry-run"
        )
        self.assertEqual(result.exit_code, 0, result.output)
        for text in ("stage: discover", f"scope: {EXAMPLE_SCOPE}", "layer: home", "limit: 5"):
            self.assertIn(text, result.output)
        self.assertIn("llm_gateway_url", result.output)

    def test_api_key_never_printed(self) -> None:
        os.environ["DM_CLOUD_LLM_API_KEY"] = "sk-very-secret-value"
        result = self.invoke("assess", "--scope", EXAMPLE_SCOPE, "--dry-run")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("sk-very-secret-value", result.output)
        self.assertNotIn("cloud_api_key", result.output)

    def test_limit_zero_rejected(self) -> None:
        result = self.invoke("crawl", "--scope", EXAMPLE_SCOPE, "--limit", "0", "--dry-run")
        self.assertNotEqual(result.exit_code, 0)

    def test_missing_scope_dir_exits_2(self) -> None:
        missing = str(Path(self.tmp.name) / "nope")
        result = self.invoke("screen", "--scope", missing, "--dry-run")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("nope", result.output)

    def test_bad_setting_exits_2_naming_variable(self) -> None:
        os.environ["DM_MAX_CRAWL_CONCURRENCY"] = "zero"
        result = self.invoke("parse", "--scope", EXAMPLE_SCOPE, "--dry-run")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("DM_MAX_CRAWL_CONCURRENCY", result.output)

    def test_non_dry_run_not_implemented(self) -> None:
        result = self.invoke("publish", "--scope", EXAMPLE_SCOPE)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not implemented yet", result.output)


if __name__ == "__main__":
    unittest.main()
