"""Tests for the environment-based engine settings (PF-03)."""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.config import Settings, load_settings


class SettingsTestCase(unittest.TestCase):
    """Hermetic base: no ambient DM_* variables leak into settings."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def env_file(self, *lines: str) -> Path:
        path = self.tmp_path / ".env"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def absent_env_file(self) -> Path:
        return self.tmp_path / "absent.env"


class DefaultsTests(SettingsTestCase):
    def test_defaults_apply_without_env_or_file(self) -> None:
        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings, Settings())
        self.assertEqual(settings.llm_gateway_url, "http://localhost:8000/v1")
        self.assertEqual(settings.local_model, "local-model")
        self.assertEqual(settings.embedding_url, "http://localhost:8001/v1")
        self.assertEqual(settings.reranker_url, "http://localhost:8002/v1")
        self.assertEqual(settings.cloud_llm_model, "cloud-model")
        self.assertEqual(settings.llm_budget_usd, 0.0)
        self.assertEqual(settings.max_crawl_concurrency, 2)

    def test_missing_optional_cloud_key_is_none(self) -> None:
        self.assertIsNone(load_settings(self.absent_env_file()).cloud_api_key)

    def test_cloud_key_is_absent_from_repr(self) -> None:
        os.environ["DM_CLOUD_LLM_API_KEY"] = "super-secret-key"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.cloud_api_key, "super-secret-key")
        self.assertNotIn("super-secret-key", repr(settings))
        self.assertNotIn("super-secret-key", str(settings))

    def test_settings_are_frozen(self) -> None:
        settings = load_settings(self.absent_env_file())

        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.max_crawl_concurrency = 9  # type: ignore[misc]


class DotenvTests(SettingsTestCase):
    def test_dotenv_values_are_loaded(self) -> None:
        env_file = self.env_file(
            "# comment line",
            "",
            "DM_LLM_GATEWAY_URL=http://gateway.local:9000/v1",
            "DM_LOCAL_MODEL='quoted-model'",
            'DM_EMBEDDING_URL="http://emb.local:9001/v1"',
            "DM_LLM_BUDGET_USD=2.5",
            "DM_MAX_CRAWL_CONCURRENCY=4",
            "IGNORED_KEY=ignored",
        )

        settings = load_settings(env_file)

        self.assertEqual(settings.llm_gateway_url, "http://gateway.local:9000/v1")
        self.assertEqual(settings.local_model, "quoted-model")
        self.assertEqual(settings.embedding_url, "http://emb.local:9001/v1")
        self.assertEqual(settings.llm_budget_usd, 2.5)
        self.assertEqual(settings.max_crawl_concurrency, 4)

    def test_default_dotenv_is_read_from_cwd(self) -> None:
        self.env_file("DM_LOCAL_MODEL=from-cwd")
        previous_cwd = os.getcwd()
        os.chdir(self.tmp_path)
        self.addCleanup(os.chdir, previous_cwd)

        self.assertEqual(load_settings().local_model, "from-cwd")

    def test_real_environment_overrides_dotenv(self) -> None:
        env_file = self.env_file(
            "DM_LOCAL_MODEL=from-file",
            "DM_RERANKER_URL=http://file.local:1/v1",
        )
        os.environ["DM_LOCAL_MODEL"] = "from-env"

        settings = load_settings(env_file)

        self.assertEqual(settings.local_model, "from-env")
        self.assertEqual(settings.reranker_url, "http://file.local:1/v1")

    def test_environment_without_dotenv(self) -> None:
        os.environ["DM_CLOUD_LLM_API_KEY"] = "test-key"
        os.environ["DM_CLOUD_LLM_MODEL"] = "cloud-test"
        os.environ["DM_LLM_BUDGET_USD"] = "12.5"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.cloud_api_key, "test-key")
        self.assertEqual(settings.cloud_llm_model, "cloud-test")
        self.assertEqual(settings.llm_budget_usd, 12.5)

    def test_malformed_dotenv_line_fails_fast(self) -> None:
        env_file = self.env_file("DM_LOCAL_MODEL")

        with self.assertRaisesRegex(ValueError, ".env"):
            load_settings(env_file)


class ValidationTests(SettingsTestCase):
    def test_invalid_int_names_the_variable(self) -> None:
        os.environ["DM_MAX_CRAWL_CONCURRENCY"] = "two"

        with self.assertRaisesRegex(ValueError, "DM_MAX_CRAWL_CONCURRENCY"):
            load_settings(self.absent_env_file())

    def test_invalid_float_names_the_variable(self) -> None:
        os.environ["DM_LLM_BUDGET_USD"] = "1,50"

        with self.assertRaisesRegex(ValueError, "DM_LLM_BUDGET_USD"):
            load_settings(self.absent_env_file())

    def test_negative_budget_is_rejected(self) -> None:
        os.environ["DM_LLM_BUDGET_USD"] = "-0.01"

        with self.assertRaisesRegex(ValueError, "DM_LLM_BUDGET_USD"):
            load_settings(self.absent_env_file())

    def test_non_finite_budget_is_rejected(self) -> None:
        os.environ["DM_LLM_BUDGET_USD"] = "inf"

        with self.assertRaisesRegex(ValueError, "DM_LLM_BUDGET_USD"):
            load_settings(self.absent_env_file())

    def test_concurrency_below_one_is_rejected(self) -> None:
        os.environ["DM_MAX_CRAWL_CONCURRENCY"] = "0"

        with self.assertRaisesRegex(ValueError, "DM_MAX_CRAWL_CONCURRENCY"):
            load_settings(self.absent_env_file())


if __name__ == "__main__":
    unittest.main()
