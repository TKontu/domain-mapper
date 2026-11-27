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
        self.assertEqual(settings.local_cost_per_mtok_in, 0.0)
        self.assertEqual(settings.local_cost_per_mtok_out, 0.0)
        self.assertEqual(settings.cloud_cost_per_mtok_in, 0.0)
        self.assertEqual(settings.cloud_cost_per_mtok_out, 0.0)
        self.assertEqual(settings.max_crawl_concurrency, 2)

    def test_missing_optional_cloud_key_is_none(self) -> None:
        self.assertIsNone(load_settings(self.absent_env_file()).cloud_api_key)

    def test_database_url_defaults_to_the_local_compose_placeholders(self) -> None:
        settings = load_settings(self.absent_env_file())

        self.assertEqual(
            settings.database_url,
            "postgresql+psycopg://domain_mapper@localhost:5432/domain_mapper",
        )

    def test_cloud_key_is_absent_from_repr(self) -> None:
        os.environ["DM_CLOUD_LLM_API_KEY"] = "super-secret-key"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.cloud_api_key, "super-secret-key")
        self.assertNotIn("super-secret-key", repr(settings))
        self.assertNotIn("super-secret-key", str(settings))

    def test_gateway_key_is_loaded_and_absent_from_repr(self) -> None:
        os.environ["DM_LLM_GATEWAY_API_KEY"] = "fake-test-key"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.llm_gateway_api_key, "fake-test-key")
        self.assertNotIn("fake-test-key", repr(settings))
        self.assertNotIn("fake-test-key", str(settings))

    def test_blank_or_missing_gateway_key_is_none(self) -> None:
        self.assertIsNone(load_settings(self.absent_env_file()).llm_gateway_api_key)
        os.environ["DM_LLM_GATEWAY_API_KEY"] = "   "
        self.assertIsNone(load_settings(self.absent_env_file()).llm_gateway_api_key)

    def test_database_url_password_is_absent_from_repr(self) -> None:
        os.environ["DM_DATABASE_URL"] = (
            "postgresql+psycopg://domain_mapper:db-secret@db:5432/domain_mapper"
        )

        settings = load_settings(self.absent_env_file())

        self.assertNotIn("db-secret", repr(settings))
        self.assertNotIn("db-secret", str(settings))

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

    def test_database_url_comes_from_the_environment(self) -> None:
        os.environ["DM_DATABASE_URL"] = "postgresql+psycopg://u:p@db:5432/other"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.database_url, "postgresql+psycopg://u:p@db:5432/other")

    def test_blank_database_url_falls_back_to_the_default(self) -> None:
        env_file = self.env_file("DM_DATABASE_URL=")

        settings = load_settings(env_file)

        self.assertEqual(
            settings.database_url,
            "postgresql+psycopg://domain_mapper@localhost:5432/domain_mapper",
        )

    def test_environment_without_dotenv(self) -> None:
        os.environ["DM_CLOUD_LLM_API_KEY"] = "test-key"
        os.environ["DM_CLOUD_LLM_MODEL"] = "cloud-test"
        os.environ["DM_LLM_BUDGET_USD"] = "12.5"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.cloud_api_key, "test-key")
        self.assertEqual(settings.cloud_llm_model, "cloud-test")
        self.assertEqual(settings.llm_budget_usd, 12.5)

    def test_cost_rates_load_from_a_dotenv_file(self) -> None:
        env_file = self.env_file(
            "DM_LOCAL_COST_PER_MTOK_IN=0.5",
            "DM_LOCAL_COST_PER_MTOK_OUT=1.5",
            "DM_CLOUD_COST_PER_MTOK_IN=3.0",
            "DM_CLOUD_COST_PER_MTOK_OUT=15.0",
        )

        settings = load_settings(env_file)

        self.assertEqual(settings.local_cost_per_mtok_in, 0.5)
        self.assertEqual(settings.local_cost_per_mtok_out, 1.5)
        self.assertEqual(settings.cloud_cost_per_mtok_in, 3.0)
        self.assertEqual(settings.cloud_cost_per_mtok_out, 15.0)

    def test_cost_rates_come_from_the_environment(self) -> None:
        os.environ["DM_LOCAL_COST_PER_MTOK_IN"] = "2"
        os.environ["DM_LOCAL_COST_PER_MTOK_OUT"] = "4"
        os.environ["DM_CLOUD_COST_PER_MTOK_IN"] = "6"
        os.environ["DM_CLOUD_COST_PER_MTOK_OUT"] = "8"

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.local_cost_per_mtok_in, 2.0)
        self.assertEqual(settings.local_cost_per_mtok_out, 4.0)
        self.assertEqual(settings.cloud_cost_per_mtok_in, 6.0)
        self.assertEqual(settings.cloud_cost_per_mtok_out, 8.0)

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

    def test_negative_cost_rate_is_rejected(self) -> None:
        for name in (
            "LOCAL_COST_PER_MTOK_IN",
            "LOCAL_COST_PER_MTOK_OUT",
            "CLOUD_COST_PER_MTOK_IN",
            "CLOUD_COST_PER_MTOK_OUT",
        ):
            with self.subTest(name=name):
                os.environ[f"DM_{name}"] = "-0.01"
                self.addCleanup(os.environ.pop, f"DM_{name}", None)
                with self.assertRaisesRegex(ValueError, f"DM_{name}"):
                    load_settings(self.absent_env_file())
                os.environ.pop(f"DM_{name}", None)

    def test_non_finite_cost_rate_is_rejected(self) -> None:
        for name in (
            "LOCAL_COST_PER_MTOK_IN",
            "LOCAL_COST_PER_MTOK_OUT",
            "CLOUD_COST_PER_MTOK_IN",
            "CLOUD_COST_PER_MTOK_OUT",
        ):
            for value in ("nan", "inf"):
                with self.subTest(name=name, value=value):
                    os.environ[f"DM_{name}"] = value
                    self.addCleanup(os.environ.pop, f"DM_{name}", None)
                    with self.assertRaisesRegex(ValueError, f"DM_{name}"):
                        load_settings(self.absent_env_file())
                    os.environ.pop(f"DM_{name}", None)

    def test_concurrency_below_one_is_rejected(self) -> None:
        os.environ["DM_MAX_CRAWL_CONCURRENCY"] = "0"

        with self.assertRaisesRegex(ValueError, "DM_MAX_CRAWL_CONCURRENCY"):
            load_settings(self.absent_env_file())


class LlmTimeoutTests(SettingsTestCase):
    def test_timeout_defaults_to_the_adapter_default(self) -> None:
        from adapters.llm import DEFAULT_TIMEOUT_SECONDS

        settings = load_settings(self.absent_env_file())

        self.assertEqual(settings.llm_timeout_s, 60.0)
        self.assertEqual(settings.llm_timeout_s, DEFAULT_TIMEOUT_SECONDS)

    def test_timeout_comes_from_the_environment(self) -> None:
        os.environ["DM_LLM_TIMEOUT_S"] = "900"

        self.assertEqual(load_settings(self.absent_env_file()).llm_timeout_s, 900.0)

    def test_non_numeric_timeout_names_the_variable(self) -> None:
        os.environ["DM_LLM_TIMEOUT_S"] = "abc"

        with self.assertRaisesRegex(ValueError, "DM_LLM_TIMEOUT_S"):
            load_settings(self.absent_env_file())

    def test_non_positive_timeout_names_the_variable(self) -> None:
        for value in ("0", "-5"):
            os.environ["DM_LLM_TIMEOUT_S"] = value
            with self.assertRaisesRegex(ValueError, "DM_LLM_TIMEOUT_S"):
                load_settings(self.absent_env_file())


if __name__ == "__main__":
    unittest.main()
