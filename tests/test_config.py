"""Tests for the environment-based engine settings (PF-03)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from core.config import Settings, load_settings


@pytest.fixture(autouse=True)
def clean_dm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test hermetic: no ambient DM_* variables leak into settings."""
    for key in list(os.environ):
        if key.startswith("DM_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def no_env_file(tmp_path: Path) -> Path:
    return tmp_path / "absent.env"


def test_defaults_apply_without_env_or_file(no_env_file: Path) -> None:
    settings = load_settings(no_env_file)

    assert settings == Settings()
    assert settings.llm_gateway_url == "http://localhost:8000/v1"
    assert settings.local_model == "local-model"
    assert settings.embedding_url == "http://localhost:8001/v1"
    assert settings.reranker_url == "http://localhost:8002/v1"
    assert settings.cloud_llm_model == "cloud-model"
    assert settings.llm_budget_usd == 0.0
    assert settings.max_crawl_concurrency == 2


def test_missing_optional_cloud_key_is_none(no_env_file: Path) -> None:
    assert load_settings(no_env_file).cloud_api_key is None


def test_dotenv_values_are_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line",
                "",
                "DM_LLM_GATEWAY_URL=http://gateway.local:9000/v1",
                "DM_LOCAL_MODEL='quoted-model'",
                'DM_EMBEDDING_URL="http://emb.local:9001/v1"',
                "DM_LLM_BUDGET_USD=2.5",
                "DM_MAX_CRAWL_CONCURRENCY=4",
                "IGNORED_KEY=ignored",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.llm_gateway_url == "http://gateway.local:9000/v1"
    assert settings.local_model == "quoted-model"
    assert settings.embedding_url == "http://emb.local:9001/v1"
    assert settings.llm_budget_usd == 2.5
    assert settings.max_crawl_concurrency == 4


def test_default_dotenv_is_read_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("DM_LOCAL_MODEL=from-cwd\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert load_settings().local_model == "from-cwd"


def test_real_environment_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DM_LOCAL_MODEL=from-file\nDM_RERANKER_URL=http://file.local:1/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DM_LOCAL_MODEL", "from-env")

    settings = load_settings(env_file)

    assert settings.local_model == "from-env"
    assert settings.reranker_url == "http://file.local:1/v1"


def test_environment_without_dotenv(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_CLOUD_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DM_CLOUD_LLM_MODEL", "cloud-test")
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "12.5")

    settings = load_settings(no_env_file)

    assert settings.cloud_api_key == "test-key"
    assert settings.cloud_llm_model == "cloud-test"
    assert settings.llm_budget_usd == 12.5


def test_invalid_int_names_the_variable(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_MAX_CRAWL_CONCURRENCY", "two")

    with pytest.raises(ValueError, match="DM_MAX_CRAWL_CONCURRENCY"):
        load_settings(no_env_file)


def test_invalid_float_names_the_variable(
    no_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "1,50")

    with pytest.raises(ValueError, match="DM_LLM_BUDGET_USD"):
        load_settings(no_env_file)


def test_negative_budget_is_rejected(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "-0.01")

    with pytest.raises(ValueError, match="DM_LLM_BUDGET_USD"):
        load_settings(no_env_file)


def test_non_finite_budget_is_rejected(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_LLM_BUDGET_USD", "inf")

    with pytest.raises(ValueError, match="DM_LLM_BUDGET_USD"):
        load_settings(no_env_file)


def test_concurrency_below_one_is_rejected(
    no_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DM_MAX_CRAWL_CONCURRENCY", "0")

    with pytest.raises(ValueError, match="DM_MAX_CRAWL_CONCURRENCY"):
        load_settings(no_env_file)


def test_malformed_dotenv_line_fails_fast(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DM_LOCAL_MODEL\n", encoding="utf-8")

    with pytest.raises(ValueError, match=".env"):
        load_settings(env_file)


def test_cloud_key_is_absent_from_repr(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_CLOUD_LLM_API_KEY", "super-secret-key")

    settings = load_settings(no_env_file)

    assert settings.cloud_api_key == "super-secret-key"
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings)


def test_settings_are_frozen(no_env_file: Path) -> None:
    settings = load_settings(no_env_file)

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.max_crawl_concurrency = 9  # type: ignore[misc]
