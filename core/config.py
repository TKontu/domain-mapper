"""Runtime settings loaded from ``DM_*`` environment variables.

An optional ``.env`` file can seed the values; real environment variables always
override it. Values are validated once at load time so a bad deployment fails
fast instead of halfway through a run. The cloud API key is optional and never
included in the settings object's ``repr``/``str``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "DM_"
DEFAULT_ENV_FILE = ".env"

DEFAULT_LLM_GATEWAY_URL = "http://localhost:8000/v1"
DEFAULT_LOCAL_MODEL = "local-model"
DEFAULT_EMBEDDING_URL = "http://localhost:8001/v1"
DEFAULT_RERANKER_URL = "http://localhost:8002/v1"
DEFAULT_CLOUD_LLM_MODEL = "cloud-model"
DEFAULT_LLM_BUDGET_USD = 0.0
DEFAULT_MAX_CRAWL_CONCURRENCY = 2
# Local compose `db` placeholders (host/port/user/database); the password is set
# in .env, so it is not part of the default URL.
DEFAULT_DATABASE_URL = "postgresql+psycopg://domain_mapper@localhost:5432/domain_mapper"


@dataclass(frozen=True)
class Settings:
    """Validated, immutable engine settings."""

    llm_gateway_url: str = DEFAULT_LLM_GATEWAY_URL
    local_model: str = DEFAULT_LOCAL_MODEL
    embedding_url: str = DEFAULT_EMBEDDING_URL
    reranker_url: str = DEFAULT_RERANKER_URL
    cloud_api_key: str | None = field(default=None, repr=False)
    cloud_llm_model: str = DEFAULT_CLOUD_LLM_MODEL
    llm_budget_usd: float = DEFAULT_LLM_BUDGET_USD
    max_crawl_concurrency: int = DEFAULT_MAX_CRAWL_CONCURRENCY
    database_url: str = field(default=DEFAULT_DATABASE_URL, repr=False)


def load_settings(env_file: str | Path = DEFAULT_ENV_FILE) -> Settings:
    """Return settings from the environment, optionally seeded by ``env_file``.

    A missing ``env_file`` is not an error (fresh checkouts have no ``.env``).
    Raises ``ValueError`` naming the offending variable for malformed numbers,
    a negative budget or a crawl concurrency below one.
    """
    file_values = _read_env_file(Path(env_file))

    def raw(name: str) -> str | None:
        key = f"{ENV_PREFIX}{name}"
        if key in os.environ:
            return os.environ[key]
        return file_values.get(key)

    cloud_key = raw("CLOUD_LLM_API_KEY")
    return Settings(
        llm_gateway_url=_text(raw("LLM_GATEWAY_URL"), DEFAULT_LLM_GATEWAY_URL),
        local_model=_text(raw("LOCAL_MODEL"), DEFAULT_LOCAL_MODEL),
        embedding_url=_text(raw("EMBEDDING_URL"), DEFAULT_EMBEDDING_URL),
        reranker_url=_text(raw("RERANKER_URL"), DEFAULT_RERANKER_URL),
        cloud_api_key=None if cloud_key is None or not cloud_key.strip() else cloud_key,
        cloud_llm_model=_text(raw("CLOUD_LLM_MODEL"), DEFAULT_CLOUD_LLM_MODEL),
        llm_budget_usd=_parse_float(
            "LLM_BUDGET_USD", raw("LLM_BUDGET_USD"), DEFAULT_LLM_BUDGET_USD
        ),
        max_crawl_concurrency=_parse_int(
            "MAX_CRAWL_CONCURRENCY", raw("MAX_CRAWL_CONCURRENCY"), DEFAULT_MAX_CRAWL_CONCURRENCY
        ),
        database_url=_text(raw("DATABASE_URL"), DEFAULT_DATABASE_URL),
    )


def _text(value: str | None, default: str) -> str:
    return default if value is None or not value.strip() else value


def _parse_float(name: str, value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {value!r}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{ENV_PREFIX}{name} must be a finite number, got {value!r}")
    if parsed < 0:
        raise ValueError(f"{ENV_PREFIX}{name} must be >= 0, got {value!r}")
    return parsed


def _parse_int(name: str, value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {value!r}") from None
    if parsed < 1:
        raise ValueError(f"{ENV_PREFIX}{name} must be >= 1, got {value!r}")
    return parsed


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` file; quoted values lose their surrounding quotes."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, separator, value = stripped.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"{path}:{number}: expected KEY=VALUE, got {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values
