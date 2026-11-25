"""Guards for the local docker compose stack (PF-05).

Stdlib only: the gate runs offline without PyYAML or a docker daemon, so these
tests read the shipped files as text and check the properties the item fixes:
a pinned Postgres 16 + pgvector database, MinIO only under an opt-in profile,
an app image built from the Dockerfile, and no literal credentials anywhere.

Each guard is a pure ``*_violations`` function so the negative controls below
can feed it a broken copy and prove the guard is not vacuous.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMPOSE = "compose.yaml"
DOCKERFILE = "Dockerfile"
DOCKERIGNORE = ".dockerignore"
INIT_SQL = "deploy/postgres/init/01-vector.sql"
COMPOSE_DOCS = "docs/compose.md"
ENV_EXAMPLE = ".env.example"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def service_blocks(compose: str) -> dict[str, str]:
    """Split the ``services:`` mapping into name -> indented block text."""
    blocks: dict[str, list[str]] = {}
    in_services = False
    current: str | None = None
    for line in compose.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            if current is not None:
                blocks[current].append(line)
            continue
        if not line.startswith(" "):
            in_services = line.startswith("services:")
            current = None
            continue
        if not in_services:
            continue
        match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def image_of(block: str) -> str:
    match = re.search(r"^\s+image:\s*(.+?)\s*$", block, re.M)
    return match.group(1) if match else ""


def default_image(value: str) -> str:
    """Resolve the ``${VAR:-default}`` form used for overridable images."""
    match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]+)\}", value)
    return match.group(1) if match else value


def code_lines(block: str) -> str:
    """Drop comment lines so prose in a service block cannot satisfy a guard."""
    return "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))


def profiles_of(block: str) -> set[str]:
    match = re.search(r"^\s+profiles:\s*(.*)$", block, re.M)
    if not match:
        return set()
    inline = match.group(1).strip()
    if inline.startswith("["):
        return {part.strip().strip("'\"") for part in inline.strip("[]").split(",") if part.strip()}
    profiles: set[str] = set()
    for line in block[match.end() :].splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if not item.startswith("- "):
            break
        profiles.add(item[2:].strip().strip("'\""))
    return profiles


def compose_violations(compose: str) -> list[str]:
    """Return one message per broken stack property (empty list = all good)."""
    problems: list[str] = []
    services = service_blocks(compose)

    expected = {"db", "app", "minio"}
    if set(services) != expected:
        problems.append(f"services: expected {sorted(expected)}, found {sorted(services)}")
    if set(services) != expected:
        return problems

    db = services["db"]
    image = image_of(db)
    if not re.fullmatch(r"pgvector/pgvector:\d+\.\d+\.\d+-pg16", image):
        problems.append(f"db.image: expected a pinned pgvector pg16 tag, found {image!r}")
    if "./deploy/postgres/init:/docker-entrypoint-initdb.d:ro" not in db:
        problems.append(
            "db.init: init scripts not mounted read-only at /docker-entrypoint-initdb.d"
        )
    if "POSTGRES_PASSWORD: ${DM_POSTGRES_PASSWORD:?" not in db:
        problems.append(
            "db.password: POSTGRES_PASSWORD must come from ${DM_POSTGRES_PASSWORD:?...}"
        )
    for line in db.splitlines():
        match = re.match(r"\s+POSTGRES_(?:DB|USER|PASSWORD):[ \t]*(.*)$", line)
        if match and not match.group(1).startswith("${"):
            problems.append("db.password: literal POSTGRES_* credential value in compose")
            break
    if "pg_isready" not in db:
        problems.append("db.healthcheck: missing pg_isready healthcheck")

    app = code_lines(services["app"])
    if "dockerfile: Dockerfile" not in app:
        problems.append("app.build: expected dockerfile: Dockerfile")
    if "path: .env" not in app or "required: false" not in app:
        problems.append("app.env_file: .env must be optional (path: .env, required: false)")
    if "condition: service_healthy" not in app:
        problems.append("app.depends_on: must wait for the healthy db")
    if re.search(r"^\s+depends_on:", app, re.M) and re.search(r"^\s+(?:-\s*)?minio\s*:", app, re.M):
        problems.append("app.depends_on: app must not depend on the optional minio profile")

    minio = services["minio"]
    if "minio" not in profiles_of(minio):
        problems.append("minio.profile: minio service is not behind the opt-in `minio` profile")
    if "MINIO_ROOT_USER: ${DM_MINIO_ROOT_USER:?" not in minio:
        problems.append("minio.user: MINIO_ROOT_USER must come from ${DM_MINIO_ROOT_USER:?...}")
    if "MINIO_ROOT_PASSWORD: ${DM_MINIO_ROOT_PASSWORD:?" not in minio:
        problems.append(
            "minio.password: MINIO_ROOT_PASSWORD must come from ${DM_MINIO_ROOT_PASSWORD:?...}"
        )
    default = default_image(image_of(services["minio"]))
    pinned = re.fullmatch(r"[^:]+:20\d{2}\.\d{1,2}\.\d{1,2}-debian-12-r\d+", default) or re.search(
        r"@sha256:[0-9a-f]{64}$", default
    )
    if not pinned:
        problems.append(
            f"minio.image: minio image is not pinned, found {image_of(services['minio'])!r}"
        )

    return problems


def dockerignore_violations(ignore: str) -> list[str]:
    entries = {
        line.strip().strip("/")
        for line in ignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")
    }
    required = [".env", ".env.*", ".git", ".agent-runs", "scopes", "snapshots"]
    return [
        f"dockerignore: {pattern!r} is not excluded from the build context"
        for pattern in required
        if pattern not in entries
    ]


def dockerfile_violations(dockerfile: str) -> list[str]:
    problems: list[str] = []
    if not re.search(r"^FROM python:3\.12-slim", dockerfile, re.M):
        problems.append("dockerfile.base: expected FROM python:3.12-slim")
    if not re.search(r"^RUN uv sync --frozen", dockerfile, re.M):
        problems.append("dockerfile.uv: expected a frozen `uv sync`")
    if re.search(r"^COPY .*\.env", dockerfile, re.M):
        problems.append("dockerfile.secret: Dockerfile copies a .env into the image")
    if "DM_" in dockerfile:
        problems.append("dockerfile.secret: environment secrets baked into the image")
    return problems


def env_example_violations(env_example: str) -> list[str]:
    values: dict[str, str] = {}
    for line in env_example.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    problems = [
        f"env.example: missing {key}"
        for key in ("DM_POSTGRES_DB", "DM_POSTGRES_USER", "DM_POSTGRES_PASSWORD")
        if key not in values
    ]
    if values.get("DM_POSTGRES_PASSWORD"):
        problems.append("env.example: DM_POSTGRES_PASSWORD must stay an empty placeholder")
    return problems


class ComposeGuardTests(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        for name in (COMPOSE, DOCKERFILE, DOCKERIGNORE, INIT_SQL, COMPOSE_DOCS, ENV_EXAMPLE):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file(), f"{name} is missing")

    def test_compose_stack_is_valid(self) -> None:
        self.assertEqual(compose_violations(read(COMPOSE)), [])

    def test_vector_extension_is_created_on_first_init(self) -> None:
        sql = read(INIT_SQL)
        self.assertRegex(sql, r"CREATE EXTENSION IF NOT EXISTS vector;")

    def test_build_context_excludes_private_material(self) -> None:
        self.assertEqual(dockerignore_violations(read(DOCKERIGNORE)), [])

    def test_app_image_is_python312_with_frozen_uv_sync(self) -> None:
        self.assertEqual(dockerfile_violations(read(DOCKERFILE)), [])

    def test_env_example_has_postgres_placeholders(self) -> None:
        self.assertEqual(env_example_violations(read(ENV_EXAMPLE)), [])

    def test_compose_docs_show_usage_and_live_check(self) -> None:
        docs = read(COMPOSE_DOCS)
        for needle in (
            "docker compose up -d db",
            "--profile minio",
            "down -v",
            "pg_extension",
            "DM_POSTGRES_PASSWORD",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, docs)


class ComposeGuardNegativeControlTests(unittest.TestCase):
    """Mutate the real files; each guard must notice its property break."""

    def setUp(self) -> None:
        self.compose = read(COMPOSE)
        self.ignore = read(DOCKERIGNORE)
        self.dockerfile = read(DOCKERFILE)
        self.env_example = read(ENV_EXAMPLE)

    def assert_flags(self, problems: list[str], prefix: str) -> None:
        self.assertTrue(
            any(problem.startswith(prefix) for problem in problems),
            f"guard {prefix!r} did not fire: {problems}",
        )

    def test_detects_floating_db_image(self) -> None:
        broken = self.compose.replace("pgvector/pgvector:0.8.7-pg16", "pgvector/pgvector:pg16")
        self.assertEqual(self.compose.count("pgvector/pgvector:0.8.7-pg16"), 1)
        self.assert_flags(compose_violations(broken), "db.image")

    def test_detects_literal_db_password(self) -> None:
        broken = self.compose.replace(
            "POSTGRES_PASSWORD: ${DM_POSTGRES_PASSWORD:?",
            "POSTGRES_PASSWORD: hunter2 # ${DM_POSTGRES_PASSWORD:?",
        )
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "db.password")

    def test_detects_unmounted_init_script(self) -> None:
        broken = self.compose.replace(
            "      - ./deploy/postgres/init:/docker-entrypoint-initdb.d:ro\n", ""
        )
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "db.init")

    def test_detects_minio_without_profile(self) -> None:
        broken = self.compose.replace('    profiles: ["minio"]\n', "")
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "minio.profile")

    def test_detects_required_env_file(self) -> None:
        broken = self.compose.replace("required: false", "required: true")
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "app.env_file")

    def test_detects_app_depending_on_minio(self) -> None:
        broken = self.compose.replace(
            "      db:\n        condition: service_healthy",
            "      db:\n        condition: service_healthy\n      minio:\n        condition: service_started",
            1,
        )
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "app.depends_on")

    def test_detects_unpinned_minio_image(self) -> None:
        broken = self.compose.replace(
            "bitnamilegacy/minio:2025.7.23-debian-12-r5", "bitnamilegacy/minio:latest"
        )
        self.assertNotEqual(broken, self.compose)
        self.assert_flags(compose_violations(broken), "minio.image")

    def test_detects_secret_in_build_context(self) -> None:
        self.assert_flags(
            dockerignore_violations(self.ignore.replace(".env\n", "", 1)), "dockerignore"
        )
        self.assert_flags(
            dockerignore_violations(self.ignore.replace("scopes\n", "", 1)), "dockerignore"
        )

    def test_detects_floating_python_base(self) -> None:
        broken = self.dockerfile.replace("FROM python:3.12-slim", "FROM python:latest")
        self.assertNotEqual(broken, self.dockerfile)
        self.assert_flags(dockerfile_violations(broken), "dockerfile.base")

    def test_detects_env_copy_in_dockerfile(self) -> None:
        broken = self.dockerfile + "\nCOPY .env .\n"
        self.assert_flags(dockerfile_violations(broken), "dockerfile.secret")

    def test_detects_secret_value_in_env_example(self) -> None:
        broken = self.env_example.replace(
            "DM_POSTGRES_PASSWORD=\n", "DM_POSTGRES_PASSWORD=hunter2\n"
        )
        self.assertNotEqual(broken, self.env_example)
        self.assert_flags(env_example_violations(broken), "env.example")

    def test_detects_missing_env_example_key(self) -> None:
        broken = self.env_example.replace("DM_POSTGRES_USER=", "DM_POSTGRES_USERNAME=")
        self.assertNotEqual(broken, self.env_example)
        self.assert_flags(env_example_violations(broken), "env.example")


if __name__ == "__main__":
    unittest.main()
