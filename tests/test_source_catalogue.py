"""Tests for the ``source`` catalogue seeded from a scope pack (AD-02).

The suite runs offline against in-memory SQLite. The optional PostgreSQL test is
skipped unless ``DM_TEST_DATABASE_URL`` points at a disposable database.

Every pack is built under ``tmp_path`` so the tests exercise the real
``load_scope_pack`` path and never touch ``scopes/_example`` (owned by SP-04).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.pool import StaticPool

from core.runs import Base, create_all
from core.scope_pack import load_scope_pack
from core.source_catalogue import SeedSummary, Source, seed_sources

POSTGRES_URL = os.environ.get("DM_TEST_DATABASE_URL")

SCOPE = {
    "id": "fixture",
    "version": "0.0.1",
    "title": "Fixture landscape",
    "perspective": "Test perspective.",
    "segments": [{"id": "alpha", "label": "Alpha", "definition": "First segment."}],
    "layers": [
        {"id": "all", "countries": ["XA"], "discovery_target": "notable", "rubric_depth": "core"}
    ],
}

RUBRIC = {
    "status_scale": [{"id": "yes", "label": "Yes", "value": 1.0, "color": "green"}],
    "topics": [
        {
            "id": "topic",
            "label": "Topic",
            "segments": ["alpha"],
            "criteria": [
                {
                    "id": "c1",
                    "question": "Does it apply?",
                    "applies_to": ["alpha"],
                    "statuses": {"yes": "Yes"},
                }
            ],
        }
    ],
}

FIRST_SOURCE = {
    "id": "alpha_directory",
    "role": "discovery",
    "adapter": "web_search",
    "countries": ["XA"],
    "access": "open",
    "licence": "CC-BY-4.0",
    "params": {"name": "Alpha directory", "base_url": "https://alpha.example.test/"},
}

SECOND_SOURCE = {
    "id": "alpha_register",
    "role": "register",
    "adapter": "http",
    "access": "restricted",
}


def memory_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def write_pack(
    root: Path,
    sources: list[dict[str, Any]],
    scope_id: str = "fixture",
) -> Path:
    """Write a minimal valid pack under ``root`` and return its directory."""
    pack_dir = root / f"pack-{scope_id}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    scope = dict(SCOPE, id=scope_id)
    for name, data in (
        ("scope.yaml", scope),
        ("rubric.yaml", RUBRIC),
        ("sources.yaml", {"sources": sources}),
    ):
        (pack_dir / name).write_text(yaml.safe_dump(data), encoding="utf-8")
    return pack_dir


def rows_by_scope(engine: Engine) -> dict[tuple[str, str], Source]:
    with sa.orm.Session(engine) as session:
        rows = session.scalars(sa.select(Source).order_by(Source.scope_id, Source.id)).all()
    return {(row.scope_id, row.id): row for row in rows}


def snapshot(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        key: {
            "name": row.name,
            "roles": list(row.roles),
            "access_class": row.access_class,
            "licence": row.licence,
            "base_url": row.base_url,
            "adapter": row.adapter,
        }
        for key, row in rows_by_scope(engine).items()
    }


def writes_during(engine: Engine, action: Any) -> list[str]:
    """Run ``action`` and return the INSERT/UPDATE/DELETE statements it issued."""
    statements: list[str] = []

    def record(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: Any
    ) -> None:
        verb = statement.lstrip().split(None, 1)[0].upper()
        if verb in {"INSERT", "UPDATE", "DELETE"}:
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        action()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return statements


class SourceCatalogueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = memory_engine()
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def seed(self, sources: list[dict[str, Any]], scope_id: str = "fixture") -> SeedSummary:
        pack = load_scope_pack(write_pack(self.tmp, sources, scope_id=scope_id))
        return seed_sources(self.engine, pack)


class SchemaTests(SourceCatalogueTestCase):
    def test_create_all_creates_the_source_table(self) -> None:
        columns = {column["name"] for column in sa.inspect(self.engine).get_columns("source")}
        self.assertEqual(
            columns,
            {
                "id",
                "scope_id",
                "name",
                "roles",
                "access_class",
                "licence",
                "base_url",
                "adapter",
            },
        )

    def test_row_key_is_scoped_to_the_scope(self) -> None:
        key = sa.inspect(self.engine).get_pk_constraint("source")
        self.assertEqual(set(key["constrained_columns"]), {"scope_id", "id"})


class SeedTests(SourceCatalogueTestCase):
    def test_seed_inserts_every_pack_source_with_mapped_fields(self) -> None:
        summary = self.seed([FIRST_SOURCE, SECOND_SOURCE])

        self.assertEqual(summary, SeedSummary(inserted=2, updated=0, unchanged=0))
        rows = rows_by_scope(self.engine)
        self.assertEqual(set(rows), {("fixture", "alpha_directory"), ("fixture", "alpha_register")})
        first = rows[("fixture", "alpha_directory")]
        self.assertEqual(first.name, "Alpha directory")
        self.assertEqual(list(first.roles), ["discovery"])
        self.assertEqual(first.access_class, "open")
        self.assertEqual(first.licence, "CC-BY-4.0")
        self.assertEqual(first.base_url, "https://alpha.example.test/")
        self.assertEqual(first.adapter, "web_search")
        second = rows[("fixture", "alpha_register")]
        # No params: the name falls back to the id and the URL stays NULL.
        self.assertEqual(second.name, "alpha_register")
        self.assertIsNone(second.base_url)
        self.assertEqual(list(second.roles), ["register"])
        self.assertEqual(second.access_class, "restricted")
        self.assertIsNone(second.licence)

    def test_reseed_writes_nothing(self) -> None:
        self.seed([FIRST_SOURCE, SECOND_SOURCE])
        before = snapshot(self.engine)

        summary: list[SeedSummary] = []
        statements = writes_during(
            self.engine, lambda: summary.append(self.seed([FIRST_SOURCE, SECOND_SOURCE]))
        )

        self.assertEqual(summary, [SeedSummary(inserted=0, updated=0, unchanged=2)])
        self.assertEqual(statements, [])
        self.assertEqual(snapshot(self.engine), before)

    def test_changed_entry_updates_only_its_row(self) -> None:
        self.seed([FIRST_SOURCE, SECOND_SOURCE])
        changed = dict(FIRST_SOURCE, licence="CC0-1.0", access="restricted")
        changed["params"] = dict(FIRST_SOURCE["params"], base_url="https://beta.example.test/")

        summary = self.seed([changed, SECOND_SOURCE])

        self.assertEqual(summary, SeedSummary(inserted=0, updated=1, unchanged=1))
        row = rows_by_scope(self.engine)[("fixture", "alpha_directory")]
        self.assertEqual(row.licence, "CC0-1.0")
        self.assertEqual(row.access_class, "restricted")
        self.assertEqual(row.base_url, "https://beta.example.test/")
        # A third seed of the changed pack is unchanged again.
        self.assertEqual(self.seed([changed, SECOND_SOURCE]), SeedSummary(unchanged=2))

    def test_added_entry_inserts_only_that_row(self) -> None:
        self.seed([FIRST_SOURCE])
        summary = self.seed([FIRST_SOURCE, SECOND_SOURCE])
        self.assertEqual(summary, SeedSummary(inserted=1, updated=0, unchanged=1))
        self.assertEqual(len(rows_by_scope(self.engine)), 2)

    def test_entry_removed_from_the_pack_keeps_its_row(self) -> None:
        # Retirement is a later decision; rows referenced by candidates must not
        # silently disappear on the next seed.
        self.seed([FIRST_SOURCE, SECOND_SOURCE])
        summary = self.seed([FIRST_SOURCE])
        self.assertEqual(summary, SeedSummary(unchanged=1))
        self.assertIn(("fixture", "alpha_register"), rows_by_scope(self.engine))

    def test_same_source_id_in_two_scopes_stays_two_rows(self) -> None:
        self.seed([FIRST_SOURCE], scope_id="fixture")
        self.seed([FIRST_SOURCE], scope_id="other")

        rows = rows_by_scope(self.engine)
        self.assertEqual(set(rows), {("fixture", "alpha_directory"), ("other", "alpha_directory")})
        self.assertEqual(self.seed([FIRST_SOURCE], scope_id="other"), SeedSummary(unchanged=1))
        self.assertEqual(len(rows_by_scope(self.engine)), 2)

    def test_empty_pack_is_a_noop(self) -> None:
        self.assertEqual(self.seed([]), SeedSummary())
        self.assertEqual(rows_by_scope(self.engine), {})


@unittest.skipUnless(POSTGRES_URL, "DM_TEST_DATABASE_URL is not set; PostgreSQL path not exercised")
class PostgresSeedTests(unittest.TestCase):
    """Optional: proves the table and seed work on PostgreSQL 16."""

    def setUp(self) -> None:
        assert POSTGRES_URL is not None
        self.engine = create_engine(POSTGRES_URL)
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as connection:
            Base.metadata.drop_all(connection)
        create_all(self.engine)
        self.addCleanup(self._drop_tables)

    def _drop_tables(self) -> None:
        with self.engine.begin() as connection:
            Base.metadata.drop_all(connection)

    def test_seed_is_idempotent_and_updates_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack = load_scope_pack(write_pack(Path(tmp), [FIRST_SOURCE, SECOND_SOURCE]))
            self.assertEqual(seed_sources(self.engine, pack), SeedSummary(inserted=2))
            self.assertEqual(seed_sources(self.engine, pack), SeedSummary(unchanged=2))
            changed = dict(FIRST_SOURCE, licence="CC0-1.0")
            pack = load_scope_pack(write_pack(Path(tmp), [changed, SECOND_SOURCE]))
            self.assertEqual(seed_sources(self.engine, pack), SeedSummary(updated=1, unchanged=1))


if __name__ == "__main__":
    unittest.main()
