"""Tests for :func:`core.resolve.apply_resolution` (RV-02); in-memory/temp SQLite, offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import sqlalchemy as sa
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from core import resolve as resolve_mod
from core.candidate_store import Candidate
from app.cli import app
from core.entity_store import Entity
from core.resolve import ApplySummary, apply_resolution
from core.runs import Run, create_all

FOUND_AT = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


def _seed(engine: sa.Engine, rows: list[tuple[str, str, int | None]]) -> None:
    with Session(engine) as session, session.begin():
        for name, url, resolved in rows:
            session.add(
                Candidate(
                    name=name,
                    url=url,
                    channel="search",
                    query=None,
                    found_at=FOUND_AT,
                    resolved_entity_id=resolved,
                )
            )


def _entities(engine: sa.Engine) -> list[tuple[int, str, str, str, str, int | None]]:
    with Session(engine) as session:
        return [
            (e.id, e.scope_id, e.name, e.primary_domain, e.status, e.group_id)
            for e in session.scalars(sa.select(Entity).order_by(Entity.id))
        ]


def _links(engine: sa.Engine) -> list[tuple[int, int | None]]:
    with Session(engine) as session:
        return [
            tuple(r)  # type: ignore[misc]
            for r in session.execute(
                sa.select(Candidate.id, Candidate.resolved_entity_id).order_by(Candidate.id)
            )
        ]


class ApplyResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.engine = sa.create_engine(f"sqlite:///{tmp.name}/dm.sqlite")
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)

    def seed3(self) -> None:
        _seed(
            self.engine,
            [
                ("A first", "https://a.example.com/1", None),
                ("B", "https://b.example.com/", None),
                ("A second", "https://www.A.example.com/2", None),
            ],
        )

    def test_creates_one_entity_per_domain_and_links_all(self) -> None:
        for batch in (1, 2, 500):
            with self.subTest(batch=batch):
                self.engine.dispose()
                with self.engine.begin() as conn:
                    conn.execute(sa.delete(Candidate))
                    conn.execute(sa.delete(Entity))
                self.seed3()
                summary = apply_resolution(self.engine, "example", batch=batch)
                self.assertEqual(summary, ApplySummary(created=2, linked=3, skipped=0))
                ents = _entities(self.engine)
                self.assertEqual(
                    [(s, n, d, st, g) for _, s, n, d, st, g in ents],
                    [
                        ("example", "A first", "a.example.com", "unscreened", None),
                        ("example", "B", "b.example.com", "unscreened", None),
                    ],
                )
                a_id, b_id = ents[0][0], ents[1][0]
                self.assertEqual([e for _, e in _links(self.engine)], [a_id, b_id, a_id])

    def test_rerun_is_idempotent(self) -> None:
        self.seed3()
        _seed(self.engine, [("bad", "not a url", None)])
        apply_resolution(self.engine, "example", batch=2)
        before = (_entities(self.engine), _links(self.engine))
        summary = apply_resolution(self.engine, "example", batch=2)
        self.assertEqual(summary, ApplySummary(created=0, linked=0, skipped=1))
        self.assertEqual((_entities(self.engine), _links(self.engine)), before)

    def test_new_candidate_links_to_existing_entity(self) -> None:
        self.seed3()
        apply_resolution(self.engine, "example")
        _seed(self.engine, [("A third", "http://a.example.com:8080/x", None)])
        summary = apply_resolution(self.engine, "example")
        self.assertEqual(summary, ApplySummary(created=0, linked=1, skipped=0))
        a_id = _entities(self.engine)[0][0]
        self.assertEqual(_links(self.engine)[-1][1], a_id)
        self.assertEqual(len(_entities(self.engine)), 2)

    def test_already_resolved_rows_untouched(self) -> None:
        _seed(
            self.engine,
            [("old", "https://a.example.com/", 77), ("new", "https://a.example.com/", None)],
        )
        summary = apply_resolution(self.engine, "example")
        self.assertEqual(summary, ApplySummary(created=1, linked=1, skipped=0))
        links = _links(self.engine)
        self.assertEqual(links[0][1], 77)
        self.assertEqual(links[1][1], _entities(self.engine)[0][0])

    def test_unnormalisable_counted_and_left_null(self) -> None:
        _seed(
            self.engine,
            [
                ("bad", "not a url", None),
                ("ok", "https://a.example.com/", None),
                ("ftp", "ftp://x", None),
            ],
        )
        summary = apply_resolution(self.engine, "example", batch=1)
        self.assertEqual(summary, ApplySummary(created=1, linked=1, skipped=2))
        links = _links(self.engine)
        self.assertIsNone(links[0][1])
        self.assertIsNone(links[2][1])

    def test_same_domain_two_scopes_two_entities(self) -> None:
        _seed(self.engine, [("a", "https://a.example.com/", None)])
        apply_resolution(self.engine, "one")
        _seed(self.engine, [("a2", "https://a.example.com/", None)])
        summary = apply_resolution(self.engine, "two")
        self.assertEqual(summary, ApplySummary(created=1, linked=1, skipped=0))
        ents = _entities(self.engine)
        self.assertEqual(
            [(s, d) for _, s, _, d, _, _ in ents],
            [("one", "a.example.com"), ("two", "a.example.com")],
        )

    def test_failure_rolls_back_only_that_batch(self) -> None:
        self.seed3()
        real = resolve_mod._link
        calls = {"n": 0}

        def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 3:  # page 1 links a and b; page 2 links A second
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        with mock.patch.object(resolve_mod, "_link", flaky), self.assertRaises(RuntimeError):
            apply_resolution(self.engine, "example", batch=2)
        ents = _entities(self.engine)
        self.assertEqual([d for _, _, _, d, _, _ in ents], ["a.example.com", "b.example.com"])
        self.assertEqual([e is None for _, e in _links(self.engine)], [False, False, True])
        # A clean rerun finishes the rolled-back batch against the committed entity.
        summary = apply_resolution(self.engine, "example", batch=2)
        self.assertEqual(summary, ApplySummary(created=0, linked=1, skipped=0))

    def test_failure_rolls_back_new_entity_of_that_batch(self) -> None:
        self.seed3()
        real = resolve_mod._link
        calls = {"n": 0}

        def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        with mock.patch.object(resolve_mod, "_link", flaky), self.assertRaises(RuntimeError):
            apply_resolution(self.engine, "example", batch=1)
        # batch 1 (a) committed; batch 2 (b) rolled back including its new entity.
        self.assertEqual([d for _, _, _, d, _, _ in _entities(self.engine)], ["a.example.com"])
        summary = apply_resolution(self.engine, "example", batch=1)
        self.assertEqual(summary, ApplySummary(created=1, linked=2, skipped=0))

    def _created_runs(self) -> dict[str, int | None]:
        with Session(self.engine) as session:
            return {e.primary_domain: e.created_run for e in session.scalars(sa.select(Entity))}

    def test_run_id_sets_created_run_default_none(self) -> None:
        self.seed3()
        apply_resolution(self.engine, "other")
        self.assertEqual(set(self._created_runs().values()), {None})

    def test_run_id_on_new_entities_reused_keep_original(self) -> None:
        _seed(self.engine, [("A", "https://a.example.com/", None)])
        apply_resolution(self.engine, "example", run_id=7)
        self.assertEqual(self._created_runs(), {"a.example.com": 7})
        _seed(
            self.engine,
            [("A2", "https://a.example.com/2", None), ("B", "https://b.example.com/", None)],
        )
        apply_resolution(self.engine, "example", run_id=8)
        self.assertEqual(self._created_runs(), {"a.example.com": 7, "b.example.com": 8})

    def test_rejects_bad_batch(self) -> None:
        with self.assertRaises(ValueError):
            apply_resolution(self.engine, "example", batch=0)


class ResolveApplyCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.db_url = f"sqlite:///{tmp.name}/dm.sqlite"
        os.environ["DM_DATABASE_URL"] = self.db_url
        engine = sa.create_engine(self.db_url)
        try:
            create_all(engine)
            _seed(
                engine,
                [
                    ("A first", "https://a.example.com/1", None),
                    ("B", "https://b.example.com/", None),
                    ("A second", "https://a.example.com/2", None),
                    ("bad", "not a url", None),
                ],
            )
        finally:
            engine.dispose()

    def invoke(self, *args: str):  # type: ignore[no-untyped-def]
        return self.runner.invoke(app, ["resolve", "--scope", EXAMPLE_SCOPE, *args])

    def test_apply_prints_summary_and_rerun_is_zero(self) -> None:
        first = self.invoke("--apply")
        self.assertEqual(first.exit_code, 0, first.output)
        self.assertEqual(first.stdout.splitlines(), ["created 2 linked 3 skipped 1"])
        second = self.invoke("--apply")
        self.assertEqual(second.exit_code, 0, second.output)
        # RT-01 (134/A1): unchanged candidates -> the resolve unit is skipped, not re-applied.
        self.assertEqual(second.stdout.splitlines(), ["created 0 linked 0 skipped 0", "skipped 1"])
        engine = sa.create_engine(self.db_url)
        try:
            self.assertEqual({s for _, s, _, _, _, _ in _entities(engine)}, {"example"})
        finally:
            engine.dispose()

    def test_apply_sets_created_run_to_resolve_run_id(self) -> None:
        result = self.invoke("--apply")
        self.assertEqual(result.exit_code, 0, result.output)
        engine = sa.create_engine(self.db_url)
        try:
            with Session(engine) as session:
                run_ids = list(session.scalars(sa.select(Run.id).where(Run.stage == "resolve")))
                created = set(session.scalars(sa.select(Entity.created_run)))
        finally:
            engine.dispose()
        self.assertEqual(len(run_ids), 1)
        self.assertEqual(created, {run_ids[0]})

    def test_without_apply_is_read_only(self) -> None:
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "a.example.com\t2\tA first",
                "b.example.com\t1\tB",
                "domains 2 candidates 3",
                "skipped 1",
            ],
        )
        engine = sa.create_engine(self.db_url)
        try:
            self.assertEqual(_entities(engine), [])
        finally:
            engine.dispose()

    def test_apply_db_failure_exits_2_without_url(self) -> None:
        os.environ["DM_DATABASE_URL"] = "postgresql+psycopg://user:hunter2@db.invalid:1/x"
        with mock.patch("app.cli.create_db_engine", side_effect=RuntimeError("db.invalid hunter2")):
            result = self.invoke("--apply")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (RuntimeError)", result.stderr)
        self.assertNotIn("db.invalid", result.output)
        self.assertNotIn("hunter2", result.output)

    def test_apply_mid_run_failure_exits_2(self) -> None:
        with mock.patch.object(
            resolve_mod, "_link", side_effect=sa.exc.OperationalError("x", {}, Exception("x"))
        ):
            result = self.invoke("--apply")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("error: database unavailable (OperationalError)", result.stderr)
        self.assertNotIn("sqlite", result.output)
