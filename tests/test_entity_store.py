"""Tests for the read-only entity pager :func:`core.entity_store.iter_entities` (CR-04).

Offline: a temp SQLite file, entities inserted directly. The pager must page by id (keyset),
filter by scope id, keep id order and honour ``limit`` without holding every row at once.
"""

from __future__ import annotations

import tempfile
import unittest

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.entity_store import STATUS_UNSCREENED, Entity, iter_entities
from core.runs import create_all


class IterEntitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.engine = sa.create_engine(f"sqlite:///{tmp.name}/dm.sqlite")
        self.addCleanup(self.engine.dispose)
        create_all(self.engine)

    def seed(self, rows: list[tuple[str, str]]) -> list[int]:
        """Insert ``(scope_id, primary_domain)`` in order; return the assigned ids."""
        ids: list[int] = []
        with Session(self.engine) as session, session.begin():
            for scope_id, domain in rows:
                entity = Entity(
                    scope_id=scope_id,
                    name=domain,
                    primary_domain=domain,
                    status=STATUS_UNSCREENED,
                )
                session.add(entity)
                session.flush()
                ids.append(entity.id)
        return ids

    def collect(self, scope_id: str = "example", **kwargs: int | None) -> list[tuple[int, str]]:
        return [(e.id, e.primary_domain) for e in iter_entities(self.engine, scope_id, **kwargs)]

    def test_filters_scope_and_keeps_id_order_across_pages(self) -> None:
        ids = self.seed(
            [
                ("example", "a.example.com"),
                ("other", "b.example.com"),
                ("example", "c.example.com"),
                ("example", "d.example.com"),
                ("other", "e.example.com"),
                ("example", "f.example.com"),
            ]
        )
        expected = [
            (ids[0], "a.example.com"),
            (ids[2], "c.example.com"),
            (ids[3], "d.example.com"),
            (ids[5], "f.example.com"),
        ]
        # batch 2 walks several keyset pages, including a full 2-row page and a partial one.
        self.assertEqual(self.collect(batch=2), expected)
        self.assertEqual(self.collect(batch=500), expected)

    def test_exact_page_boundary_returns_every_row_once(self) -> None:
        ids = self.seed([("example", f"h{i}.example.com") for i in range(4)])
        self.assertEqual(
            [entity_id for entity_id, _ in self.collect(batch=2)],
            ids,
        )

    def test_limit_stops_at_n_in_id_order(self) -> None:
        ids = self.seed([("example", f"h{i}.example.com") for i in range(5)])
        self.assertEqual(
            [entity_id for entity_id, _ in self.collect(batch=2, limit=3)],
            ids[:3],
        )
        self.assertEqual([entity_id for entity_id, _ in self.collect(batch=2, limit=1)], ids[:1])
        self.assertEqual(
            [entity_id for entity_id, _ in self.collect(batch=2, limit=10)],
            ids,
        )

    def test_empty_scope_and_missing_scope_yield_nothing(self) -> None:
        self.seed([("other", "a.example.com")])
        self.assertEqual(self.collect("example"), [])
        self.assertEqual(self.collect("no-such-scope"), [])

    def test_rejects_non_positive_batch_and_limit(self) -> None:
        for kwargs in ({"batch": 0}, {"batch": -1}, {"limit": 0}, {"limit": -2}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                iter_entities(self.engine, "example", **kwargs)


if __name__ == "__main__":
    unittest.main()
