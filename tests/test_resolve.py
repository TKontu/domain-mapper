"""Tests for ``core.resolve`` (RV-01); SQLite in-memory only, no network."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.candidate_store import Candidate
from core.resolve import DomainGroup, group_candidates, normalize_domain
from core.runs import create_all

FOUND_AT = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> sa.Engine:
    eng = sa.create_engine("sqlite://")
    create_all(eng)
    return eng


def _seed(engine: sa.Engine, rows: Iterable[tuple[str, str, int | None]] = ()) -> list[Candidate]:
    """Insert ``(name, url, resolved_entity_id)`` rows in order; return them."""
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
    return list_rows(engine)


def list_rows(engine: sa.Engine) -> list[Candidate]:
    with Session(engine) as session:
        return list(session.scalars(sa.select(Candidate).order_by(Candidate.id)))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://WWW.Example.COM:8443/a?b", "example.com"),
        ("http://example.com.", "example.com"),
        ("http://www.example.com", "example.com"),
        ("  https://example.com/path  ", "example.com"),
        ("http://user:secret@Example.com:8080/path?q=1#frag", "example.com"),
        ("https://münchen.example/kontakt", "xn--mnchen-3ya.example"),
        ("http://[::1]/", "::1"),
        ("http://127.0.0.1:8000/x", "127.0.0.1"),
        ("ftp://example.com/file", None),
        ("ftp://x", None),
        ("mailto:someone@example.com", None),
        ("", None),
        ("   ", None),
        ("not a url", None),
        ("http://", None),
        ("http://exa mple.com/", None),
        ("http://exa%6dple.com/", None),
        ("http://a..b/", None),
    ],
)
def test_normalize_domain(url: str, expected: str | None) -> None:
    assert normalize_domain(url) == expected


def test_groups_sorted_by_count_then_domain(engine: sa.Engine) -> None:
    _seed(
        engine,
        [
            ("A first", "https://a.example.com/1", None),
            ("B", "https://b.example.com/", None),
            ("A second", "https://www.a.example.com/2", None),
            ("C", "https://C.EXAMPLE.COM:443/", None),
        ],
    )
    summary = group_candidates(engine)
    assert summary.groups == (
        DomainGroup(domain="a.example.com", name="A first", count=2),
        DomainGroup(domain="b.example.com", name="B", count=1),
        DomainGroup(domain="c.example.com", name="C", count=1),
    )
    assert (summary.candidates, summary.skipped) == (4, 0)


def test_resolved_rows_never_counted(engine: sa.Engine) -> None:
    _seed(
        engine,
        [
            ("kept", "https://a.example.com/", None),
            ("same domain but resolved", "https://www.a.example.com/", 7),
            ("other domain but resolved", "https://z.example.com/", 7),
        ],
    )
    summary = group_candidates(engine)
    # A removed ``resolved_entity_id IS NULL`` filter would show a count of 3 and two domains.
    assert summary.groups == (DomainGroup(domain="a.example.com", name="kept", count=1),)
    assert (summary.candidates, summary.skipped) == (1, 0)


def test_unnormalisable_urls_counted_as_skipped(engine: sa.Engine) -> None:
    _seed(
        engine,
        [
            ("ok", "https://a.example.com/", None),
            ("not a url", "not a url", None),
            ("ftp", "ftp://files.example.com/x", None),
            ("blank url", "", None),
            ("skipped but resolved", "not a url either", 3),
        ],
    )
    summary = group_candidates(engine)
    assert summary.groups == (DomainGroup(domain="a.example.com", name="ok", count=1),)
    assert (summary.candidates, summary.skipped) == (1, 3)


def test_empty_table(engine: sa.Engine) -> None:
    summary = group_candidates(engine)
    assert summary.groups == ()
    assert (summary.candidates, summary.skipped) == (0, 0)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_every_row_seen_across_pages(engine: sa.Engine, batch: int) -> None:
    rows = [(f"N{i}", f"https://d{i}.example.com/", None) for i in range(5)]
    rows += [("dup", "https://d0.example.com/again", None), ("bad", "nope", None)]
    _seed(engine, rows)
    summary = group_candidates(engine, batch=batch)
    assert len(summary.groups) == 5
    assert summary.groups[0] == DomainGroup(domain="d0.example.com", name="N0", count=2)
    assert summary.candidates == 6
    assert summary.skipped == 1


def test_batch_below_one_rejected(engine: sa.Engine) -> None:
    with pytest.raises(ValueError, match="batch"):
        group_candidates(engine, batch=0)


def test_performs_no_insert_update_or_delete(engine: sa.Engine) -> None:
    _seed(
        engine,
        [
            ("a", "https://a.example.com/", None),
            ("b", "https://b.example.com/", 5),
            ("bad", "not a url", None),
        ],
    )
    before = [(r.id, r.name, r.url, r.resolved_entity_id) for r in list_rows(engine)]
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        group_candidates(engine)
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)
    writes = [
        s for s in statements if s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
    ]
    assert writes == []
    after = [(r.id, r.name, r.url, r.resolved_entity_id) for r in list_rows(engine)]
    assert after == before
