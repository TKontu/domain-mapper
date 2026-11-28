"""Per-URL snapshot comparison of one entity across two crawl runs (SD-01).

ARCHITECTURE §4 "Re-crawl": only changed snapshots (by ``sha256``) should trigger
re-assessment. This module is the read-only first slice: for one entity it takes the
latest snapshot per URL in each of two runs (latest by ``fetched_at``, then ``id``) and
classifies every URL as ``changed``, ``unchanged``, ``added`` (head only) or ``removed``
(base only). By default the two most recent runs (highest ids) that hold snapshots for
the entity are compared. Selection happens in SQL per (run, url) for that one entity; no
whole-table load, no writes, no body diff.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.snapshot_store import SnapshotRow

STATUSES = ("changed", "unchanged", "added", "removed")


class SnapshotDiffError(ValueError):
    """The requested comparison has no two runs with snapshots for the entity."""


@dataclass(frozen=True)
class UrlDiff:
    url: str
    status: str


@dataclass(frozen=True)
class SnapshotDiff:
    entity_id: int
    base_run_id: int
    head_run_id: int
    urls: list[UrlDiff]

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(STATUSES, 0)
        for item in self.urls:
            out[item.status] += 1
        return out


def _latest_hashes(session: Session, entity_id: int, run_id: int) -> dict[str, str]:
    ranked = (
        sa.select(
            SnapshotRow.url,
            SnapshotRow.sha256,
            sa.func.row_number()
            .over(
                partition_by=SnapshotRow.url,
                order_by=(SnapshotRow.fetched_at.desc(), SnapshotRow.id.desc()),
            )
            .label("rn"),
        )
        .where(SnapshotRow.entity_id == entity_id, SnapshotRow.run_id == run_id)
        .subquery()
    )
    rows = session.execute(sa.select(ranked.c.url, ranked.c.sha256).where(ranked.c.rn == 1))
    return {url: sha for url, sha in rows}


def _runs_with_snapshots(session: Session, entity_id: int) -> list[int]:
    stmt = (
        sa.select(SnapshotRow.run_id)
        .where(SnapshotRow.entity_id == entity_id, SnapshotRow.run_id.is_not(None))
        .group_by(SnapshotRow.run_id)
        .order_by(SnapshotRow.run_id.desc())
    )
    return [r for r in session.scalars(stmt) if r is not None]


def diff_entity_runs(
    engine: Engine,
    entity_id: int,
    base_run_id: int | None = None,
    head_run_id: int | None = None,
) -> SnapshotDiff:
    """Compare the latest snapshot per URL of ``entity_id`` in two runs; read-only.

    Unset runs default to the most recent runs holding snapshots for the entity: the head
    is the newest (other than an explicit base), the base the newest run older than the head.
    Raises :class:`SnapshotDiffError` when fewer than two such runs exist or a named run
    has no snapshots for the entity.
    """
    with Session(engine) as session:
        runs = _runs_with_snapshots(session, entity_id)
        for named in (base_run_id, head_run_id):
            if named is not None and named not in runs:
                raise SnapshotDiffError(f"run {named} has no snapshots for entity {entity_id}")
        head = head_run_id
        if head is None:
            head = next((r for r in runs if r != base_run_id), None)
        base = base_run_id
        if base is None and head is not None:
            base = next((r for r in runs if r < head), None)
        if head is None or base is None or base == head:
            raise SnapshotDiffError(f"entity {entity_id} has fewer than two runs with snapshots")
        base_hashes = _latest_hashes(session, entity_id, base)
        head_hashes = _latest_hashes(session, entity_id, head)
    urls = []
    for url in sorted(base_hashes.keys() | head_hashes.keys()):
        if url not in head_hashes:
            status = "removed"
        elif url not in base_hashes:
            status = "added"
        elif base_hashes[url] == head_hashes[url]:
            status = "unchanged"
        else:
            status = "changed"
        urls.append(UrlDiff(url=url, status=status))
    return SnapshotDiff(entity_id=entity_id, base_run_id=base, head_run_id=head, urls=urls)
