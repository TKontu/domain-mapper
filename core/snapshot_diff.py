"""Per-URL snapshot comparison of one entity across two crawl runs (SD-01).

ARCHITECTURE §4 "Re-crawl": only changed snapshots (by ``sha256``) should trigger
re-assessment. This module is the read-only first slice: for one entity it takes the
latest snapshot per URL in each of two runs (latest by ``fetched_at``, then ``id``) and
classifies every URL as ``changed``, ``unchanged``, ``added`` (head only) or ``removed``
(base only). By default the two most recent runs (highest ids) that hold snapshots for
the entity are compared. Selection happens in SQL per (run, url) for that one entity; no
whole-table load, no writes, no body diff.

:func:`stale_assessments` (SD-02) maps a diff onto the entity's current assessments: one
is stale when any of its evidence URLs is ``changed`` or ``removed``. Production caller:
``app.cli.run_reassess_plan``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.assessment_store import iter_current_assessments
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


def _latest_rows(session: Session, entity_id: int, run_id: int) -> dict[str, tuple[str, str]]:
    """Latest snapshot per URL in one run: ``url -> (sha256, final_url)``."""
    ranked = (
        sa.select(
            SnapshotRow.url,
            SnapshotRow.sha256,
            SnapshotRow.final_url,
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
    rows = session.execute(
        sa.select(ranked.c.url, ranked.c.sha256, ranked.c.final_url).where(ranked.c.rn == 1)
    )
    return {url: (sha, final) for url, sha, final in rows}


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
    """See :func:`_diff_with_final_urls`; returns only the diff."""
    return _diff_with_final_urls(engine, entity_id, base_run_id, head_run_id)[0]


def _diff_with_final_urls(
    engine: Engine,
    entity_id: int,
    base_run_id: int | None,
    head_run_id: int | None,
) -> tuple[SnapshotDiff, dict[str, set[str]]]:
    """Compare the latest snapshot per URL of ``entity_id`` in two runs; read-only.

    Unset runs default to the most recent runs holding snapshots for the entity: the head
    is the newest (newer than an explicit base), the base the newest run older than the head.
    Raises :class:`SnapshotDiffError` when fewer than two such runs exist, a named run has no
    snapshots for the entity, base and head are the same run, or an explicit base has no newer
    run (never diffs in reverse).
    """
    with Session(engine) as session:
        runs = _runs_with_snapshots(session, entity_id)
        for named in (base_run_id, head_run_id):
            if named is not None and named not in runs:
                raise SnapshotDiffError(f"run {named} has no snapshots for entity {entity_id}")
        if base_run_id is not None and base_run_id == head_run_id:
            raise SnapshotDiffError("base and head run are the same")
        head = head_run_id
        if head is None and base_run_id is not None:
            head = next((r for r in runs if r > base_run_id), None)
            if head is None:
                raise SnapshotDiffError(
                    f"no run newer than base run {base_run_id} has snapshots for entity {entity_id}"
                )
        if head is None:
            head = runs[0] if runs else None
        base = base_run_id
        if base is None and head is not None:
            base = next((r for r in runs if r < head), None)
        if head is None or base is None or base == head:
            raise SnapshotDiffError(f"entity {entity_id} has fewer than two runs with snapshots")
        base_rows = _latest_rows(session, entity_id, base)
        head_rows = _latest_rows(session, entity_id, head)
    base_hashes = {u: sha for u, (sha, _f) in base_rows.items()}
    head_hashes = {u: sha for u, (sha, _f) in head_rows.items()}
    final_urls: dict[str, set[str]] = {}
    for rows in (base_rows, head_rows):
        for u, (_sha, final) in rows.items():
            final_urls.setdefault(u, set()).add(final)
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
    diff = SnapshotDiff(entity_id=entity_id, base_run_id=base, head_run_id=head, urls=urls)
    return diff, final_urls


STALE_STATUSES = frozenset({"changed", "removed"})


@dataclass(frozen=True)
class StaleAssessment:
    assessment_id: int
    criterion_id: str
    rubric_version: str
    urls: list[UrlDiff]


@dataclass(frozen=True)
class StalePlan:
    diff: SnapshotDiff
    stale: list[StaleAssessment]
    current: int


def stale_assessments(
    engine: Engine,
    entity_id: int,
    base_run_id: int | None = None,
    head_run_id: int | None = None,
) -> StalePlan:
    """Current assessments of ``entity_id`` citing a changed or removed URL; read-only.

    Runs :func:`diff_entity_runs` (same run selection and errors), then streams the entity's
    current assessments. Evidence without a URL is ignored; each stale assessment lists its
    triggering URLs once, sorted. Result sorted by criterion id then rubric version.

    SD-03: evidence cites a snapshot's ``final_url``, so an evidence URL matches a changed or
    removed URL when it equals that URL or its ``final_url`` in the base or head run (exact
    match only). The reported trigger is always the diffed snapshot ``url``.
    """
    diff, final_urls = _diff_with_final_urls(engine, entity_id, base_run_id, head_run_id)
    flagged = {d.url: d.status for d in diff.urls if d.status in STALE_STATUSES}
    trigger: dict[str, set[str]] = {}
    for url in flagged:
        for alias in {url} | final_urls.get(url, set()):
            trigger.setdefault(alias, set()).add(url)
    stale: list[StaleAssessment] = []
    current = 0
    for assessment in iter_current_assessments(engine, entity_id=entity_id):
        current += 1
        hits = sorted(
            {u for ev in assessment.evidence if ev.url is not None for u in trigger.get(ev.url, ())}
        )
        if hits:
            stale.append(
                StaleAssessment(
                    assessment_id=assessment.id,
                    criterion_id=assessment.criterion_id,
                    rubric_version=assessment.rubric_version,
                    urls=[UrlDiff(url=u, status=flagged[u]) for u in hits],
                )
            )
    stale.sort(key=lambda s: (s.criterion_id, s.rubric_version))
    return StalePlan(diff=diff, stale=stale, current=current)
