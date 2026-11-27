"""Domain normalisation and candidate grouping for the Resolve stage (RV-01).

ARCHITECTURE.md §1 Resolve normalises domains before candidates are matched to entities.
:func:`normalize_domain` maps one observed URL to a lowercase ASCII domain (or ``None``);
:func:`group_candidates` streams unresolved ``candidate`` rows and counts them per normalised
domain for ``landscape resolve``.

Semantics:

* Only ``http``/``https`` URLs normalise; anything unparsable, another scheme or a junk host
  becomes ``None``.
* The host is lowercased and IDNA-encoded to ASCII, a trailing dot and a leading ``www.`` are
  stripped, and the port, path, query, fragment and userinfo are dropped. IP literals are kept
  as-is (``127.0.0.1``, ``::1``).
* Only rows with ``resolved_entity_id IS NULL`` are read; the first row seen per domain supplies
  the group's name. :func:`group_candidates` writes nothing.
* :func:`apply_resolution` (RV-02) writes: one ``entity`` per ``(scope_id, normalised domain)``
  (reused if it already exists, else created with the first-found name and status
  ``unscreened``) and sets ``candidate.resolved_entity_id``. Each batch of candidates is one
  transaction; a failure rolls back that batch only and re-raises. Unnormalisable URLs are
  counted as skipped and stay NULL, so a rerun reports them again and creates/links nothing new.

Production caller: ``landscape resolve`` (``app.cli.run_resolve``). Consumer: the future Resolve
merge-suggestion row (unwired by design). Rows are streamed with ``yield_per``, so memory is
bounded by the number of distinct domains, not by the number of rows.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.candidate_store import Candidate
from core.entity_store import STATUS_UNSCREENED, Entity

DEFAULT_BATCH = 500
"""Rows fetched per page by :func:`group_candidates`."""

_HTTP_SCHEMES = frozenset({"http", "https"})
_WWW_PREFIX = "www."
# Post-IDNA host: letters/digits, dot, hyphen, underscore (internal names) and colon for IPv6.
_HOST_RE = re.compile(r"^[a-z0-9._:-]+$")


@dataclass(frozen=True, slots=True)
class DomainGroup:
    """Unresolved candidates that share one normalised domain."""

    domain: str
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class ResolveSummary:
    """Grouped unresolved candidates; ``candidates`` is the sum of the group counts."""

    groups: tuple[DomainGroup, ...]
    candidates: int
    skipped: int


@dataclass(frozen=True, slots=True)
class ApplySummary:
    """Outcome of :func:`apply_resolution`."""

    created: int
    linked: int
    skipped: int


def normalize_domain(url: str) -> str | None:
    """Return the normalised domain of an ``http(s)`` URL, or ``None``.

    ``HTTPS://WWW.Example.COM:8443/a?b`` → ``example.com``; ``http://[::1]/`` → ``::1``.
    An invalid port (non-numeric or outside 0-65535) or a host that is exactly ``www`` gives
    ``None``.
    """
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError for a non-numeric or out-of-range port
    except ValueError:  # e.g. an unterminated IPv6 bracket or an invalid port
        return None
    if scheme not in _HTTP_SCHEMES or not host:
        return None
    host = host.lower().rstrip(".")
    if not host or host == "www":
        return None
    if not _is_ip_literal(host):
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None
    if not _HOST_RE.match(host):
        return None
    if host.startswith(_WWW_PREFIX):
        host = host[len(_WWW_PREFIX) :]
    return host or None


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def group_candidates(engine: Engine, *, batch: int = DEFAULT_BATCH) -> ResolveSummary:
    """Count unresolved candidates per normalised domain, sorted by count then domain.

    Rows stream in insertion order with ``yield_per(batch)``; ``skipped`` counts rows whose URL
    does not normalise. Read-only: no INSERT/UPDATE/DELETE and no commit.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    skipped = 0
    stmt = (
        sa.select(Candidate.name, Candidate.url)
        .where(Candidate.resolved_entity_id.is_(None))
        .order_by(Candidate.id)
    )
    with Session(engine) as session:
        for name, url in session.execute(stmt).yield_per(batch):
            domain = normalize_domain(url)
            if domain is None:
                skipped += 1
                continue
            if domain in counts:
                counts[domain] += 1
            else:
                counts[domain] = 1
                names[domain] = name
    groups = tuple(
        DomainGroup(domain=domain, name=names[domain], count=count)
        for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return ResolveSummary(groups=groups, candidates=sum(counts.values()), skipped=skipped)


def apply_resolution(
    engine: Engine, scope_id: str, *, batch: int = DEFAULT_BATCH, run_id: int | None = None
) -> ApplySummary:
    """Link unresolved candidates to per-domain entities of ``scope_id``; create missing ones.

    Candidates are paged in id order by keyset (``id > last``, ``LIMIT batch``) with each page
    read, resolved and committed in its own transaction; the read uses ``yield_per(batch)``.
    Memory holds only the committed ``domain -> entity_id`` map (bounded by distinct domains).
    Newly created entities get ``created_run = run_id``; reused entities are not modified.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    known: dict[str, int] = {}
    created = linked = skipped = 0
    last_id = 0
    while True:
        stmt = (
            sa.select(Candidate.id, Candidate.name, Candidate.url)
            .where(Candidate.resolved_entity_id.is_(None), Candidate.id > last_id)
            .order_by(Candidate.id)
            .limit(batch)
        )
        with Session(engine) as session, session.begin():
            rows = list(session.execute(stmt).yield_per(batch))
            if not rows:
                break
            last_id = rows[-1][0]
            new_known: dict[str, int] = {}
            by_entity: dict[int, list[int]] = {}
            batch_created = batch_skipped = 0
            for cand_id, name, url in rows:
                domain = normalize_domain(url)
                if domain is None:
                    batch_skipped += 1
                    continue
                entity_id = known.get(domain) or new_known.get(domain)
                if entity_id is None:
                    entity_id = session.scalar(
                        sa.select(Entity.id).where(
                            Entity.scope_id == scope_id, Entity.primary_domain == domain
                        )
                    )
                    if entity_id is None:
                        entity = Entity(
                            scope_id=scope_id,
                            name=name,
                            primary_domain=domain,
                            status=STATUS_UNSCREENED,
                            created_run=run_id,
                        )
                        session.add(entity)
                        session.flush()
                        entity_id = entity.id
                        batch_created += 1
                    new_known[domain] = entity_id
                by_entity.setdefault(entity_id, []).append(cand_id)
            batch_linked = sum(_link(session, eid, ids) for eid, ids in by_entity.items())
        # Only reached after the batch committed.
        known.update(new_known)
        created += batch_created
        linked += batch_linked
        skipped += batch_skipped
    return ApplySummary(created=created, linked=linked, skipped=skipped)


def _link(session: Session, entity_id: int, candidate_ids: list[int]) -> int:
    """Set ``resolved_entity_id`` on still-unresolved ``candidate_ids``; return rows updated."""
    result = session.execute(
        sa.update(Candidate)
        .where(Candidate.id.in_(candidate_ids), Candidate.resolved_entity_id.is_(None))
        .values(resolved_entity_id=entity_id)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount)  # type: ignore[attr-defined]
