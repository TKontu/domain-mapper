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
  the group's name. Nothing is written.

Production caller: ``landscape resolve`` (``core.cli.run_resolve``). Consumer: the future Resolve
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


def normalize_domain(url: str) -> str | None:
    """Return the normalised domain of an ``http(s)`` URL, or ``None``.

    ``HTTPS://WWW.Example.COM:8443/a?b`` → ``example.com``; ``http://[::1]/`` → ``::1``.
    """
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        host = parts.hostname
    except ValueError:  # e.g. an unterminated IPv6 bracket
        return None
    if scheme not in _HTTP_SCHEMES or not host:
        return None
    host = host.lower().rstrip(".")
    if not host:
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
