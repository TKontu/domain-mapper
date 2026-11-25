"""Register, tender and publication adapter interfaces (AD-04).

ARCHITECTURE.md §4 gives the pipeline five authoritative-source channels:
business registers for entity resolution and quantitative attributes,
accreditation registers for claim verification, person-level professional
registers, public tender award notices, and publication indexes. §10 requires
each to sit behind a reusable adapter so a scope pack selects implementations
per country or per site without changing ``core/``.

This module defines the interfaces only:

* :class:`BusinessRegister` — lookup by register id, search by name/country.
* :class:`AccreditationRegister` — entries for an organisation, including the
  ``active`` / ``suspended`` / ``withdrawn`` status and the scope text.
* :class:`ProfessionalRegister` — person-level entries limited to the
  ``listed_professional`` columns of §5, per the §9 limits.
* :class:`TenderSource` — award notices by classification codes, keywords and
  country.
* :class:`PublicationSource` — works by institution and topic.

Every record type is a frozen dataclass. ``InMemory*`` classes implement each
protocol from a fixed record list, for tests and the example scope; they are
not production sources and make no network calls.

Search methods return a bounded ``list`` in registration order: at most
``limit`` records (default :data:`DEFAULT_LIMIT`), never more. Filters of
different kinds combine with AND; values within one kind combine with OR. Text
filters are case-insensitive substring matches; classification codes match by
prefix so a coarse code selects its narrower children. ``lookup`` returns
``None`` when the id is absent.

Person-level data: :class:`ListedProfessional` deliberately carries only the
§5 ``listed_professional`` columns (minus the database surrogate id) and no
extra personal attributes.

The module is **unwired by design**. The first production callers are the
Discover, Resolve, Enrich and Claim verification rows, which will select an
adapter per source from the scope pack's catalogue.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Final, Literal, Protocol, runtime_checkable

DEFAULT_LIMIT: Final = 25

AccreditationStatus = Literal["active", "suspended", "withdrawn"]


class RegisterError(RuntimeError):
    """A register or source could not be read, or its records are inconsistent."""


@dataclass(frozen=True)
class BusinessRecord:
    """One legal entity as a business register publishes it."""

    register: str
    register_id: str
    name: str
    country: str
    founded: date | None
    employees_band: str | None
    status: str
    source_url: str


@dataclass(frozen=True)
class AccreditationRecord:
    """One accreditation entry, including suspended and withdrawn scopes."""

    register: str
    register_id: str
    organisation: str
    country: str | None
    status: AccreditationStatus
    scope_text: str
    valid_from: date | None
    valid_until: date | None
    source_url: str


@dataclass(frozen=True)
class ListedProfessional:
    """A person-level register entry, limited to the §5 columns.

    ``entity_id`` links the entry to a company entity when one exists; it is
    ``None`` for an unlinked professional.
    """

    name: str
    register: str
    register_id: str
    expertise_area: str | None
    register_url: str
    valid_until: date | None
    entity_id: str | None = None


@dataclass(frozen=True)
class TenderAward:
    """One awarded tender notice."""

    notice_id: str
    title: str
    buyer: str | None
    supplier: str | None
    country: str
    classification_codes: tuple[str, ...]
    award_date: date | None
    value: Decimal | None
    currency: str | None
    source_url: str


@dataclass(frozen=True)
class Publication:
    """One published work and the institutions/topics it is indexed under."""

    work_id: str
    title: str
    institutions: tuple[str, ...]
    topics: tuple[str, ...]
    published_on: date | None
    doi: str | None
    source_url: str


@runtime_checkable
class BusinessRegister(Protocol):
    """Business-register adapter; one implementation per register/country."""

    name: str

    def lookup(self, register_id: str) -> BusinessRecord | None: ...

    def search(
        self,
        name: str,
        *,
        country: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[BusinessRecord]: ...


@runtime_checkable
class AccreditationRegister(Protocol):
    """Accreditation-register adapter, including withdrawn and suspended entries."""

    name: str

    def lookup(self, register_id: str) -> AccreditationRecord | None: ...

    def entries_for(
        self,
        organisation: str,
        *,
        country: str | None = None,
        status: AccreditationStatus | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[AccreditationRecord]: ...


@runtime_checkable
class ProfessionalRegister(Protocol):
    """Person-level professional-register adapter (public register data only)."""

    name: str

    def lookup(self, register_id: str) -> ListedProfessional | None: ...

    def search(
        self,
        name: str | None = None,
        *,
        expertise_area: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[ListedProfessional]: ...


@runtime_checkable
class TenderSource(Protocol):
    """Tender-award adapter for one publication portal."""

    name: str

    def award_notices(
        self,
        *,
        classification_codes: Sequence[str] = (),
        keywords: Sequence[str] = (),
        country: str | None = None,
        since: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[TenderAward]: ...


@runtime_checkable
class PublicationSource(Protocol):
    """Publication-index adapter for one scholarly index."""

    name: str

    def works(
        self,
        *,
        institution: str | None = None,
        topics: Sequence[str] = (),
        since: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Publication]: ...


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _require_country(value: object, name: str) -> str | None:
    country = _optional_text(value, name)
    return None if country is None else country.upper()


def _require_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be an integer >= 1")
    return value


def _require_terms(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not a string")
    terms: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"{name} must contain only strings")
        if not item.strip():
            raise ValueError(f"{name} must not contain blank terms")
        terms.append(item)
    return terms


def _require_date(value: object, name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a date")
    return value


def _matches(haystack: str, needle: str) -> bool:
    return needle.casefold() in haystack.casefold()


def _require_status(value: object) -> AccreditationStatus:
    if value == "active":
        return "active"
    if value == "suspended":
        return "suspended"
    if value == "withdrawn":
        return "withdrawn"
    raise ValueError("status must be one of active, suspended, withdrawn")


class _InMemorySource[RecordT]:
    """Shared fixed-record store: insertion order, unique ids, no I/O."""

    name: str

    def __init__(
        self,
        name: str,
        records: Sequence[RecordT],
        *,
        identity: Callable[[RecordT], str],
        id_label: str,
    ) -> None:
        self.name = _require_text(name, "name")
        self._records: tuple[RecordT, ...] = tuple(records)
        self._by_id: dict[str, RecordT] = {}
        for record in self._records:
            key = _require_text(identity(record), id_label)
            if key in self._by_id:
                raise RegisterError(f"records repeat {id_label} {key!r}")
            self._by_id[key] = record

    def _get(self, register_id: str) -> RecordT | None:
        return self._by_id.get(_require_text(register_id, "register_id"))

    def _bounded(self, hits: list[RecordT], limit: int) -> list[RecordT]:
        return hits[:limit]


class InMemoryBusinessRegister(_InMemorySource[BusinessRecord]):
    """In-memory :class:`BusinessRegister` for tests and the example scope."""

    def __init__(
        self,
        records: Sequence[BusinessRecord],
        *,
        name: str = "in-memory-business-register",
    ) -> None:
        super().__init__(
            name,
            records,
            identity=lambda record: record.register_id,
            id_label="register_id",
        )

    def lookup(self, register_id: str) -> BusinessRecord | None:
        return self._get(register_id)

    def search(
        self,
        name: str,
        *,
        country: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[BusinessRecord]:
        needle = _require_text(name, "name")
        country_code = _require_country(country, "country")
        bound = _require_limit(limit)
        hits = [
            record
            for record in self._records
            if _matches(record.name, needle)
            and (country_code is None or record.country.upper() == country_code)
        ]
        return self._bounded(hits, bound)


class InMemoryAccreditationRegister(_InMemorySource[AccreditationRecord]):
    """In-memory :class:`AccreditationRegister` for tests and the example scope."""

    def __init__(
        self,
        records: Sequence[AccreditationRecord],
        *,
        name: str = "in-memory-accreditation-register",
    ) -> None:
        super().__init__(
            name,
            records,
            identity=lambda record: record.register_id,
            id_label="register_id",
        )

    def lookup(self, register_id: str) -> AccreditationRecord | None:
        return self._get(register_id)

    def entries_for(
        self,
        organisation: str,
        *,
        country: str | None = None,
        status: AccreditationStatus | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[AccreditationRecord]:
        organisation_name = _require_text(organisation, "organisation")
        country_code = _require_country(country, "country")
        wanted_status = None if status is None else _require_status(status)
        bound = _require_limit(limit)
        hits = [
            record
            for record in self._records
            if _matches(record.organisation, organisation_name)
            and (country_code is None or (record.country or "").upper() == country_code)
            and (wanted_status is None or record.status == wanted_status)
        ]
        return self._bounded(hits, bound)


class InMemoryProfessionalRegister(_InMemorySource[ListedProfessional]):
    """In-memory :class:`ProfessionalRegister` for tests and the example scope."""

    def __init__(
        self,
        records: Sequence[ListedProfessional],
        *,
        name: str = "in-memory-professional-register",
    ) -> None:
        super().__init__(
            name,
            records,
            identity=lambda record: record.register_id,
            id_label="register_id",
        )

    def lookup(self, register_id: str) -> ListedProfessional | None:
        return self._get(register_id)

    def search(
        self,
        name: str | None = None,
        *,
        expertise_area: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[ListedProfessional]:
        name_needle = _optional_text(name, "name")
        expertise_needle = _optional_text(expertise_area, "expertise_area")
        bound = _require_limit(limit)
        hits = [
            record
            for record in self._records
            if (name_needle is None or _matches(record.name, name_needle))
            and (
                expertise_needle is None
                or (
                    record.expertise_area is not None
                    and _matches(record.expertise_area, expertise_needle)
                )
            )
        ]
        return self._bounded(hits, bound)


class InMemoryTenderSource(_InMemorySource[TenderAward]):
    """In-memory :class:`TenderSource` for tests and the example scope."""

    def __init__(
        self,
        records: Sequence[TenderAward],
        *,
        name: str = "in-memory-tender-source",
    ) -> None:
        super().__init__(
            name,
            records,
            identity=lambda record: record.notice_id,
            id_label="notice_id",
        )

    def award_notices(
        self,
        *,
        classification_codes: Sequence[str] = (),
        keywords: Sequence[str] = (),
        country: str | None = None,
        since: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[TenderAward]:
        codes = _require_terms(classification_codes, "classification_codes")
        words = _require_terms(keywords, "keywords")
        country_code = _require_country(country, "country")
        start = None if since is None else _require_date(since, "since")
        bound = _require_limit(limit)

        hits: list[TenderAward] = []
        for record in self._records:
            if codes and not any(
                notice_code.casefold().startswith(code.casefold())
                for code in codes
                for notice_code in record.classification_codes
            ):
                continue
            if words:
                haystack = " ".join((record.title, record.buyer or "", record.supplier or ""))
                if not any(_matches(haystack, word) for word in words):
                    continue
            if country_code is not None and record.country.upper() != country_code:
                continue
            if start is not None and (record.award_date is None or record.award_date < start):
                continue
            hits.append(record)
        return self._bounded(hits, bound)


class InMemoryPublicationSource(_InMemorySource[Publication]):
    """In-memory :class:`PublicationSource` for tests and the example scope."""

    def __init__(
        self,
        records: Sequence[Publication],
        *,
        name: str = "in-memory-publication-source",
    ) -> None:
        super().__init__(
            name,
            records,
            identity=lambda record: record.work_id,
            id_label="work_id",
        )

    def works(
        self,
        *,
        institution: str | None = None,
        topics: Sequence[str] = (),
        since: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Publication]:
        institution_needle = _optional_text(institution, "institution")
        topic_terms = _require_terms(topics, "topics")
        start = None if since is None else _require_date(since, "since")
        bound = _require_limit(limit)

        hits: list[Publication] = []
        for record in self._records:
            if institution_needle is not None and not any(
                _matches(name, institution_needle) for name in record.institutions
            ):
                continue
            if topic_terms and not any(
                _matches(topic, term) for term in topic_terms for topic in record.topics
            ):
                continue
            if start is not None and (record.published_on is None or record.published_on < start):
                continue
            hits.append(record)
        return self._bounded(hits, bound)
