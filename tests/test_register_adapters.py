"""Tests for the register, tender and publication adapter interfaces (AD-04).

The suite is offline: it exercises the protocols, their frozen record types and
the in-memory implementations only. No register, tender portal or publication
index is contacted.
"""

from __future__ import annotations

import unittest
from dataclasses import fields
from datetime import date
from decimal import Decimal

from adapters.registers import (
    DEFAULT_LIMIT,
    AccreditationRecord,
    AccreditationRegister,
    BusinessRecord,
    BusinessRegister,
    InMemoryAccreditationRegister,
    InMemoryBusinessRegister,
    InMemoryProfessionalRegister,
    InMemoryPublicationSource,
    InMemoryTenderSource,
    ListedProfessional,
    ProfessionalRegister,
    Publication,
    PublicationSource,
    RegisterError,
    TenderAward,
    TenderSource,
)

BUSINESS = (
    BusinessRecord(
        register_id="A-1",
        register="trade-register-a",
        name="Acme Analytics",
        country="DE",
        founded=date(2001, 4, 2),
        employees_band="10-49",
        status="active",
        source_url="https://register-a.example/company/A-1",
    ),
    BusinessRecord(
        register_id="A-2",
        register="trade-register-a",
        name="Northwind Labs",
        country="DE",
        founded=date(2015, 9, 1),
        employees_band="1-9",
        status="active",
        source_url="https://register-a.example/company/A-2",
    ),
    BusinessRecord(
        register_id="B-1",
        register="trade-register-b",
        name="Acme Analytics GmbH",
        country="FR",
        founded=None,
        employees_band=None,
        status="dissolved",
        source_url="https://register-b.example/company/B-1",
    ),
)

ACCREDITATIONS = (
    AccreditationRecord(
        register_id="ACC-1",
        register="accreditation-register-a",
        organisation="Acme Analytics",
        country="DE",
        status="active",
        scope_text="Inspection of measurement equipment",
        valid_from=date(2023, 1, 1),
        valid_until=date(2026, 1, 1),
        source_url="https://accreditation.example/ACC-1",
    ),
    AccreditationRecord(
        register_id="ACC-2",
        register="accreditation-register-a",
        organisation="Northwind Labs",
        country="DE",
        status="suspended",
        scope_text="Calibration of reference weights",
        valid_from=date(2022, 5, 1),
        valid_until=date(2025, 5, 1),
        source_url="https://accreditation.example/ACC-2",
    ),
    AccreditationRecord(
        register_id="ACC-3",
        register="accreditation-register-a",
        organisation="Acme Analytics",
        country="FR",
        status="withdrawn",
        scope_text="Inspection of pressure vessels",
        valid_from=date(2018, 1, 1),
        valid_until=date(2021, 1, 1),
        source_url="https://accreditation.example/ACC-3",
    ),
)

PROFESSIONALS = (
    ListedProfessional(
        name="Jordan Example",
        register="professional-register-a",
        register_id="P-1",
        expertise_area="auditing",
        register_url="https://professionals.example/P-1",
        valid_until=date(2027, 3, 31),
        entity_id="ent-1",
    ),
    ListedProfessional(
        name="Robin Sample",
        register="professional-register-a",
        register_id="P-2",
        expertise_area="tax advice",
        register_url="https://professionals.example/P-2",
        valid_until=None,
    ),
)

TENDERS = (
    TenderAward(
        notice_id="T-1",
        title="Market research services",
        buyer="City of Example",
        supplier="Acme Analytics",
        country="DE",
        classification_codes=("79310000", "79000000"),
        award_date=date(2024, 5, 1),
        value=Decimal("12000.50"),
        currency="EUR",
        source_url="https://tenders.example/T-1",
    ),
    TenderAward(
        notice_id="T-2",
        title="Software licence renewal",
        buyer="Example Regional Authority",
        supplier="Northwind Labs",
        country="FR",
        classification_codes=("48000000",),
        award_date=date(2023, 2, 1),
        value=None,
        currency=None,
        source_url="https://tenders.example/T-2",
    ),
    TenderAward(
        notice_id="T-3",
        title="Advertising campaign",
        buyer="City of Example",
        supplier=None,
        country="DE",
        classification_codes=("79310000",),
        award_date=date(2022, 7, 1),
        value=Decimal("8000"),
        currency="EUR",
        source_url="https://tenders.example/T-3",
    ),
)

PUBLICATIONS = (
    Publication(
        work_id="W-1",
        title="Measuring service quality",
        institutions=("Acme Analytics", "Example University"),
        topics=("market research", "statistics"),
        published_on=date(2024, 6, 1),
        doi="10.0000/example.1",
        source_url="https://publications.example/W-1",
    ),
    Publication(
        work_id="W-2",
        title="Maintaining software systems",
        institutions=("Northwind Labs",),
        topics=("software",),
        published_on=date(2022, 1, 1),
        doi=None,
        source_url="https://publications.example/W-2",
    ),
)


class ProtocolConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.business = InMemoryBusinessRegister(BUSINESS)
        self.accreditation = InMemoryAccreditationRegister(ACCREDITATIONS)
        self.professional = InMemoryProfessionalRegister(PROFESSIONALS)
        self.tender = InMemoryTenderSource(TENDERS)
        self.publication = InMemoryPublicationSource(PUBLICATIONS)

    def test_each_fake_satisfies_its_runtime_protocol(self) -> None:
        self.assertIsInstance(self.business, BusinessRegister)
        self.assertIsInstance(self.accreditation, AccreditationRegister)
        self.assertIsInstance(self.professional, ProfessionalRegister)
        self.assertIsInstance(self.tender, TenderSource)
        self.assertIsInstance(self.publication, PublicationSource)

    def test_a_missing_lookup_method_does_not_satisfy_the_protocol(self) -> None:
        class NameOnly:
            name = "not-a-register"

        self.assertNotIsInstance(NameOnly(), BusinessRegister)
        self.assertNotIsInstance(NameOnly(), PublicationSource)

    def test_fakes_expose_a_source_name(self) -> None:
        for source in (self.business, self.accreditation, self.professional):
            self.assertIsInstance(source.name, str)
            self.assertTrue(source.name)
        self.assertEqual(self.business.name, "in-memory-business-register")
        self.assertEqual(self.accreditation.name, "in-memory-accreditation-register")
        self.assertEqual(self.professional.name, "in-memory-professional-register")
        self.assertEqual(self.tender.name, "in-memory-tender-source")
        self.assertEqual(self.publication.name, "in-memory-publication-source")

    def test_records_are_frozen(self) -> None:
        records = (
            BUSINESS[0],
            ACCREDITATIONS[0],
            PROFESSIONALS[0],
            TENDERS[0],
            PUBLICATIONS[0],
        )
        for record in records:
            field_name = fields(record)[0].name
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(Exception):
                    setattr(record, field_name, "changed")

    def test_record_collections_are_tuples(self) -> None:
        award = TENDERS[0]
        work = PUBLICATIONS[0]
        self.assertIsInstance(award.classification_codes, tuple)
        self.assertIsInstance(work.institutions, tuple)
        self.assertIsInstance(work.topics, tuple)

    def test_listed_professional_carries_only_the_section_5_columns(self) -> None:
        # ARCHITECTURE.md §5: listed_professional(id, entity_id NULL, name, register,
        # register_id, expertise_area, register_url, valid_until). The DB surrogate
        # `id` is omitted; no other person-level field is present (§9).
        self.assertEqual(
            {field.name for field in fields(ListedProfessional)},
            {
                "entity_id",
                "name",
                "register",
                "register_id",
                "expertise_area",
                "register_url",
                "valid_until",
            },
        )


class BusinessRegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.register = InMemoryBusinessRegister(BUSINESS)

    def test_lookup_by_register_id_returns_every_field(self) -> None:
        self.assertEqual(self.register.lookup("A-1"), BUSINESS[0])
        self.assertIsNone(self.register.lookup("missing"))
        self.assertEqual(self.register.lookup("B-1").founded, None)  # type: ignore[union-attr]

    def test_search_by_name_is_case_insensitive_substring(self) -> None:
        hits = self.register.search("acme")
        self.assertEqual([hit.register_id for hit in hits], ["A-1", "B-1"])

    def test_search_by_country_narrows_the_hits(self) -> None:
        self.assertEqual(
            [hit.register_id for hit in self.register.search("acme", country="DE")], ["A-1"]
        )
        self.assertEqual(self.register.search("acme", country="de")[0].country, "DE")

    def test_search_respects_limit_and_default_bound(self) -> None:
        self.assertEqual(len(self.register.search("a", limit=1)), 1)
        filler = [
            BusinessRecord(
                register_id=f"F-{n}",
                register="trade-register-a",
                name=f"Filler {n}",
                country="DE",
                founded=None,
                employees_band=None,
                status="active",
                source_url=f"https://register-a.example/company/F-{n}",
            )
            for n in range(DEFAULT_LIMIT + 5)
        ]
        bounded = InMemoryBusinessRegister(filler)
        self.assertEqual(len(bounded.search("filler")), DEFAULT_LIMIT)

    def test_lookup_and_search_reject_blank_identifiers(self) -> None:
        for bad in ("", "  "):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.register.lookup(bad)
                with self.assertRaises(ValueError):
                    self.register.search(bad)

    def test_search_rejects_invalid_limit(self) -> None:
        for bad in (0, -1, True, 2.5, "2", None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.register.search("acme", limit=bad)  # type: ignore[arg-type]

    def test_duplicate_register_ids_are_rejected(self) -> None:
        with self.assertRaises(RegisterError):
            InMemoryBusinessRegister((BUSINESS[0], BUSINESS[0]))


class AccreditationRegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.register = InMemoryAccreditationRegister(ACCREDITATIONS)

    def test_lookup_keeps_status_and_scope_text(self) -> None:
        active = self.register.lookup("ACC-1")
        self.assertEqual(active.status, "active")  # type: ignore[union-attr]
        self.assertEqual(active.scope_text, "Inspection of measurement equipment")  # type: ignore[union-attr]
        self.assertEqual(active.valid_until, date(2026, 1, 1))  # type: ignore[union-attr]
        self.assertIsNone(self.register.lookup("missing"))

    def test_entries_for_returns_every_status(self) -> None:
        statuses = [entry.status for entry in self.register.entries_for("acme")]
        self.assertEqual(statuses, ["active", "withdrawn"])

    def test_entries_for_filters_by_status_and_country(self) -> None:
        active = self.register.entries_for("acme", status="active")
        self.assertEqual([entry.register_id for entry in active], ["ACC-1"])
        suspended = self.register.entries_for("northwind", status="suspended")
        self.assertEqual([entry.register_id for entry in suspended], ["ACC-2"])
        active = self.register.entries_for("acme", country="DE")
        self.assertEqual([entry.register_id for entry in active], ["ACC-1"])

    def test_entries_for_respects_limit(self) -> None:
        self.assertEqual(len(self.register.entries_for("acme", limit=1)), 1)

    def test_entries_for_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self.register.entries_for("acme", limit=0)
        with self.assertRaises(ValueError):
            self.register.lookup(" ")

    def test_duplicate_register_ids_are_rejected(self) -> None:
        with self.assertRaises(RegisterError):
            InMemoryAccreditationRegister((ACCREDITATIONS[0], ACCREDITATIONS[0]))


class ProfessionalRegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.register = InMemoryProfessionalRegister(PROFESSIONALS)

    def test_lookup_returns_the_listed_columns(self) -> None:
        self.assertEqual(self.register.lookup("P-1"), PROFESSIONALS[0])
        self.assertIsNone(self.register.lookup("missing"))

    def test_entity_id_defaults_to_none(self) -> None:
        self.assertIsNone(self.register.lookup("P-2").entity_id)  # type: ignore[union-attr]

    def test_search_by_name_and_expertise_area(self) -> None:
        self.assertEqual([entry.register_id for entry in self.register.search("jordan")], ["P-1"])
        self.assertEqual(
            [entry.register_id for entry in self.register.search(expertise_area="tax")],
            ["P-2"],
        )

    def test_search_respects_limit(self) -> None:
        self.assertEqual(len(self.register.search(limit=1)), 1)

    def test_duplicate_register_ids_are_rejected(self) -> None:
        with self.assertRaises(RegisterError):
            InMemoryProfessionalRegister((PROFESSIONALS[0], PROFESSIONALS[0]))


class TenderSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = InMemoryTenderSource(TENDERS)

    def test_classification_codes_match_by_prefix(self) -> None:
        hits = self.source.award_notices(classification_codes=["79"])
        self.assertEqual([hit.notice_id for hit in hits], ["T-1", "T-3"])

    def test_keywords_match_title_buyer_or_supplier(self) -> None:
        self.assertEqual(
            [hit.notice_id for hit in self.source.award_notices(keywords=["research"])],
            ["T-1"],
        )
        self.assertEqual(
            [hit.notice_id for hit in self.source.award_notices(keywords=["northwind"])],
            ["T-2"],
        )

    def test_filters_combine_across_kinds_and_any_value_within_a_kind(self) -> None:
        self.assertEqual(
            [
                hit.notice_id
                for hit in self.source.award_notices(
                    classification_codes=["79"], keywords=["campaign"]
                )
            ],
            ["T-3"],
        )
        self.assertEqual(
            [
                hit.notice_id
                for hit in self.source.award_notices(classification_codes=["48", "7931"])
            ],
            ["T-1", "T-2", "T-3"],
        )

    def test_country_and_since_filters(self) -> None:
        self.assertEqual(
            [hit.notice_id for hit in self.source.award_notices(country="fr")],
            ["T-2"],
        )
        self.assertEqual(
            [hit.notice_id for hit in self.source.award_notices(since=date(2023, 1, 1))],
            ["T-1", "T-2"],
        )

    def test_empty_filters_match_all_and_limit_bounds_the_result(self) -> None:
        self.assertEqual(len(self.source.award_notices()), 3)
        self.assertEqual(len(self.source.award_notices(limit=2)), 2)

    def test_award_values_are_decimal_or_none(self) -> None:
        self.assertEqual(
            self.source.award_notices(keywords=["research"])[0].value, Decimal("12000.50")
        )
        self.assertIsNone(self.source.award_notices(keywords=["licence"])[0].value)

    def test_rejects_blank_terms_and_bare_strings(self) -> None:
        with self.assertRaises(TypeError):
            self.source.award_notices(classification_codes="79")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.source.award_notices(keywords="research")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.source.award_notices(keywords=[" "])
        with self.assertRaises(ValueError):
            self.source.award_notices(limit=0)
        with self.assertRaises((TypeError, ValueError)):
            self.source.award_notices(since="2023-01-01")  # type: ignore[arg-type]

    def test_duplicate_notice_ids_are_rejected(self) -> None:
        with self.assertRaises(RegisterError):
            InMemoryTenderSource((TENDERS[0], TENDERS[0]))


class PublicationSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = InMemoryPublicationSource(PUBLICATIONS)

    def test_works_filters_by_institution_substring(self) -> None:
        self.assertEqual([work.work_id for work in self.source.works(institution="acme")], ["W-1"])

    def test_works_filters_by_topic(self) -> None:
        self.assertEqual(
            [work.work_id for work in self.source.works(topics=["statistics"])],
            ["W-1"],
        )
        self.assertEqual(
            [work.work_id for work in self.source.works(institution="labs", topics=["software"])],
            ["W-2"],
        )

    def test_works_requires_an_institution_and_supports_since(self) -> None:
        with self.assertRaises(ValueError):
            self.source.works(institution=" ")
        self.assertEqual(
            [work.work_id for work in self.source.works(since=date(2023, 1, 1))],
            ["W-1"],
        )

    def test_works_respects_limit(self) -> None:
        self.assertEqual(len(self.source.works(limit=1)), 1)

    def test_works_rejects_a_bare_topic_string(self) -> None:
        with self.assertRaises(TypeError):
            self.source.works(institution="acme", topics="statistics")  # type: ignore[arg-type]

    def test_duplicate_work_ids_are_rejected(self) -> None:
        with self.assertRaises(RegisterError):
            InMemoryPublicationSource((PUBLICATIONS[0], PUBLICATIONS[0]))


if __name__ == "__main__":
    unittest.main()
