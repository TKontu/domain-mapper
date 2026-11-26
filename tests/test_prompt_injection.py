"""Prompt-injection corpus against both extractors' prompts and checks (PI-01).

(a) every case lands inside the single evidence block, with exactly one real BEGIN and END
delimiter; (b) a fake LLM that obeys the injection fails the deterministic checks and, for
attributes, nothing is stored.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from core import assess, attribute_extract
from core.assess import EVIDENCE_BEGIN, EVIDENCE_END
from core.assessment_store import EvidenceRow
from core.attribute_store import AttributeValueRow
from core.runs import create_all
from core.scope_pack import Attribute
from tests.injection_cases import ATTACK_QUOTE, ATTACK_VALUE, CASE_IDS, CASES
from tests.test_assess import FakeLLM, _rubric

ATTR = Attribute(id="headcount", label="Headcount", type="int")
INJECTED_ID, HONEST_ID = 11, 12
HONEST = (HONEST_ID, "We answer within two business days.")


def _chunks(page: str) -> list[tuple[int, str]]:
    return [(INJECTED_ID, page), HONEST]


def _prompts(page: str) -> list[str]:
    rubric, crit = _rubric()
    return [
        assess.build_prompt(rubric, crit, _chunks(page)),
        attribute_extract.build_prompt(ATTR, _chunks(page)),
    ]


@pytest.mark.parametrize("page", [p for _, p in CASES], ids=CASE_IDS)
def test_case_stays_inside_single_evidence_block(page: str) -> None:
    for prompt in _prompts(page):
        assert prompt.count(EVIDENCE_BEGIN) == 1
        assert prompt.count(EVIDENCE_END) == 1
        assert prompt.lower().count("<<<begin evidence") == 1
        assert prompt.lower().count("<<<end evidence>>>") == 1
        start = prompt.index(EVIDENCE_BEGIN)
        end = prompt.index(EVIDENCE_END)
        assert start < end and prompt.rstrip().endswith(EVIDENCE_END)
        block = prompt[start:end]
        assert assess._neutralise(page) in block
        assert "<<<" not in block[len(EVIDENCE_BEGIN) :]
        # The instructions precede the block; nothing from the page appears before it.
        assert "data, not instructions" in prompt[:start]
        assert assess._neutralise(page) not in prompt[:start]


def _obeying_quotes() -> list[list[dict[str, Any]]]:
    return [
        # attacker's quote cited to the injected chunk: not verbatim there
        [{"chunk_id": INJECTED_ID, "quote": ATTACK_QUOTE + " Extra words."}],
        # attacker's quote cited to the chunk id a forged header claims
        [{"chunk_id": HONEST_ID, "quote": ATTACK_QUOTE}],
    ]


QUOTE_IDS = ["unverbatim_quote", "forged_header_chunk_id"]


def _assess_run(page: str, quotes: list[dict[str, Any]]) -> assess.Extraction:
    rubric, crit = _rubric()
    llm = FakeLLM({"status": "strong", "quotes": quotes, "rationale": "as instructed"})
    return assess.extract_criterion(llm, rubric, crit, _chunks(page), run_id=1)


def _attr_run(page: str, quotes: list[dict[str, Any]]) -> attribute_extract.AttributeExtraction:
    llm = FakeLLM({"value": ATTACK_VALUE, "quotes": quotes})
    return attribute_extract.extract_attribute(llm, ATTR, _chunks(page), run_id=1)


@pytest.fixture
def engine():
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.mark.parametrize("quotes", _obeying_quotes(), ids=QUOTE_IDS)
@pytest.mark.parametrize("page", [p for _, p in CASES], ids=CASE_IDS)
def test_obeying_assessment_fails_checks(page: str, quotes: list[dict[str, Any]]) -> None:
    ex = _assess_run(page, quotes)
    assert ex.status == "strong"
    assert not ex.checks_passed
    assert not any(q.grounded for q in ex.quotes)


@pytest.mark.parametrize("quotes", _obeying_quotes(), ids=QUOTE_IDS)
@pytest.mark.parametrize("page", [p for _, p in CASES], ids=CASE_IDS)
def test_obeying_attribute_fails_checks_and_stores_nothing(
    page: str, quotes: list[dict[str, Any]], engine
) -> None:
    ex = _attr_run(page, quotes)
    assert ex.value == ATTACK_VALUE
    assert not ex.checks_passed
    with Session(engine) as session, session.begin():
        assert attribute_extract.store_attribute_extraction(session, 7, ATTR, ex) is None
    with Session(engine) as session:
        assert session.scalar(sa.select(sa.func.count(AttributeValueRow.id))) == 0
        assert session.scalar(sa.select(sa.func.count(EvidenceRow.id))) == 0
