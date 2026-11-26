"""Tests for core.aggregate (AG-01): topic scores and coverage.

ARCHITECTURE.md §10: a topic score is the mean of mapped status values over applicable,
non-silent criteria; coverage is the share of applicable criteria with evidence. Both are
reported together. Fixtures are neutral inline models plus the ``_example`` pack (D-003).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.aggregate import TopicScore, score_topics
from core.assessment_store import CurrentAssessment, CurrentEvidence
from core.scope_pack import Criterion, Rubric, Status, Topic, load_scope_pack

EXAMPLE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


def status(status_id: str, value: float | None) -> Status:
    return Status(id=status_id, label=status_id.title(), value=value, color="grey")


SCALE = [
    status("strong", 3),
    status("standard", 2),
    status("weak", 1),
    status("silent", None),
    status("na", None),
]


def criterion(cid: str, *applies_to: str) -> Criterion:
    return Criterion(
        id=cid,
        question=f"Question for {cid}?",
        applies_to=list(applies_to) or ["agency"],
        statuses={},
    )


def topic(tid: str, *criteria: Criterion, segments: tuple[str, ...] = ()) -> Topic:
    return Topic(id=tid, label=tid.title(), segments=list(segments), criteria=list(criteria))


def rubric(*topics: Topic) -> Rubric:
    return Rubric(status_scale=SCALE, topics=list(topics))


def assessment(
    criterion_id: str,
    status_id: str,
    *grounded: bool,
    entity_id: int = 1,
) -> CurrentAssessment:
    return CurrentAssessment(
        id=1,
        entity_id=entity_id,
        criterion_id=criterion_id,
        status=status_id,
        confidence=None,
        rationale="",
        rubric_version="test",
        extractor_model="model",
        verifier_model="model",
        decided_by="auto",
        route_kind="accept",
        run_id=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        evidence=tuple(
            CurrentEvidence(chunk_id=i + 1, quote="quote", grounded=flag)
            for i, flag in enumerate(grounded)
        ),
    )


def test_score_is_mean_of_mapped_values() -> None:
    r = rubric(topic("quality", criterion("c1"), criterion("c2")))
    (score,) = score_topics(
        r, [assessment("c1", "strong"), assessment("c2", "standard")], ["agency"]
    )
    assert score == TopicScore(
        topic_id="quality", score=2.5, coverage=0.0, applicable=2, scored=2, evidenced=0
    )


def test_silent_excluded_from_score_but_counted_in_applicable_and_coverage() -> None:
    r = rubric(topic("quality", criterion("c1"), criterion("c2")))
    scores = score_topics(
        r,
        [assessment("c1", "strong", True), assessment("c2", "silent", True)],
        ["agency"],
    )
    assert scores == [
        TopicScore(topic_id="quality", score=3.0, coverage=1.0, applicable=2, scored=1, evidenced=2)
    ]


def test_na_status_removes_criterion_from_applicable() -> None:
    r = rubric(topic("quality", criterion("c1"), criterion("c2")))
    scores = score_topics(
        r,
        [assessment("c1", "na", True), assessment("c2", "strong", True)],
        ["agency"],
    )
    assert scores == [
        TopicScore(topic_id="quality", score=3.0, coverage=1.0, applicable=1, scored=1, evidenced=1)
    ]


def test_criterion_segment_filtering() -> None:
    r = rubric(
        topic(
            "quality",
            criterion("c1", "agency"),
            criterion("c2", "agency", "freelancer_network"),
        )
    )
    scores = score_topics(
        r,
        [assessment("c1", "strong"), assessment("c2", "standard")],
        ["freelancer_network"],
    )
    assert scores == [
        TopicScore(topic_id="quality", score=2.0, coverage=0.0, applicable=1, scored=1, evidenced=0)
    ]


def test_topic_segment_filtering() -> None:
    r = rubric(
        topic("agency_only", criterion("c1", "agency", "freelancer_network"), segments=("agency",)),
        topic("all", criterion("c2", "agency", "freelancer_network")),
    )
    scores = score_topics(
        r,
        [assessment("c1", "strong"), assessment("c2", "standard")],
        ["freelancer_network"],
    )
    assert [s.topic_id for s in scores] == ["agency_only", "all"]
    assert scores[0] == TopicScore(
        topic_id="agency_only", score=None, coverage=None, applicable=0, scored=0, evidenced=0
    )
    assert scores[1].score == 2.0


def test_unassessed_applicable_criterion_counts_only_in_applicable() -> None:
    r = rubric(topic("quality", criterion("c1"), criterion("c2")))
    scores = score_topics(r, [assessment("c1", "strong", True)], ["agency"])
    assert scores == [
        TopicScore(topic_id="quality", score=3.0, coverage=0.5, applicable=2, scored=1, evidenced=1)
    ]


def test_ungrounded_evidence_does_not_count_for_coverage() -> None:
    r = rubric(topic("quality", criterion("c1")))
    scores = score_topics(r, [assessment("c1", "strong", False, False)], ["agency"])
    assert scores == [
        TopicScore(topic_id="quality", score=3.0, coverage=0.0, applicable=1, scored=1, evidenced=0)
    ]


def test_zero_applicable_topic_has_no_score_and_no_coverage() -> None:
    r = rubric(topic("quality", criterion("c1", "freelancer_network")))
    scores = score_topics(r, [assessment("c1", "strong", True)], ["agency"])
    assert scores == [
        TopicScore(
            topic_id="quality", score=None, coverage=None, applicable=0, scored=0, evidenced=0
        )
    ]


def test_duplicate_assessment_for_one_criterion_raises() -> None:
    r = rubric(topic("quality", criterion("c1")))
    with pytest.raises(ValueError, match="c1"):
        score_topics(r, [assessment("c1", "strong"), assessment("c1", "weak")], ["agency"])


def test_unknown_status_raises_naming_criterion_and_status() -> None:
    r = rubric(topic("quality", criterion("c1")))
    with pytest.raises(ValueError, match="c1.*mystery|mystery.*c1"):
        score_topics(r, [assessment("c1", "mystery")], ["agency"])


def test_assessment_for_unknown_criterion_is_ignored() -> None:
    r = rubric(topic("quality", criterion("c1")))
    scores = score_topics(
        r,
        [assessment("c1", "strong"), assessment("ghost", "strong")],
        ["agency"],
    )
    assert scores == [
        TopicScore(topic_id="quality", score=3.0, coverage=0.0, applicable=1, scored=1, evidenced=0)
    ]


def test_topics_are_returned_in_rubric_order() -> None:
    r = rubric(topic("zeta", criterion("c1")), topic("alpha", criterion("c2")))
    scores = score_topics(r, [], ["agency"])
    assert [s.topic_id for s in scores] == ["zeta", "alpha"]


def test_example_pack_rubric_scores_agency() -> None:
    r = load_scope_pack(EXAMPLE).rubric
    scores = score_topics(
        r,
        [assessment("iso17100", "strong", True), assessment("public_terms", "standard", True)],
        ["agency"],
    )
    assert [s.topic_id for s in scores] == ["certification", "terms"]
    assert scores[0].score == 3.0
    assert scores[1].score == 2.0
    assert all(s.coverage == 1.0 for s in scores)
