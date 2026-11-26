"""Topic scores and coverage over current assessments (ARCHITECTURE.md §10).

One :class:`TopicScore` per rubric topic, in rubric order. A criterion is *applicable* to an
entity when its topic applies to one of the entity's segments (an empty topic segment list means
all segments), its own ``applies_to`` intersects the entity's segments, and its current
assessment is not an explicit "does not apply" (status id ``na`` with a null scale value).

* ``scored`` counts applicable criteria whose assessment status maps to a non-null value; the
  topic ``score`` is the mean of those values. ``silent`` (or any other null-valued status that is
  not ``na``) stays applicable but is excluded from the score.
* ``evidenced`` counts applicable criteria whose assessment has at least one grounded evidence
  item; ``coverage`` is ``evidenced / applicable``. Both figures are always reported together.

Callers pass the current assessments for one entity (normally one rubric version, so at most one
assessment per criterion id); a duplicate criterion id or a status outside ``rubric.status_scale``
raises :class:`ValueError`. Assessments for criterion ids outside the rubric are ignored. Silent
penalties and profile weights are later rows.

Unwired by design: the production caller is the later "scores CLI" row, which feeds
``iter_current_assessments`` for one entity. This loop consumes that iterable once and holds only
the per-criterion state of the rubric (bounded by rubric size).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Final

from core.assessment_store import CurrentAssessment
from core.scope_pack import Rubric

NA: Final = "na"
"""Status id meaning "does not apply" when its scale value is null."""


@dataclass(frozen=True)
class TopicScore:
    """Score and coverage of one rubric topic for one entity."""

    topic_id: str
    score: float | None
    coverage: float | None
    applicable: int
    scored: int
    evidenced: int


def score_topics(
    rubric: Rubric,
    assessments: Iterable[CurrentAssessment],
    entity_segments: Collection[str],
) -> list[TopicScore]:
    """Return one :class:`TopicScore` per rubric topic, in rubric order."""
    values = {status.id: status.value for status in rubric.status_scale}
    criteria = {criterion.id for topic in rubric.topics for criterion in topic.criteria}
    current: dict[str, CurrentAssessment] = {}
    for item in assessments:
        if item.criterion_id not in criteria:
            continue
        if item.criterion_id in current:
            raise ValueError(f"duplicate assessment for criterion {item.criterion_id!r}")
        if item.status not in values:
            raise ValueError(f"criterion {item.criterion_id!r} has unknown status {item.status!r}")
        current[item.criterion_id] = item

    segments = set(entity_segments)
    scores: list[TopicScore] = []
    for topic in rubric.topics:
        applicable = scored = evidenced = 0
        total = 0.0
        if not topic.segments or segments.intersection(topic.segments):
            for criterion in topic.criteria:
                if not segments.intersection(criterion.applies_to):
                    continue
                assessed = current.get(criterion.id)
                if (
                    assessed is not None
                    and assessed.status == NA
                    and values[assessed.status] is None
                ):
                    continue
                applicable += 1
                if assessed is None:
                    continue
                value = values[assessed.status]
                if value is not None:
                    scored += 1
                    total += value
                if any(evidence.grounded for evidence in assessed.evidence):
                    evidenced += 1
        scores.append(
            TopicScore(
                topic_id=topic.id,
                score=total / scored if scored else None,
                coverage=evidenced / applicable if applicable else None,
                applicable=applicable,
                scored=scored,
                evidenced=evidenced,
            )
        )
    return scores
