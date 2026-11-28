"""Per-criterion evaluation metrics for auto vs. human status labels (EV-01).

Implements the metrics of ARCHITECTURE.md §7 over status ids from §3.2: n, accuracy,
unweighted Cohen's kappa, silent-rate (share of auto labels that are ``silent``) and
false-"strong" rate (share of pairs where auto is the top status but human is not).

Unwired by design: the future eval-harness CLI / gold-set loader (EV-02) calls
:func:`compute_metrics`. The input iterable is consumed once; memory is
O(criteria x statuses) counts, never the whole list.

Kappa convention: when expected agreement is 1 (every label is one class), kappa is
undefined in the textbook formula; we return 1.0 if observed agreement is also 1, else 0.0.
Empty groups yield ``n=0`` and ``None`` for every metric.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

OVERALL = "__overall__"


@dataclass(frozen=True)
class CriterionMetrics:
    criterion_id: str
    n: int
    accuracy: float | None
    kappa: float | None
    silent_rate: float | None
    false_strong_rate: float | None


@dataclass
class _Counts:
    n: int = 0
    agree: int = 0
    silent: int = 0
    false_strong: int = 0
    auto: Counter[str] = field(default_factory=Counter)
    human: Counter[str] = field(default_factory=Counter)

    def add(self, auto: str, human: str, top_status: str, silent_status: str) -> None:
        self.n += 1
        self.agree += auto == human
        self.silent += auto == silent_status
        self.false_strong += auto == top_status and human != top_status
        self.auto[auto] += 1
        self.human[human] += 1

    def finish(self, criterion_id: str) -> CriterionMetrics:
        if self.n == 0:
            return CriterionMetrics(criterion_id, 0, None, None, None, None)
        n = self.n
        p_o = self.agree / n
        p_e = sum(self.auto[k] * self.human[k] for k in self.auto) / (n * n)
        if p_e >= 1.0:
            kappa = 1.0 if p_o >= 1.0 else 0.0
        else:
            kappa = (p_o - p_e) / (1.0 - p_e)
        return CriterionMetrics(criterion_id, n, p_o, kappa, self.silent / n, self.false_strong / n)


def compute_metrics(
    pairs: Iterable[tuple[str, str, str]],
    *,
    top_status: str = "strong",
    silent_status: str = "silent",
) -> list[CriterionMetrics]:
    """Compute metrics from ``(criterion_id, auto_status, human_status)`` triples.

    Returns one row per criterion sorted by id, followed by an ``OVERALL`` row.
    """
    groups: dict[str, _Counts] = {}
    overall = _Counts()
    for criterion_id, auto, human in pairs:
        groups.setdefault(criterion_id, _Counts()).add(auto, human, top_status, silent_status)
        overall.add(auto, human, top_status, silent_status)
    rows = [groups[c].finish(c) for c in sorted(groups)]
    rows.append(overall.finish(OVERALL))
    return rows
