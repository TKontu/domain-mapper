"""Tests for core.eval_metrics (EV-01). Neutral fixtures only (D-003)."""

import pytest
from core.eval_metrics import OVERALL, compute_metrics


def _by_id(rows):
    return {r.criterion_id: r for r in rows}


def test_perfect_agreement_kappa_one():
    pairs = [("c1", "strong", "strong"), ("c1", "weak", "weak"), ("c1", "silent", "silent")]
    row = _by_id(compute_metrics(pairs))["c1"]
    assert row.n == 3
    assert row.accuracy == 1.0
    assert row.kappa == 1.0


def test_textbook_2x2_kappa():
    # Classic table: a=20 (yes,yes), b=5 (auto yes, human no), c=10, d=15. kappa = 0.4
    pairs = (
        [("c1", "strong", "strong")] * 20
        + [("c1", "strong", "weak")] * 5
        + [("c1", "weak", "strong")] * 10
        + [("c1", "weak", "weak")] * 15
    )
    row = _by_id(compute_metrics(pairs))["c1"]
    assert row.n == 50
    assert row.accuracy == pytest.approx(0.7)
    assert row.kappa == pytest.approx(0.4)
    assert row.false_strong_rate == pytest.approx(5 / 50)


def test_kappa_single_class_defined():
    row = _by_id(compute_metrics([("c1", "na", "na")] * 4))["c1"]
    assert row.kappa == 1.0


def test_empty_input():
    rows = compute_metrics(iter([]))
    assert len(rows) == 1
    overall = rows[0]
    assert overall.criterion_id == OVERALL
    assert overall.n == 0
    assert overall.accuracy is None
    assert overall.kappa is None
    assert overall.silent_rate is None
    assert overall.false_strong_rate is None


def test_silent_and_false_strong_rates():
    pairs = [
        ("c1", "silent", "weak"),
        ("c1", "strong", "standard"),
        ("c1", "strong", "strong"),
        ("c1", "silent", "silent"),
    ]
    row = _by_id(compute_metrics(pairs))["c1"]
    assert row.silent_rate == pytest.approx(0.5)
    assert row.false_strong_rate == pytest.approx(0.25)


def test_top_status_parameter():
    pairs = [("c1", "high", "low"), ("c1", "low", "low")]
    row = _by_id(compute_metrics(pairs, top_status="high"))["c1"]
    assert row.false_strong_rate == pytest.approx(0.5)


def test_grouping_and_order_with_overall():
    pairs = iter([("b", "strong", "strong"), ("a", "weak", "strong"), ("b", "weak", "weak")])
    rows = compute_metrics(pairs)
    assert [r.criterion_id for r in rows] == ["a", "b", OVERALL]
    by = _by_id(rows)
    assert by["a"].n == 1 and by["b"].n == 2
    assert by[OVERALL].n == 3
    assert by[OVERALL].accuracy == pytest.approx(2 / 3)
