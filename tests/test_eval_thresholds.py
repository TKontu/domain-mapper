"""Tests for ``core.eval_thresholds`` (EV-03); tmp_path YAML only."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.eval_metrics import OVERALL, CriterionMetrics
from core.eval_thresholds import ThresholdsError, check_gate, load_thresholds

REPO_ENGINE = Path(__file__).resolve().parent.parent / "eval" / "thresholds.yaml"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def m(cid="c", n=20, acc=0.9, kappa=0.7, silent=0.1, fs=0.0) -> CriterionMetrics:
    return CriterionMetrics(cid, n, acc, kappa, silent, fs)


def test_repo_engine_file_loads_with_provisional_defaults():
    t = load_thresholds(REPO_ENGINE, None)
    assert t.for_criterion("any") == {
        "min_n": 10,
        "min_accuracy": 0.8,
        "min_kappa": 0.6,
        "max_silent_rate": 0.2,
        "max_false_strong_rate": 0.05,
    }


def test_merge_precedence(tmp_path):
    engine = write(
        tmp_path,
        "e.yaml",
        "default: {min_n: 10, min_accuracy: 0.8}\ncriteria: {a: {min_accuracy: 0.5}, b: {min_n: 3}}\n",
    )
    override = write(
        tmp_path,
        "o.yaml",
        "default: {min_n: 5, max_silent_rate: 0.3}\ncriteria: {a: {min_kappa: 0.1}, b: {min_n: 1}}\n",
    )
    t = load_thresholds(engine, override)
    assert t.for_criterion("x") == {"min_n": 5, "min_accuracy": 0.8, "max_silent_rate": 0.3}
    assert t.for_criterion("a") == {
        "min_n": 5,
        "min_accuracy": 0.5,
        "max_silent_rate": 0.3,
        "min_kappa": 0.1,
    }
    assert t.for_criterion("b")["min_n"] == 1


def test_override_without_default_section(tmp_path):
    engine = write(tmp_path, "e.yaml", "default: {min_n: 10}\n")
    override = write(tmp_path, "o.yaml", "criteria: {a: {min_n: 2}}\n")
    t = load_thresholds(engine, override)
    assert t.for_criterion("a") == {"min_n": 2}
    assert t.for_criterion("b") == {"min_n": 10}


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("criteria: {}\n", "default"),
        ("default: {bogus: 1}\n", "bogus"),
        ("default: {min_accuracy: high}\n", "min_accuracy"),
        ("default: {min_accuracy: true}\n", "min_accuracy"),
        ("default: {max_silent_rate: 1.5}\n", "max_silent_rate"),
        ("default: {min_accuracy: -0.1}\n", "min_accuracy"),
        ("default: {min_kappa: -1.5}\n", "min_kappa"),
        ("default: {min_n: -1}\n", "min_n"),
        ("default: {min_n: 2.5}\n", "min_n"),
        ("default: {}\nextra: 1\n", "extra"),
        ("default: {}\ncriteria: {a: {nope: 1}}\n", "nope"),
        ("default: {}\ncriteria: {a: 3}\n", "a"),
        ("default: [1\n", "YAML"),
        ("- 1\n", "mapping"),
    ],
)
def test_validation_errors_name_file_and_key_not_value(tmp_path, text, needle):
    path = write(tmp_path, "e.yaml", text)
    with pytest.raises(ThresholdsError) as exc:
        load_thresholds(path, None)
    msg = str(exc.value)
    assert str(path) in msg
    assert needle in msg
    assert "high" not in msg and "1.5" not in msg and "2.5" not in msg


def test_override_error_names_override_file(tmp_path):
    engine = write(tmp_path, "e.yaml", "default: {}\n")
    override = write(tmp_path, "o.yaml", "default: {min_kappa: 2}\n")
    with pytest.raises(ThresholdsError, match="o.yaml"):
        load_thresholds(engine, override)


def test_missing_engine_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_thresholds(tmp_path / "none.yaml", None)


def test_missing_override_file_is_engine_only(tmp_path):
    engine = write(tmp_path, "e.yaml", "default: {min_n: 4}\n")
    t = load_thresholds(engine, tmp_path / "missing.yaml")
    assert t.for_criterion("a") == {"min_n": 4}


def test_check_gate_pass_fail_per_threshold(tmp_path):
    t = load_thresholds(REPO_ENGINE, None)
    results = check_gate(
        [
            m("ok"),
            m("small", n=9),
            m("acc", acc=0.79),
            m("kap", kappa=0.5),
            m("sil", silent=0.21),
            m("fs", fs=0.06),
            m(OVERALL, n=0),
        ],
        t,
    )
    by = {r.criterion_id: r for r in results}
    assert OVERALL not in by
    assert by["ok"].passed and by["ok"].reasons == ()
    assert by["small"].reasons == ("n<10",)
    assert by["acc"].reasons == ("accuracy<0.8",)
    assert by["kap"].reasons == ("kappa<0.6",)
    assert by["sil"].reasons == ("silent_rate>0.2",)
    assert by["fs"].reasons == ("false_strong_rate>0.05",)
    assert [r.criterion_id for r in results] == sorted(by)


def test_boundary_values_pass(tmp_path):
    t = load_thresholds(REPO_ENGINE, None)
    (r,) = check_gate([m(n=10, acc=0.8, kappa=0.6, silent=0.2, fs=0.05)], t)
    assert r.passed


def test_none_metric_with_set_threshold_fails(tmp_path):
    engine = write(tmp_path, "e.yaml", "default: {min_kappa: 0.1}\n")
    t = load_thresholds(engine, None)
    (r,) = check_gate([CriterionMetrics("c", 0, None, None, None, None)], t)
    assert not r.passed and r.reasons == ("kappa=none",)


def test_omitted_key_means_no_check(tmp_path):
    engine = write(tmp_path, "e.yaml", "default: {}\n")
    t = load_thresholds(engine, None)
    (r,) = check_gate([CriterionMetrics("c", 0, None, None, None, None)], t)
    assert r.passed
