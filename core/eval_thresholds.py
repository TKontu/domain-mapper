"""Release-gate thresholds and per-criterion gate check (EV-03, ARCHITECTURE.md §7).

Format (YAML): ``default: {<key>: number}`` and optional ``criteria: {<id>: {<key>: number}}``
with keys ``min_n``, ``min_accuracy``, ``min_kappa``, ``max_silent_rate``,
``max_false_strong_rate``. ``default`` is required in the engine file only. An omitted key
means no check. The scope override is merged key by key: override ``default`` over engine
``default``; per-criterion keys (engine, then override) over the merged default.

Errors (:class:`ThresholdsError`) name the file and key, never the value. Memory is bounded
by the number of criteria.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.eval_metrics import OVERALL, CriterionMetrics

KEYS = ("min_n", "min_accuracy", "min_kappa", "max_silent_rate", "max_false_strong_rate")
_TOP = frozenset({"default", "criteria"})

Values = dict[str, float]


class ThresholdsError(ValueError):
    """A thresholds file is malformed; the message names file and key, never a value."""


@dataclass(frozen=True)
class Thresholds:
    default: Values
    criteria: Mapping[str, Values] = field(default_factory=dict)

    def for_criterion(self, criterion_id: str) -> Values:
        return {**self.default, **self.criteria.get(criterion_id, {})}


@dataclass(frozen=True)
class GateResult:
    criterion_id: str
    passed: bool
    reasons: tuple[str, ...]


def _values(path: Path, where: str, raw: object) -> Values:
    if not isinstance(raw, dict):
        raise ThresholdsError(f"{path}: {where} must be a mapping")
    out: Values = {}
    for key, value in raw.items():
        if key not in KEYS:
            raise ThresholdsError(f"{path}: unknown key {where}.{key}")
        name = f"{where}.{key}"
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ThresholdsError(f"{path}: {name} must be a number")
        if key == "min_n":
            if not isinstance(value, int) or value < 0:
                raise ThresholdsError(f"{path}: {name} must be a non-negative integer")
        elif key == "min_kappa":
            if not -1.0 <= value <= 1.0:
                raise ThresholdsError(f"{path}: {name} must be within [-1, 1]")
        elif not 0.0 <= value <= 1.0:
            raise ThresholdsError(f"{path}: {name} must be within [0, 1]")
        out[key] = value
    return out


def _read(path: Path, *, require_default: bool) -> tuple[Values, dict[str, Values]]:
    text = path.read_text(encoding="utf-8")
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ThresholdsError(f"{path}: invalid YAML") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ThresholdsError(f"{path}: top level must be a mapping")
    for key in data:
        if key not in _TOP:
            raise ThresholdsError(f"{path}: unknown key {key}")
    if require_default and "default" not in data:
        raise ThresholdsError(f"{path}: missing key default")
    default = _values(path, "default", data.get("default") or {})
    raw_criteria = data.get("criteria") or {}
    if not isinstance(raw_criteria, dict):
        raise ThresholdsError(f"{path}: criteria must be a mapping")
    criteria = {
        str(cid): _values(path, f"criteria.{cid}", raw) for cid, raw in raw_criteria.items()
    }
    return default, criteria


def load_thresholds(engine_path: Path, override_path: Path | None) -> Thresholds:
    """Load the engine file (must exist) and merge an optional override (missing = ignored).

    Raises ``FileNotFoundError`` for a missing engine file, :class:`ThresholdsError` otherwise.
    """
    default, criteria = _read(engine_path, require_default=True)
    if override_path is not None and override_path.is_file():
        o_default, o_criteria = _read(override_path, require_default=False)
        default = {**default, **o_default}
        for cid, values in o_criteria.items():
            criteria[cid] = {**criteria.get(cid, {}), **values}
    return Thresholds(default, criteria)


def _fmt(value: float) -> str:
    return f"{value:g}"


def check_gate(metrics: Iterable[CriterionMetrics], thresholds: Thresholds) -> list[GateResult]:
    """Check each criterion (``OVERALL`` excluded) against its thresholds; sorted by id.

    ``n`` below ``min_n`` fails (``n<K``); a ``None`` metric with a set threshold fails
    (``<metric>=none``); bounds are inclusive.
    """
    results: list[GateResult] = []
    for m in metrics:
        if m.criterion_id == OVERALL:
            continue
        t = thresholds.for_criterion(m.criterion_id)
        reasons: list[str] = []
        if "min_n" in t and m.n < t["min_n"]:
            reasons.append(f"n<{_fmt(t['min_n'])}")
        checks: tuple[tuple[str, str, float | None, bool], ...] = (
            ("min_accuracy", "accuracy", m.accuracy, True),
            ("min_kappa", "kappa", m.kappa, True),
            ("max_silent_rate", "silent_rate", m.silent_rate, False),
            ("max_false_strong_rate", "false_strong_rate", m.false_strong_rate, False),
        )
        for key, name, value, is_min in checks:
            if key not in t:
                continue
            bound = t[key]
            if value is None:
                reasons.append(f"{name}=none")
            elif is_min and value < bound:
                reasons.append(f"{name}<{_fmt(bound)}")
            elif not is_min and value > bound:
                reasons.append(f"{name}>{_fmt(bound)}")
        results.append(GateResult(m.criterion_id, not reasons, tuple(reasons)))
    results.sort(key=lambda r: r.criterion_id)
    return results
