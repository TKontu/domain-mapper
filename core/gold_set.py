"""Gold-set loader for the eval harness (EV-02, ARCHITECTURE.md §7).

Format (engine choice): JSON Lines, by default at ``<scope>/gold/labels.jsonl``, one object
per line ``{"primary_domain": str, "criterion_id": str, "status": str}``. Blank lines are
skipped. A malformed line, a missing or non-string key, a status outside the scope rubric's
status scale (§3.2 ids) or a duplicate ``(primary_domain, criterion_id)`` raises
:class:`GoldSetError` naming the 1-based line number, never the line content.

The file is streamed line by line; only the set of seen keys is held for duplicate
detection (gold sets are ~30 entities x criteria, bounded). Production caller:
``landscape eval`` (:func:`app.cli.run_eval`).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

_KEYS = ("primary_domain", "criterion_id", "status")


class GoldSetError(ValueError):
    """A gold-set line is malformed, unknown-status or duplicate."""


@dataclass(frozen=True)
class GoldLabel:
    primary_domain: str
    criterion_id: str
    status: str


def load_gold_set(path: str | Path, statuses: Collection[str]) -> Iterator[GoldLabel]:
    """Yield gold labels from ``path``; validate each status against ``statuses``.

    Opening is eager-on-first-``next``; ``OSError`` from a missing file propagates to the caller.
    """
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                raise GoldSetError(f"gold line {lineno}: not valid JSON") from None
            if not isinstance(obj, dict):
                raise GoldSetError(f"gold line {lineno}: expected a JSON object")
            for key in _KEYS:
                if not isinstance(obj.get(key), str):
                    raise GoldSetError(f"gold line {lineno}: missing or non-string {key!r}")
            if obj["status"] not in statuses:
                raise GoldSetError(f"gold line {lineno}: status not in the rubric status scale")
            pair = (obj["primary_domain"], obj["criterion_id"])
            if pair in seen:
                raise GoldSetError(f"gold line {lineno}: duplicate (primary_domain, criterion_id)")
            seen.add(pair)
            yield GoldLabel(*pair, obj["status"])
