"""Import GS-02 review answers as gold labels (GS-04, ARCHITECTURE.md §7).

Inputs: a GS-02 review root's ``answers.json`` (cell id -> ``{status, quote, note, ...}``,
bounded, ~40 cells) and the GS-01 blind ``cells.jsonl`` (streamed line by line; only the
cell id -> (primary_domain, criterion_id) map is kept). Outputs, written atomically
(``.tmp`` then rename): ``<out>/labels.jsonl`` in the :mod:`core.gold_set` format and the
sidecar ``<out>/label_notes.jsonl`` keeping each answered cell's quote and note.

Cells without an answer (or with an empty status) are skipped and counted. An answer naming an
unknown cell, or whose status is outside the rubric scale, raises :class:`GoldImportError`
naming the cell id, never the quote or note text. An existing ``labels.jsonl`` is refused unless
``force``. Production caller: ``landscape gold-import`` (:func:`app.cli.gold_import`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

LABELS_FILE: Final = "labels.jsonl"
NOTES_FILE: Final = "label_notes.jsonl"


class GoldImportError(ValueError):
    """Answers or cells cannot be imported as gold labels."""


@dataclass(frozen=True)
class ImportSummary:
    imported: int
    unanswered: int


def _cells(path: Path) -> dict[str, tuple[str, str]]:
    cells: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                raise GoldImportError(f"cells line {lineno}: not valid JSON") from None
            keys = ("cell_id", "primary_domain", "criterion_id")
            if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) for k in keys):
                raise GoldImportError(f"cells line {lineno}: missing cell_id/domain/criterion")
            cells[obj["cell_id"]] = (obj["primary_domain"], obj["criterion_id"])
    return cells


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def import_labels(
    root: Path,
    cells_path: Path,
    out: Path,
    statuses: Collection[str],
    *,
    force: bool = False,
) -> ImportSummary:
    """Write ``labels.jsonl`` and ``label_notes.jsonl`` under ``out``; return counts."""
    labels_path = out / LABELS_FILE
    if labels_path.exists() and not force:
        raise GoldImportError(f"{labels_path} exists; pass --force to overwrite")
    try:
        answers = json.loads((root / "answers.json").read_text(encoding="utf-8"))
    except ValueError:
        raise GoldImportError("answers.json: not valid JSON") from None
    if not isinstance(answers, dict):
        raise GoldImportError("answers.json: expected a JSON object")
    cells = _cells(cells_path)
    for cid in sorted(answers):
        if cid not in cells:
            raise GoldImportError(f"answer for unknown cell {cid}")
    labels: list[dict[str, object]] = []
    notes: list[dict[str, object]] = []
    unanswered = 0
    for cid, (domain, criterion) in cells.items():
        ans = answers.get(cid)
        status = ans.get("status") if isinstance(ans, dict) else None
        if status is None or status == "":
            unanswered += 1
            continue
        if not isinstance(status, str) or status not in statuses:
            raise GoldImportError(f"cell {cid}: status not in the rubric status scale")
        assert isinstance(ans, dict)
        labels.append({"primary_domain": domain, "criterion_id": criterion, "status": status})
        notes.append(
            {
                "cell_id": cid,
                "primary_domain": domain,
                "criterion_id": criterion,
                "status": status,
                "quote": ans.get("quote", ""),
                "note": ans.get("note", ""),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    _write(out / NOTES_FILE, notes)
    _write(labels_path, labels)
    return ImportSummary(imported=len(labels), unanswered=unanswered)
