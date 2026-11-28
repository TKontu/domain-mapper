"""Pick and freeze a blind gold-set cell file (GS-01, ARCHITECTURE.md §7).

A *cell* is one (entity, criterion) whose current assessment has readable evidence. Selection
is deterministic: each step takes the model status least represented among chosen cells (ties:
rarer among all candidates, then status id), then the candidate with that status from an entity
already chosen, else the best-ranked new entity (rank: most distinct candidate statuses, then
SHA-256 of the primary domain), ties by criterion id. Up to ``per_entity`` cells per entity and
up to ``entities`` entities; output is in pick order.

Two files are written: ``cells.jsonl`` is blind (cell id, primary domain, criterion id,
criterion text and the frozen evidence: chunk text, snapshot URL and fetch time) and never
carries a status, confidence or assessment id; ``cells.key.jsonl`` maps each cell id to the
model's (LLM-built) status, confidence and assessment id.

An optional *judge* (``landscape gold-pick --clean``) is consulted once per candidate before it
is accepted; it may reject the cell (the next candidate replaces it) and may trim each evidence
text to a passage, applied only when the passage is a verbatim substring. At most
``max_judge_calls`` judgements are made. The judge sees no model status (blind cleaning).

Bounds: entities are read in keyset pages; per entity only its current assessments are read.
Candidate *keys* (ids, status, criterion) for readable cells are held for the whole scope
(small tuples, ~entities x criteria); evidence text is re-read only for judged/accepted cells.
Read-only on the database apart from the judge's own logged calls. Production caller:
``landscape gold-pick`` (:func:`app.cli.run_gold_pick`); consumers are GS-03/GS-04.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.assessment_store import CurrentAssessment, iter_current_assessments
from core.chunk_store import ChunkRow
from core.entity_store import iter_entities
from core.llm_calls import LlmCall
from core.runs import Run

MIN_EVIDENCE_CHARS: Final = 40
MAX_EVIDENCE_CHARS: Final = 6000
MIN_PRINTABLE_RATIO: Final = 0.97
MIN_LETTER_RATIO: Final = 0.5
GOLD_PICK_STAGE: Final = "gold-pick"
CELLS_FILE: Final = "cells.jsonl"
KEY_FILE: Final = "cells.key.jsonl"


class GoldPickError(ValueError):
    """Gold pick cannot proceed (bad arguments or a budget refusal)."""


def is_readable(text: str) -> bool:
    """Length/printable heuristic: 40..6000 chars, >=97% printable, >=50% letters/space."""
    if not MIN_EVIDENCE_CHARS <= len(text) <= MAX_EVIDENCE_CHARS:
        return False
    printable = sum(1 for ch in text if (ch.isprintable() or ch in "\n\t") and ch != "�")
    letters = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    return printable / len(text) >= MIN_PRINTABLE_RATIO and letters / len(text) >= MIN_LETTER_RATIO


def trim_to_passage(text: str, passage: str) -> str:
    """Return ``passage`` if it is a non-blank verbatim substring of ``text``, else ``text``."""
    if passage.strip() and passage in text:
        return passage
    return text


def cell_id(primary_domain: str, criterion_id: str) -> str:
    """Stable cell id: ``c-`` + 16 hex chars of SHA-256 over domain and criterion."""
    digest = hashlib.sha256(f"{primary_domain}\0{criterion_id}".encode()).hexdigest()
    return f"c-{digest[:16]}"


@dataclass(frozen=True)
class FrozenEvidence:
    chunk_text: str
    url: str
    fetched_at: str

    def as_json(self) -> dict[str, str]:
        return {"chunk_text": self.chunk_text, "url": self.url, "fetched_at": self.fetched_at}


@dataclass(frozen=True)
class Cell:
    cell_id: str
    primary_domain: str
    criterion_id: str
    criterion: str
    evidence: tuple[FrozenEvidence, ...]
    status: str
    confidence: float | None
    assessment_id: int
    entity_id: int = 0

    def blind_json(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "primary_domain": self.primary_domain,
            "criterion_id": self.criterion_id,
            "criterion": self.criterion,
            "evidence": [e.as_json() for e in self.evidence],
        }

    def key_json(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "status": self.status,
            "confidence": self.confidence,
            "assessment_id": self.assessment_id,
        }


@dataclass(frozen=True)
class Verdict:
    accept: bool
    passages: tuple[str, ...] = ()


Judge = Callable[[Cell], Verdict]


@dataclass
class PickResult:
    cells: list[Cell] = field(default_factory=list)
    judged: int = 0
    rejected: int = 0
    exhausted_calls: bool = False


@dataclass(frozen=True)
class _Candidate:
    entity_id: int
    primary_domain: str
    criterion_id: str
    status: str
    confidence: float | None
    assessment_id: int


def _iso(value: datetime) -> str:
    return value.isoformat()


def _evidence_of(engine: Engine, row: CurrentAssessment) -> list[FrozenEvidence]:
    """Frozen evidence of one assessment: distinct readable chunks, in evidence order."""
    chunk_ids = sorted({e.chunk_id for e in row.evidence})
    if not chunk_ids:
        return []
    with Session(engine) as session:
        stmt = sa.select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(chunk_ids))
        texts = {cid: text for cid, text in session.execute(stmt)}
    out: list[FrozenEvidence] = []
    seen: set[int] = set()
    for ev in row.evidence:
        text = texts.get(ev.chunk_id)
        if ev.chunk_id in seen or text is None or ev.url is None or ev.fetched_at is None:
            continue
        seen.add(ev.chunk_id)
        if is_readable(text):
            out.append(FrozenEvidence(text, ev.url, _iso(ev.fetched_at)))
    return out


def _reload_evidence(engine: Engine, cand: _Candidate) -> list[FrozenEvidence]:
    for row in iter_current_assessments(
        engine, entity_id=cand.entity_id, criterion_id=cand.criterion_id
    ):
        if row.id == cand.assessment_id:
            return _evidence_of(engine, row)
    return []  # pragma: no cover - the assessment was current a moment ago


def _candidates(
    engine: Engine, scope_id: str, criteria: Mapping[str, str]
) -> list[list[_Candidate]]:
    """Per-entity lists of candidates with readable evidence, entity id order."""
    groups: list[list[_Candidate]] = []
    for ref in iter_entities(engine, scope_id):
        group: list[_Candidate] = []
        for row in iter_current_assessments(engine, entity_id=ref.id):
            if row.criterion_id not in criteria or not row.evidence:
                continue
            if not _evidence_of(engine, row):
                continue
            group.append(
                _Candidate(
                    ref.id, ref.primary_domain, row.criterion_id, row.status, row.confidence, row.id
                )
            )
        if group:
            groups.append(group)
    return groups


def _rank(group: list[_Candidate]) -> tuple[int, str]:
    statuses = {c.status for c in group}
    return (-len(statuses), hashlib.sha256(group[0].primary_domain.encode()).hexdigest())


def pick_cells(
    engine: Engine,
    scope_id: str,
    criteria: Mapping[str, str],
    *,
    entities: int = 8,
    per_entity: int = 5,
    judge: Judge | None = None,
    max_judge_calls: int | None = None,
) -> PickResult:
    """Choose cells deterministically; see the module docstring for the order and bounds.

    ``criteria`` maps criterion id -> criterion text; assessments of other criteria are ignored.
    """
    if entities < 1 or per_entity < 1:
        raise GoldPickError("--entities and --per-entity must be at least 1")
    groups = sorted(_candidates(engine, scope_id, criteria), key=_rank)
    rank = {g[0].entity_id: i for i, g in enumerate(groups)}
    pool = [c for g in groups for c in g]
    overall = Counter(c.status for c in pool)
    counts: Counter[str] = Counter()
    taken: Counter[int] = Counter()
    result = PickResult()
    while True:
        full = len(taken) >= entities
        open_ = [
            c
            for c in pool
            if taken[c.entity_id] < per_entity and (c.entity_id in taken or not full)
        ]
        if not open_:
            return result
        status = min({c.status for c in open_}, key=lambda s: (counts[s], overall[s], s))
        best = min(
            (c for c in open_ if c.status == status),
            key=lambda c: (c.entity_id not in taken, rank[c.entity_id], c.criterion_id),
        )
        pool.remove(best)
        cell = Cell(
            cell_id=cell_id(best.primary_domain, best.criterion_id),
            primary_domain=best.primary_domain,
            criterion_id=best.criterion_id,
            criterion=criteria[best.criterion_id],
            evidence=tuple(_reload_evidence(engine, best)),
            status=best.status,
            confidence=best.confidence,
            assessment_id=best.assessment_id,
            entity_id=best.entity_id,
        )
        if judge is not None:
            if max_judge_calls is not None and result.judged >= max_judge_calls:
                result.exhausted_calls = True
                return result
            result.judged += 1
            verdict = judge(cell)
            if not verdict.accept:
                result.rejected += 1
                continue
            cell = _apply_trim(cell, verdict.passages)
        result.cells.append(cell)
        counts[cell.status] += 1
        taken[cell.entity_id] += 1


def _apply_trim(cell: Cell, passages: Iterable[str]) -> Cell:
    trimmed = list(cell.evidence)
    for i, passage in enumerate(passages):
        if i < len(trimmed) and isinstance(passage, str):
            ev = trimmed[i]
            trimmed[i] = FrozenEvidence(
                trim_to_passage(ev.chunk_text, passage), ev.url, ev.fetched_at
            )
    return Cell(
        cell.cell_id,
        cell.primary_domain,
        cell.criterion_id,
        cell.criterion,
        tuple(trimmed),
        cell.status,
        cell.confidence,
        cell.assessment_id,
        cell.entity_id,
    )


def write_cells(out_dir: Path, cells: Iterable[Cell]) -> None:
    """Write ``cells.jsonl`` (blind) and ``cells.key.jsonl`` atomically (``.tmp`` then rename)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = list(cells)
    for name, rows in (
        (CELLS_FILE, [c.blind_json() for c in cells]),
        (KEY_FILE, [c.key_json() for c in cells]),
    ):
        path = out_dir / name
        tmp = path.with_name(path.name + ".tmp")
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


# --- cleaning (cloud judge) ---------------------------------------------------

CLEAN_PURPOSE: Final = "gold_clean"
CLEAN_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "accept": {"type": "boolean"},
        "reason": {"type": "string"},
        "passages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["accept", "reason", "passages"],
}
# Tool definition and schema overhead added to the prompt-size bound, in tokens.
_PROMPT_OVERHEAD_TOKENS: Final = 500


def clean_prompt(cell: Cell) -> str:
    """The judge prompt: criterion and evidence only, never the model's status."""
    parts = [
        "You prepare cells for a human labelling set. Decide whether the evidence below is "
        "readable text and the criterion is unambiguous for this evidence. For each evidence "
        "item return in `passages` the shortest verbatim passage (copied exactly) relevant to "
        "the criterion, or an empty string to keep it whole. Do not judge the answer itself.",
        f"Criterion: {cell.criterion}",
    ]
    for i, ev in enumerate(cell.evidence):
        parts.append(f"Evidence {i} (from {ev.url}):\n<<<\n{ev.chunk_text}\n>>>")
    return "\n\n".join(parts)


def prior_cloud_spend(engine: Engine, model: str) -> float:
    """USD already logged for ``model`` or for any ``gold-pick`` run (cumulative cap basis)."""
    stmt = (
        sa.select(sa.func.coalesce(sa.func.sum(LlmCall.cost), 0.0))
        .select_from(LlmCall)
        .join(Run, Run.id == LlmCall.run_id)
        .where(sa.or_(LlmCall.model == model, Run.stage == GOLD_PICK_STAGE))
    )
    with Session(engine) as session:
        return float(session.execute(stmt).scalar_one())


def worst_case_cost(prompt: str, *, rate_in: float, rate_out: float, max_tokens: int) -> float:
    """Upper bound of one call's USD: UTF-8 bytes (>= tokens) + overhead in, ``max_tokens`` out."""
    tokens_in = len(prompt.encode("utf-8")) + _PROMPT_OVERHEAD_TOKENS
    return tokens_in / 1_000_000 * rate_in + max_tokens / 1_000_000 * rate_out


def make_cloud_judge(
    complete: Callable[[str], Mapping[str, object] | None],
    engine: Engine,
    *,
    model: str,
    budget_usd: float,
    rate_in: float,
    rate_out: float,
    max_tokens: int,
) -> Judge:
    """Wrap ``complete(prompt) -> parsed JSON`` with the cumulative budget check.

    Before every call the logged spend (:func:`prior_cloud_spend`) plus the call's
    :func:`worst_case_cost` must stay within ``budget_usd``; otherwise :class:`GoldPickError`
    is raised and no call is made. A budget or rate of zero refuses to build (cap must be live).
    """
    if budget_usd <= 0:
        raise GoldPickError("--clean needs DM_LLM_BUDGET_USD > 0 (the cap must be live)")
    if rate_in <= 0 or rate_out <= 0:
        raise GoldPickError(
            "--clean needs DM_CLOUD_COST_PER_MTOK_IN/OUT > 0 (the cap must be live)"
        )

    def judge(cell: Cell) -> Verdict:
        prompt = clean_prompt(cell)
        spent = prior_cloud_spend(engine, model)
        worst = worst_case_cost(prompt, rate_in=rate_in, rate_out=rate_out, max_tokens=max_tokens)
        if spent + worst > budget_usd:
            raise GoldPickError(
                f"budget: spent {spent:.4f} USD + next call up to {worst:.4f} USD"
                f" exceeds DM_LLM_BUDGET_USD {budget_usd:.4f}"
            )
        parsed = complete(prompt) or {}
        passages = parsed.get("passages")
        return Verdict(
            accept=parsed.get("accept") is True,
            passages=tuple(p for p in passages if isinstance(p, str))
            if isinstance(passages, list)
            else (),
        )

    return judge
