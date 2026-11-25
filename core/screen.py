"""Screen & classify one entity (ARCHITECTURE.md §4 stage 3, §6 model routing).

``screen_entity`` asks the small local model, through the single ``LLM`` protocol, whether
an entity is in scope, which scope segments it belongs to, how confident it is and why.
Out-of-scope entities keep only their reason; borderline results are flagged for review.

Unwired by design: the production caller is the SC-02 row (``landscape screen``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adapters.llm import LLM
from core.scope_pack import ScopePack

BORDERLINE_LOW = 0.35
BORDERLINE_HIGH = 0.65
MAX_REASON_CHARS = 500
PURPOSE = "screen"


class ScreenError(ValueError):
    """The model's screen output was invalid; the message names the field only."""


@dataclass(frozen=True)
class ScreenResult:
    in_scope: bool
    segments: tuple[str, ...]
    confidence: float
    reason: str
    borderline: bool


def screen_schema(scope: ScopePack) -> dict[str, object]:
    """JSON schema for the model output, with the segment enum built from ``scope``."""
    return {
        "type": "object",
        "properties": {
            "in_scope": {"type": "boolean"},
            "segments": {
                "type": "array",
                "items": {"type": "string", "enum": [s.id for s in scope.scope.segments]},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["in_scope", "segments", "confidence", "reason"],
        "additionalProperties": False,
    }


def build_prompt(scope: ScopePack, *, name: str, domain: str, text: str, max_chars: int) -> str:
    s = scope.scope
    segs = "\n".join(f"- {g.id}: {g.label} — {g.definition}" for g in s.segments)
    return (
        f"Scope: {s.title}\n"
        f"Perspective: {s.perspective}\n"
        f"Segments:\n{segs}\n\n"
        "Decide whether the entity below is in scope. If it is, list the segment ids it "
        "belongs to. Give a confidence in [0, 1] and a short reason.\n\n"
        f"Entity: {name}\nDomain: {domain}\n"
        f"Page text:\n{text[:max_chars]}\n"
    )


def screen_entity(
    llm: LLM,
    scope: ScopePack,
    *,
    name: str,
    domain: str,
    text: str,
    run_id: int,
    max_chars: int = 8000,
) -> ScreenResult:
    """Classify one entity from already-fetched page text. ``LlmError`` propagates."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    result = llm.complete(
        build_prompt(scope, name=name, domain=domain, text=text, max_chars=max_chars),
        purpose=PURPOSE,
        run_id=run_id,
        json_schema=screen_schema(scope),
    )
    return _validate(result.parsed, [g.id for g in scope.scope.segments])


def _validate(parsed: object, segment_ids: list[str]) -> ScreenResult:
    if not isinstance(parsed, dict):
        raise ScreenError("screen output: not a JSON object")
    for key in ("in_scope", "segments", "confidence", "reason"):
        if key not in parsed:
            raise ScreenError(f"screen output: missing {key}")

    in_scope = parsed["in_scope"]
    if not isinstance(in_scope, bool):
        raise ScreenError("screen output: in_scope must be a boolean")

    raw_segments = parsed["segments"]
    if not isinstance(raw_segments, list) or not all(isinstance(x, str) for x in raw_segments):
        raise ScreenError("screen output: segments must be an array of strings")
    unknown = [x for x in raw_segments if x not in segment_ids]
    if unknown:
        raise ScreenError(f"screen output: segments has {len(unknown)} unknown id(s)")

    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ScreenError("screen output: confidence must be a number")
    confidence = float(confidence)
    if math.isnan(confidence) or not 0.0 <= confidence <= 1.0:
        raise ScreenError("screen output: confidence must be in [0, 1]")

    reason = parsed["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ScreenError("screen output: reason must be a non-empty string")
    reason = reason.strip()[:MAX_REASON_CHARS]

    chosen = set(raw_segments)
    segments = tuple(s for s in segment_ids if s in chosen) if in_scope else ()
    borderline = BORDERLINE_LOW <= confidence < BORDERLINE_HIGH or (in_scope and not segments)
    return ScreenResult(
        in_scope=in_scope,
        segments=segments,
        confidence=confidence,
        reason=reason,
        borderline=borderline,
    )
