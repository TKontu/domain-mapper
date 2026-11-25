"""Structured, one-line-per-record logging (RT-02).

:func:`configure_logging` installs a single ``StreamHandler`` on the root logger
and renders every record as ``timestamp level=... logger=... message=...
key=value ...``. Callers pass structured values through the stdlib ``extra``
mechanism; no record spans more than one physical line (newlines in messages,
extras or tracebacks are escaped).

The module is named ``core.log`` rather than ``core.logging`` so it never shadows
the standard library ``logging`` package from inside the ``core`` package.

Prompt and response text are never logged: LLM call records carry a prompt
*hash* and usage counters only (see :mod:`core.llm_calls`).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final, TextIO

# Attributes every ``LogRecord`` already carries; anything else is a structured
# extra and is rendered as ``key=value``.
_STANDARD_ATTRS: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
)
# Marks the handler this module installs, so reconfiguring replaces it instead
# of stacking a second handler that duplicates every line.
_HANDLER_MARKER: Final = "_domain_mapper_log_handler"


def _render(value: object) -> str:
    """Render one field, keeping the whole record on a single physical line."""
    text = str(value)
    if text and not any(char.isspace() for char in text):
        return text
    # json quotes and escapes whitespace (newlines included) in one step.
    return json.dumps(text)


class StructuredFormatter(logging.Formatter):
    """Render a record as one line of ``key=value`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        parts = [
            timestamp,
            f"level={record.levelname}",
            f"logger={record.name}",
            f"message={_render(record.getMessage())}",
        ]
        for key in sorted(record.__dict__):
            if key in _STANDARD_ATTRS or key in {"message", "asctime"}:
                continue
            parts.append(f"{key}={_render(record.__dict__[key])}")
        if record.exc_info:
            parts.append(f"exc={_render(self.formatException(record.exc_info))}")
        return " ".join(parts)


def configure_logging(
    level: str | int = "INFO", *, stream: TextIO | None = None
) -> logging.Handler:
    """Configure root logging for structured one-line records and return the handler.

    ``level`` accepts a stdlib level name or number. A previously installed
    handler of this module is replaced, so calling this twice does not duplicate
    output. Other handlers configured by the embedding application are left
    alone. ``stream`` defaults to ``sys.stderr``; tests pass a ``StringIO``.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    return handler
