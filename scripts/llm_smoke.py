"""LL-01 live smoke: one completion through ``build_llm(settings, "local")``.

Operator tool, unwired by design. Reads settings only via ``--env-file`` and
prints/writes only model, latency, status, error class and token counts —
never the gateway URL, key, headers or any other .env value.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.llm_calls  # noqa: E402,F401  (registers llm_call)
from adapters.llm import Transport, build_llm  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.runs import Run, create_all  # noqa: E402

PROMPT = "Reply with the single word: ok"
PURPOSE = "smoke"


def run_smoke(
    env_file: str, timeout: float, db_path: str, transport: Transport | None = None
) -> dict[str, Any]:
    settings = load_settings(env_file)
    engine = create_engine(f"sqlite:///{db_path}")
    create_all(engine)
    with Session(engine) as session:
        run = Run(stage="llm_smoke")
        session.add(run)
        session.commit()
        run_id = run.id
    result: dict[str, Any] = {
        "model": settings.local_model,
        "latency_s": None,
        "status": "error",
        "error_class": None,
        "tokens_in": None,
        "tokens_out": None,
    }
    started = time.monotonic()
    try:
        llm = build_llm(settings, "local", db=engine, transport=transport, timeout=timeout)
        out = llm.complete(PROMPT, purpose=PURPOSE, run_id=run_id)
        result.update(status="ok", tokens_in=out.tokens_in, tokens_out=out.tokens_out)
    except Exception as exc:  # recorded, never re-raised: message may not be trusted
        result["error_class"] = type(exc).__name__
    result["latency_s"] = round(time.monotonic() - started, 3)
    engine.dispose()
    return result


def main(argv: Sequence[str] | None = None, transport: Transport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--db", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    db = args.db or str(Path(tempfile.mkdtemp(prefix="llm-smoke-")) / "smoke.sqlite")
    result = run_smoke(args.env_file, args.timeout, db, transport)
    text = json.dumps(result, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
