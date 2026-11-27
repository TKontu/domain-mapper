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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import Engine, create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.llm_calls  # noqa: E402,F401  (registers llm_call)
from adapters.llm import Transport, build_llm  # noqa: E402
from core.config import Settings, load_settings  # noqa: E402
from core.runs import Run, create_all  # noqa: E402

PROMPT = "Reply with the single word: ok"
LONG_PROMPT = (
    "Write one paragraph of about 200 words describing how a public library organises "
    "its catalogue. Plain prose, no lists or headings."
)
PROMPTS = {"short": PROMPT, "long": LONG_PROMPT}
PURPOSE = "smoke"


def _setup(env_file: str, db_path: str) -> tuple[Settings, Engine, int]:
    settings = load_settings(env_file)
    engine = create_engine(f"sqlite:///{db_path}")
    create_all(engine)
    with Session(engine) as session:
        run = Run(stage="llm_smoke")
        session.add(run)
        session.commit()
        run_id = run.id
    return settings, engine, run_id


def run_smoke(
    env_file: str,
    timeout: float,
    db_path: str,
    transport: Transport | None = None,
    *,
    prompt: str = "short",
    max_tokens: int | None = None,
    warmup: bool = False,
) -> dict[str, Any]:
    settings, engine, run_id = _setup(env_file, db_path)
    call = (settings, engine, run_id, timeout, transport, prompt, max_tokens)
    warm = _one_call(*call)["latency_s"] if warmup else None
    result = _one_call(*call)
    engine.dispose()
    result.update(max_tokens=max_tokens, prompt=prompt)
    if warmup:
        result["warmup_latency_s"] = warm
    return result


def run_parallel(
    env_file: str,
    timeout: float,
    db_path: str,
    parallel: int,
    rounds: int,
    transport: Transport | None = None,
    *,
    prompt: str = "short",
    max_tokens: int | None = None,
    warmup: bool = False,
) -> dict[str, Any]:
    """R rounds of N simultaneous completions; threads bounded by N.

    With ``warmup`` one call runs first; it is reported only as ``warmup_latency_s``.
    """
    settings, engine, run_id = _setup(env_file, db_path)
    call = (settings, engine, run_id, timeout, transport, prompt, max_tokens)
    warm = _one_call(*call)["latency_s"] if warmup else None
    round_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for index in range(rounds):
            started = time.monotonic()
            futures = [pool.submit(_one_call, *call) for _ in range(parallel)]
            calls = [f.result() for f in futures]
            latencies = [c["latency_s"] for c in calls]
            round_results.append(
                {
                    "round": index + 1,
                    "wall_s": round(time.monotonic() - started, 3),
                    "max_latency_s": max(latencies),
                    "min_latency_s": min(latencies),
                    "tokens_out_total": sum(c["tokens_out"] or 0 for c in calls),
                    "calls": calls,
                }
            )
    engine.dispose()
    ok = all(c["status"] == "ok" for r in round_results for c in r["calls"])
    summary: dict[str, Any] = {
        "model": settings.local_model,
        "parallel": parallel,
        "rounds": rounds,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "status": "ok" if ok else "error",
        "round_results": round_results,
    }
    if warmup:
        summary["warmup_latency_s"] = warm
    return summary


def _one_call(
    settings: Settings,
    engine: Engine,
    run_id: int,
    timeout: float,
    transport: Transport | None,
    prompt: str = "short",
    max_tokens: int | None = None,
) -> dict[str, Any]:
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
        llm = build_llm(
            settings,
            "local",
            db=engine,
            transport=transport,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        out = llm.complete(PROMPTS[prompt], purpose=PURPOSE, run_id=run_id)
        result.update(status="ok", tokens_in=out.tokens_in, tokens_out=out.tokens_out)
    except Exception as exc:  # recorded, never re-raised: message may not be trusted
        result["error_class"] = type(exc).__name__
    result["latency_s"] = round(time.monotonic() - started, 3)
    return result


def _bounded(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be {low}..{high}")
        return value

    return parse


def main(argv: Sequence[str] | None = None, transport: Transport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--db", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--parallel", type=_bounded(1, 8), default=1)
    parser.add_argument("--rounds", type=_bounded(1, 5), default=1)
    parser.add_argument("--max-tokens", type=_bounded(1, 4096), default=None)
    parser.add_argument("--long", action="store_true", help="ask for a ~200-word paragraph")
    parser.add_argument("--warmup", action="store_true", help="one unreported call first")
    args = parser.parse_args(argv)
    opts: dict[str, Any] = {
        "prompt": "long" if args.long else "short",
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
    }
    db = args.db or str(Path(tempfile.mkdtemp(prefix="llm-smoke-")) / "smoke.sqlite")
    if args.parallel == 1 and args.rounds == 1:
        result = run_smoke(args.env_file, args.timeout, db, transport, **opts)
    else:
        result = run_parallel(
            args.env_file, args.timeout, db, args.parallel, args.rounds, transport, **opts
        )
    text = json.dumps(result, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
