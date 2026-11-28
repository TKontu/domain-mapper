#!/usr/bin/env python3
"""M1-01: one serial real-source M1 run over a named scope pack with the real ``landscape`` CLI.

Steps (each a ``landscape`` subprocess, serial): ``web-search --save`` per query line of
``--queries`` (``--limit --per-query``) -> ``resolve --apply`` -> ``crawl`` (entity mode, the
pack's crawl policy, robots respected by the CLI) -> ``screen --jobs 1`` -> ``assess`` per in-scope
entity x applicable criterion. The database is a fresh SQLite file in ``--out``; snapshots go to
``--out/raw`` (``RAW_STORE_DIR`` from the env file is overridden).

Secrets: ``DM_*``, ``SEARXNG_URL``, ``FIRECRAWL_URL`` and ``CAMOUFOX_URL`` from ``--env-file`` go
only into the child environment; failure tails are redacted and every non-SQLite file under
``--out`` is scrubbed of those values. The query file and outputs may hold private scope words:
keep both in a gitignored directory.

``web-search``, ``crawl``, ``screen`` and ``assess`` exit 1 for per-unit outcomes; those are
recorded and the run continues. Any other nonzero exit stops the run. The JSON summary (steps,
candidate/entity/snapshot counts, screens, assessments by status/route, llm_calls ok/error) is the
last stdout line and ``DIR/summary.json``.

Usage: ``uv run python scripts/live_real_m1_run.py --scope DIR --queries FILE --out DIR
--env-file PATH [--per-query N] [--timeout S]``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from core.candidate_store import Candidate  # noqa: E402
from core.config import _read_env_file  # noqa: E402
from core.entity_store import Entity  # noqa: E402
from core.runs import create_all  # noqa: E402
from core.scope_pack import load_scope_pack  # noqa: E402
from core.snapshot_store import SnapshotRow  # noqa: E402

_spec = importlib.util.spec_from_file_location("live_m1_run", ROOT / "scripts" / "live_m1_run.py")
assert _spec and _spec.loader
_base = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("live_m1_run", _base)
_spec.loader.exec_module(_base)
Runner = _base.Runner
StepFailed = _base.StepFailed
_base.UNIT_OUTCOME_COMMANDS.update({"web-search", "crawl"})

SERVICE_KEYS = ("SEARXNG_URL", "FIRECRAWL_URL", "CAMOUFOX_URL")


def parse_queries(path: Path) -> list[tuple[str | None, str]]:
    """One query per line, optionally ``lang<TAB>text``; blanks and ``#`` lines skipped."""
    out: list[tuple[str | None, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        lang, sep, text = line.partition("\t")
        out.append((lang.strip(), text.strip()) if sep else (None, line.strip()))
    return out


def child_env(values: dict[str, str], db: Path, raw: Path) -> tuple[dict[str, str], list[str]]:
    """Child environment and the secret strings to redact."""
    passed = {k: v for k, v in values.items() if k.startswith("DM_") or k in SERVICE_KEYS}
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DM_") and k not in SERVICE_KEYS and k != "RAW_STORE_DIR"
    }
    env.update(passed)
    env["DM_DATABASE_URL"] = f"sqlite:///{db}"
    env["RAW_STORE_DIR"] = str(raw)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    return env, _base.secret_values(passed)


def counts(db_url: str, scope_id: str) -> dict[str, object]:
    engine = create_engine(db_url)
    try:
        create_all(engine)
        with Session(engine) as s:
            cand = s.scalar(select(func.count()).select_from(Candidate)) or 0
            ents = s.scalar(
                select(func.count()).select_from(Entity).where(Entity.scope_id == scope_id)
            )
            snaps = s.scalar(select(func.count()).select_from(SnapshotRow)) or 0
            snap_ents = s.scalar(select(func.count(func.distinct(SnapshotRow.entity_id)))) or 0
    finally:
        engine.dispose()
    return {
        "candidates": cand,
        "entities": ents or 0,
        "snapshots": snaps,
        "entities_with_snapshots": snap_ents,
        **_base.db_counts(db_url, scope_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=5, help="Results saved per query.")
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-call LLM timeout s.")
    ns = parser.parse_args(argv)
    if not 1 <= ns.per_query <= 100:
        parser.error("--per-query must be in 1..100")
    if not ns.env_file.is_file() or not ns.queries.is_file():
        parser.error("--env-file or --queries not found")
    out: Path = ns.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scope_dir = ns.scope.resolve()
    pack = load_scope_pack(scope_dir)
    criteria = [(c.id, frozenset(c.applies_to)) for t in pack.rubric.topics for c in t.criteria]
    try:
        values = _read_env_file(ns.env_file)
    except ValueError:
        print("error: --env-file is malformed (details withheld)", file=sys.stderr)
        return 2
    db = out / "m1real.sqlite"
    if db.exists():
        db.unlink()
    db_url = f"sqlite:///{db}"
    engine = create_engine(db_url)
    try:
        create_all(engine)
    finally:
        engine.dispose()
    env, secrets = child_env(values, db, out / "raw")
    env["DM_LLM_TIMEOUT_S"] = str(ns.timeout)
    runner = Runner(env, out, secrets)  # cwd=out: no stray .env from the checkout is read
    progress = out / "steps.jsonl"  # one line per finished step: survives a killed run
    progress.write_text("", encoding="utf-8")
    base_record = runner._record

    def record(step: dict[str, object], stdout: str, tail: str | None) -> str:
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(step, sort_keys=True) + "\n")
        return base_record(step, stdout, tail)

    runner._record = record  # type: ignore[method-assign]
    scope = ["--scope", str(scope_dir)]
    failed: str | None = None
    try:
        for i, (lang, text) in enumerate(parse_queries(ns.queries)):
            args = ["web-search", *scope, "--query", text, "--limit", str(ns.per_query), "--save"]
            if lang:
                args += ["--language", lang]
            runner.run(args, label=f"web-search q{i}")
        runner.run(["resolve", *scope, "--apply"])
        runner.run(["crawl", *scope])
        runner.run(["screen", *scope, "--jobs", "1"])
        engine = create_engine(db_url)
        try:
            with Session(engine) as s:
                ids = list(s.scalars(select(Entity.id).order_by(Entity.id)))
        finally:
            engine.dispose()
        for eid, cid in _base.assess_pairs(db_url, ids, criteria):
            args = ["assess", *scope, "--entity", str(eid), "--criterion", cid]
            runner.run(args, label=f"assess {eid} {cid}")
    except StepFailed as exc:
        failed = str(exc)
    scrubbed = _base.scrub_artifacts(out, secrets)
    summary: dict[str, object] = {
        "steps": runner.steps,
        "failed_step": failed,
        "per_query": ns.per_query,
        "scrubbed_files": scrubbed,
        **counts(db_url, pack.scope.id),
    }
    line = json.dumps(summary, sort_keys=True)
    (out / "summary.json").write_text(line + "\n", encoding="utf-8")
    print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
