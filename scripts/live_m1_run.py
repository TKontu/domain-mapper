#!/usr/bin/env python3
"""LL-01: one live M1 run over ``scopes/_example`` against the configured local LLM gateway.

Seeds ``--entities`` neutral synthetic entities (status ``unscreened``, one snapshot with chunks
each, via the store APIs like ``example_m1_run.py``; no seeded screens or assessments) into a
fresh SQLite file in ``--out``, then drives the real ``landscape`` CLI as subprocesses:
``screen`` (live local model) -> ``assess`` per in-scope entity x applicable criterion (keyword
retrieval only; live extractor + verifier) -> ``presence-extract --type hq`` ->
``scores-export`` -> ``export`` -> ``mosaic --entity-dir`` (unfiltered and per layer).

Secrets: the ``DM_*`` values of ``--env-file`` go only into the child processes' environment,
never into argv, stdout, stderr, logs or files written here. Child output is never echoed except
a redacted tail on failure. The CLI stores model names in the SQLite ``llm_call.model`` and
``assessment.extractor_model``/``verifier_model`` columns (left as is; the summary never reads
them) and ``export`` copies them into its JSONL, so after the steps every other file under
``--out`` is scrubbed of env-file values.

``screen`` and ``assess`` exit 1 for per-unit outcomes (an entity errored, checks failed,
adjudicate); those are recorded and the run continues. Any other nonzero exit stops the run.
The JSON summary (steps with command name, exit code and elapsed seconds; screen, assessment,
llm_call, token and mosaic counts) is the last stdout line and ``DIR/summary.json``. When every
screen ``llm_call`` errored (dead gateway) the summary has ``"degraded": true`` and the exit is 3.

Usage: ``uv run python scripts/live_m1_run.py --out DIR --env-file PATH``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.attribute_store  # noqa: E402,F401  (registers attribute_value for FKs)
from adapters.fetcher import FetchResult  # noqa: E402
from core.assessment_store import AssessmentRow  # noqa: E402
from core.chunk_store import store_chunks  # noqa: E402
from core.chunking import Chunk  # noqa: E402
from core.config import _read_env_file  # noqa: E402
from core.entity_store import Entity  # noqa: E402
from core.llm_calls import LlmCall  # noqa: E402
from core.runs import create_all  # noqa: E402
from core.scope_pack import load_scope_pack  # noqa: E402
from core.screen_store import ScreenResultRow  # noqa: E402
from core.snapshot_store import record_snapshot  # noqa: E402

SCOPE = ROOT / "scopes" / "_example"
LANDSCAPE = [sys.executable, "-c", "from app.cli import main; main()"]
FETCHED_AT = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
# Exit 1 from these commands is a per-unit outcome, not a broken run.
UNIT_OUTCOME_COMMANDS = {"screen", "assess"}

CITIES = ["Port Avalor", "Brennmark", "Calvessa"]
PROFILES = [
    (
        "{name} is a translation agency with in-house project managers, headquartered in {city}.",
        "{name} states it is ISO 17100 certified; certificate number {n:04d} is listed.",
        "General terms of service, covering liability and confidentiality, are published.",
    ),
    (
        "{name} is a cooperative network brokering independent translators, based in {city}.",
        "Terms and conditions are available on request.",
    ),
    (
        "{name} sells office furniture and has its headquarters in {city}.",
        "{name} ships desks and chairs to business customers.",
    ),
    (
        "{name} provides translation services with project management, headquartered in {city}.",
        "{name} mentions ISO alignment but gives no certificate reference.",
        "Terms of service are published online, undated.",
    ),
]


class StepFailed(Exception):
    """A child command exited with a code that stops the run."""


def entity_texts(i: int) -> tuple[str, list[str]]:
    name = f"entity{i:02d}"
    profile = PROFILES[i % len(PROFILES)]
    city = CITIES[i % len(CITIES)]
    return name, [t.format(name=name.capitalize(), city=city, n=i) for t in profile]


def seed(db_url: str, scope_id: str, n: int) -> list[int]:
    """Create the schema; seed ``n`` unscreened entities with one snapshot and chunks each."""
    engine = create_engine(db_url)
    try:
        create_all(engine)
        ids: list[int] = []
        for i in range(n):
            name, texts = entity_texts(i)
            with Session(engine) as session, session.begin():
                ent = Entity(
                    scope_id=scope_id,
                    name=name,
                    primary_domain=f"{name}.example",
                    status="unscreened",
                )
                session.add(ent)
                session.flush()
                ids.append(ent.id)
            url = f"https://{name}.example/about"
            snapshot_id = record_snapshot(
                engine,
                entity_id=ids[-1],
                url=url,
                result=FetchResult(
                    url, 200, "text/html", "\n\n".join(texts).encode(), fetched_at=FETCHED_AT
                ),
                source_kind="live_m1_seed",
            )
            chunks, start = [], 0
            for seq, text in enumerate(texts):
                chunks.append(Chunk(seq, text, None, (name,), start, start + len(text)))
                start += len(text) + 2
            store_chunks(engine, snapshot_id, chunks)
        return ids
    finally:
        engine.dispose()


SECRET_KEYS = {
    "DM_LLM_GATEWAY_API_KEY",
    "DM_LLM_GATEWAY_URL",
    "DM_LOCAL_MODEL",
    "DM_VERIFIER_MODEL",
}
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_URL")
MIN_SECRET_LEN = 4
# Exit code when every screen llm_call errored (dead gateway).
EXIT_DEGRADED = 3
SCRUB_SKIP_SUFFIXES = ("-journal", "-wal", "-shm")


def secret_values(values: dict[str, str]) -> list[str]:
    """Values of secret/URL/model keys, their URL netlocs/hosts, and encoded forms.

    Benign settings (``true``, short numbers) are never scrubbed: only the named keys and
    ``*_KEY``/``*_TOKEN``/``*_URL`` count, and values shorter than 4 chars are skipped.
    """
    base: set[str] = set()
    for key, value in values.items():
        if key in SECRET_KEYS or key.endswith(SECRET_SUFFIXES):
            base.add(value.strip())
            if key.endswith("_URL"):
                parts = urllib.parse.urlsplit(value.strip())
                base.update({parts.netloc, parts.hostname or ""})
    out: set[str] = set()
    for value in base:
        if len(value) < MIN_SECRET_LEN:
            continue
        out.update(
            {
                value,
                urllib.parse.quote(value),
                urllib.parse.quote(value, safe=""),
                urllib.parse.quote_plus(value),
                json.dumps(value)[1:-1],
            }
        )
    return sorted((v for v in out if len(v) >= MIN_SECRET_LEN), key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> str:
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    return text


class Runner:
    def __init__(self, env: dict[str, str], cwd: Path, secrets: list[str]) -> None:
        self.env, self.cwd, self.secrets = env, cwd, secrets
        self.steps: list[dict[str, object]] = []

    def run(self, args: list[str], *, label: str | None = None) -> str:
        """Run one ``landscape`` command; record it; raise StepFailed on a fatal exit."""
        name = args[0]
        started = time.monotonic()
        proc = subprocess.run(
            [*LANDSCAPE, *args],
            env=self.env,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        step: dict[str, object] = {
            "command": label or name,
            "exit_code": proc.returncode,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        self.steps.append(step)
        allowed = {0, 1} if name in UNIT_OUTCOME_COMMANDS else {0}
        if proc.returncode not in allowed:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-5:])
            sys.stderr.write(redact(tail, self.secrets) + "\n")
            raise StepFailed(str(step["command"]))
        return proc.stdout


def scrub_artifacts(out: Path, secrets: list[str]) -> int:
    """Redact secret values from every non-SQLite file under ``out``; return files changed.

    ``landscape export`` (and pages built from it) carry ``extractor_model``/``verifier_model``;
    the SQLite file keeps them (``llm_call.model``, ``assessment.*_model``) untouched.
    """
    changed = 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix == ".sqlite":
            continue
        if ".sqlite-" in path.name or path.name.endswith(SCRUB_SKIP_SUFFIXES):
            continue  # SQLite journal/WAL/SHM files: binary, owned by the database
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        clean = redact(text, secrets)
        if clean != text:
            path.write_text(clean, encoding="utf-8", errors="surrogateescape")
            changed += 1
    return changed


def mosaic_counts(page: str) -> dict[str, int]:
    """Rows and filled (assessed) cells of one mosaic page."""
    return {
        "rows": len(re.findall(r'<tr data-entity="', page)),
        "cells": len(re.findall(r'<td class="cell[^"]*"(?: style="[^"]*")? data-status=', page)),
    }


def db_counts(db_url: str, scope_id: str) -> dict[str, object]:
    engine = create_engine(db_url)
    try:
        with Session(engine) as session:
            screened = dict(
                session.execute(
                    select(Entity.status, func.count())
                    .where(Entity.scope_id == scope_id)
                    .group_by(Entity.status)
                ).all()
            )
            by_status = dict(
                session.execute(
                    select(AssessmentRow.status, func.count()).group_by(AssessmentRow.status)
                ).all()
            )
            by_route = dict(
                session.execute(
                    select(AssessmentRow.route_kind, func.count()).group_by(
                        AssessmentRow.route_kind
                    )
                ).all()
            )
            screen_calls = dict(
                session.execute(
                    select(LlmCall.status, func.count())
                    .where(LlmCall.purpose == "screen")
                    .group_by(LlmCall.status)
                ).all()
            )
            calls = dict(
                session.execute(select(LlmCall.status, func.count()).group_by(LlmCall.status)).all()
            )
            tokens_in, tokens_out = session.execute(
                select(
                    func.coalesce(func.sum(LlmCall.tokens_in), 0),
                    func.coalesce(func.sum(LlmCall.tokens_out), 0),
                )
            ).one()
    finally:
        engine.dispose()
    return {
        "screened": {
            "in_scope": screened.get("in_scope", 0),
            "out_of_scope": screened.get("out_of_scope", 0),
            "borderline": screened.get("borderline", 0),
            "unscreened": screened.get("unscreened", 0),
        },
        "assessments": {"by_status": by_status, "by_route": by_route},
        # Every screen call errored (dead gateway): the run is degraded.
        "degraded": bool(screen_calls) and screen_calls.get("ok", 0) == 0,
        "llm_calls": {"ok": calls.get("ok", 0), "error": calls.get("error", 0)},
        "tokens": {"in": int(tokens_in), "out": int(tokens_out)},
    }


def assess_pairs(
    db_url: str, ids: list[int], criteria: list[tuple[str, frozenset[str]]]
) -> list[tuple[int, str]]:
    """In-scope entities x criteria whose ``applies_to`` meets the entity's screened segments."""
    engine = create_engine(db_url)
    try:
        with Session(engine) as session:
            pairs: list[tuple[int, str]] = []
            for eid in ids:
                ent = session.get(Entity, eid)
                if ent is None or ent.status != "in_scope":
                    continue
                segs = session.scalars(
                    select(ScreenResultRow.segments)
                    .where(ScreenResultRow.entity_id == eid)
                    .order_by(ScreenResultRow.id.desc())
                    .limit(1)
                ).first()
                have = set(segs or [])
                for cid, applies in criteria:
                    if not applies or applies & have:
                        pairs.append((eid, cid))
            return pairs
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Output directory (fresh).")
    parser.add_argument("--env-file", type=Path, required=True, help="DM_* settings file.")
    parser.add_argument("--entities", type=int, default=20, help="Entities to seed (1-50).")
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-call LLM timeout s.")
    parser.add_argument("--criteria", default=None, help="Comma-separated criterion ids.")
    ns = parser.parse_args(argv)
    if not 1 <= ns.entities <= 50:
        parser.error("--entities must be in 1..50")
    if ns.timeout <= 0:
        parser.error("--timeout must be > 0")
    if not ns.env_file.is_file():
        parser.error("--env-file not found")
    out: Path = ns.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pack = load_scope_pack(SCOPE)
    all_criteria = [(c.id, frozenset(c.applies_to)) for t in pack.rubric.topics for c in t.criteria]
    if ns.criteria:
        wanted = [c.strip() for c in ns.criteria.split(",") if c.strip()]
        unknown = sorted(set(wanted) - {c for c, _ in all_criteria})
        if unknown:
            parser.error(f"unknown criteria: {', '.join(unknown)}")
        criteria = [c for c in all_criteria if c[0] in wanted]
    else:
        criteria = all_criteria

    try:
        file_values = _read_env_file(ns.env_file)
    except ValueError:
        print("error: --env-file is malformed (details withheld)", file=sys.stderr)
        return 2
    dm_values = {k: v for k, v in file_values.items() if k.startswith("DM_")}
    secrets = secret_values(dm_values)

    db = out / "live_m1.sqlite"
    if db.exists():
        db.unlink()
    db_url = f"sqlite:///{db}"
    ids = seed(db_url, pack.scope.id, ns.entities)

    env = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
    env.update(dm_values)
    env["DM_DATABASE_URL"] = db_url
    env["DM_LLM_TIMEOUT_S"] = str(ns.timeout)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    runner = Runner(env, out, secrets)  # cwd=out: no stray .env from the checkout is read
    scope = ["--scope", str(SCOPE)]
    failed: str | None = None
    mosaic: dict[str, dict[str, int]] = {}
    try:
        runner.run(["screen", *scope])
        for eid, cid in assess_pairs(db_url, ids, criteria):
            runner.run(
                ["assess", *scope, "--entity", str(eid), "--criterion", cid],
                label=f"assess {eid} {cid}",
            )
        runner.run(["presence-extract", *scope, "--type", "hq"])
        scores, export = out / "scores.jsonl", out / "export.jsonl"
        runner.run(["scores-export", *scope, "--out", str(scores)])
        runner.run(["export", *scope, "--format", "json", "--out", str(export)])
        for layer in ["all", *(la.id for la in pack.scope.layers)]:
            page = out / f"mosaic-{layer}.html"
            args = ["mosaic", *scope, "--scores", str(scores), "--export", str(export)]
            args += ["--out", str(page), "--entity-dir", str(out / f"entities-{layer}")]
            if layer != "all":
                args += ["--layer", layer]
            runner.run(args, label=f"mosaic {layer}")
            mosaic[layer] = mosaic_counts(page.read_text(encoding="utf-8"))
    except StepFailed as exc:
        failed = str(exc)
    scrubbed = scrub_artifacts(out, secrets)
    summary: dict[str, object] = {
        "entities": len(ids),
        "criteria": [c for c, _ in criteria],
        "timeout_s": ns.timeout,
        "steps": runner.steps,
        "failed_step": failed,
        "mosaic": mosaic,
        "scrubbed_files": scrubbed,
        **db_counts(db_url, pack.scope.id),
    }
    line = json.dumps(summary, sort_keys=True)
    (out / "summary.json").write_text(line + "\n", encoding="utf-8")
    print(line)
    if failed:
        return 1
    return EXIT_DEGRADED if summary["degraded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
