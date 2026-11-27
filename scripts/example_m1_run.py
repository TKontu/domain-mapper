#!/usr/bin/env python3
"""EX-01: one reproducible offline M1 run over ``scopes/_example`` (no LLM, no network).

Starts from a fresh SQLite file in ``--out``, seeds entities, screen results and assessments
with evidence through the store APIs (those stages have no LLM-free CLI), then drives the real
``landscape`` CLI as subprocesses: ``presence-set`` -> ``scores-export`` -> ``export`` ->
``mosaic`` (unfiltered and per ``--layer``, each with ``--entity-dir``). Prints one JSON summary
line (steps with exit codes, line counts, mosaic rows/cells per layer) as its last stdout line.

Usage: ``uv run python scripts/example_m1_run.py --out DIR``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.attribute_store  # noqa: E402,F401  (registers attribute_value for FKs)
from core.assess import Extraction, Quote  # noqa: E402
from core.assessment_store import record_assessment  # noqa: E402
from core.entity_store import Entity  # noqa: E402
from core.runs import Run, create_all  # noqa: E402
from core.scope_pack import load_scope_pack  # noqa: E402
from core.screen import ScreenResult  # noqa: E402
from core.screen_store import record_screen_result  # noqa: E402
from core.verify import Route  # noqa: E402

SCOPE = ROOT / "scopes" / "_example"
LANDSCAPE = [sys.executable, "-c", "from app.cli import main; main()"]

# name, segments, presences (country, type), assessments (criterion, status, quote)
ENTITIES: list[tuple[str, list[str], list[tuple[str, str]], list[tuple[str, str, str]]]] = [
    ("alpha", ["agency"], [("XA", "hq")], [("iso17100", "strong", "ISO 17100 certified")]),
    ("beta", ["agency"], [("XB", "sales")], [("public_terms", "standard", "Terms of service")]),
    ("gamma", ["freelancer_network"], [("XZ", "remote")], [("public_terms", "weak", "terms")]),
    ("delta", ["agency"], [("XC", "site"), ("XA", "field_service")], [("iso17100", "weak", "ISO")]),
    ("epsilon", ["agency"], [], [("iso17100", "red_flag", "certificate withdrawn")]),
]


def seed(db_url: str, rubric_version: str) -> list[int]:
    """Create the schema and seed entities, screens and grounded assessments; return ids."""
    engine = create_engine(db_url)
    try:
        create_all(engine)
        with Session(engine) as session, session.begin():
            run = Run(stage="assess", status="finished", started_at=datetime.now(UTC))
            session.add(run)
            session.flush()
            run_id = run.id
        ids: list[int] = []
        for chunk_id, (name, segments, _presences, assessments) in enumerate(ENTITIES, start=1):
            with Session(engine) as session, session.begin():
                ent = Entity(
                    scope_id="example",
                    name=name,
                    primary_domain=f"{name}.example",
                    status="candidate",
                )
                session.add(ent)
                session.flush()
                ids.append(ent.id)
            record_screen_result(
                engine,
                entity_id=ids[-1],
                run_id=run_id,
                result=ScreenResult(True, tuple(segments), 0.9, "example seed", False),
            )
            for criterion, status, quote in assessments:
                record_assessment(
                    engine,
                    entity_id=ids[-1],
                    criterion_id=criterion,
                    extraction=Extraction(
                        status=status,
                        quotes=(Quote(chunk_id, quote, True, "exact", (0, len(quote))),),
                        rationale=f"example seed for {name}",
                        checks_passed=True,
                        failures=(),
                    ),
                    route=Route("accept", None, "example seed"),
                    rubric_version=rubric_version,
                    extractor_model="example-none",
                    verifier_model="example-none",
                    run_id=run_id,
                )
        return ids
    finally:
        engine.dispose()


def landscape(args: list[str], env: dict[str, str], steps: list[dict[str, object]]) -> None:
    """Run one ``landscape`` CLI command; record it; abort the run on a nonzero exit."""
    proc = subprocess.run(
        [*LANDSCAPE, *args], env=env, cwd=ROOT, capture_output=True, text=True, check=False
    )
    steps.append({"command": "landscape " + " ".join(args), "exit_code": proc.returncode})
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"step failed ({proc.returncode}): landscape {' '.join(args)}")


def mosaic_counts(page: str) -> dict[str, int]:
    """Rows and filled (assessed) cells of one mosaic page."""
    return {
        "rows": len(re.findall(r'<tr data-entity="', page)),
        "cells": len(re.findall(r'<td class="cell[^"]*"(?: style="[^"]*")? data-status=', page)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Output directory (fresh).")
    out: Path = parser.parse_args(argv).out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    db = out / "example.sqlite"
    if db.exists():
        db.unlink()
    db_url = f"sqlite:///{db}"
    pack = load_scope_pack(SCOPE)
    ids = seed(db_url, pack.scope.version)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
    env["DM_DATABASE_URL"] = db_url
    scope = ["--scope", str(SCOPE)]
    steps: list[dict[str, object]] = []
    for eid, (_name, _segs, presences, _a) in zip(ids, ENTITIES, strict=True):
        for country, ptype in presences:
            landscape(
                [
                    "presence-set",
                    *scope,
                    "--entity",
                    str(eid),
                    "--country",
                    country,
                    "--type",
                    ptype,
                ],
                env,
                steps,
            )
    scores, export = out / "scores.jsonl", out / "export.jsonl"
    landscape(["scores-export", *scope, "--out", str(scores)], env, steps)
    landscape(["export", *scope, "--format", "json", "--out", str(export)], env, steps)
    mosaic: dict[str, dict[str, int]] = {}
    for layer in ["all", *(la.id for la in pack.scope.layers)]:
        page = out / f"mosaic-{layer}.html"
        args = ["mosaic", *scope, "--scores", str(scores), "--export", str(export)]
        args += ["--out", str(page), "--entity-dir", str(out / f"entities-{layer}")]
        if layer != "all":
            args += ["--layer", layer]
        landscape(args, env, steps)
        mosaic[layer] = mosaic_counts(page.read_text(encoding="utf-8"))
    lines = {p.name: len(p.read_text(encoding="utf-8").splitlines()) for p in (scores, export)}
    print(json.dumps({"steps": steps, "lines": lines, "mosaic": mosaic}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
