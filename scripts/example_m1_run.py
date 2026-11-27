#!/usr/bin/env python3
"""EX-01/EX-02: one reproducible offline M1 run over ``scopes/_example`` (no LLM, no network).

Starts from a fresh SQLite file in ``--out``, seeds entities, one snapshot per entity with
neutral chunk text (``record_snapshot`` + ``store_chunks``), screen results and assessments whose
quotes cite those real chunks through the store APIs (those stages have no LLM-free CLI), then
drives the real ``landscape`` CLI as subprocesses: ``presence-extract`` (presences the pack's
``presence_terms`` find) and ``presence-set`` (the rest) -> ``scores-export`` -> ``export`` ->
``mosaic`` (unfiltered and per ``--layer``, each with ``--entity-dir``). Prints one JSON summary
line (steps with exit codes, extracted count, line counts, mosaic rows/cells per layer) as its
last stdout line.

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

from sqlalchemy import Engine, create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.attribute_store  # noqa: E402,F401  (registers attribute_value for FKs)
from adapters.fetcher import FetchResult  # noqa: E402
from core.assess import Extraction, Quote  # noqa: E402
from core.assessment_store import record_assessment  # noqa: E402
from core.chunk_store import ChunkRow, store_chunks  # noqa: E402
from core.chunking import Chunk  # noqa: E402
from core.entity_store import Entity  # noqa: E402
from core.runs import Run, create_all  # noqa: E402
from core.scope_pack import load_scope_pack  # noqa: E402
from core.screen import ScreenResult  # noqa: E402
from core.screen_store import record_screen_result  # noqa: E402
from core.snapshot_store import record_snapshot  # noqa: E402
from core.verify import Route  # noqa: E402

SCOPE = ROOT / "scopes" / "_example"
LANDSCAPE = [sys.executable, "-c", "from app.cli import main; main()"]

FETCHED_AT = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

# name, segments, chunk texts (neutral; ``presence_terms`` only where a presence is meant),
# presences (country, type, "extract" via presence-extract or "set" via presence-set),
# assessments (criterion, status, quote; each quote appears verbatim in one chunk text)
ENTITIES: list[
    tuple[str, list[str], list[str], list[tuple[str, str, str]], list[tuple[str, str, str]]]
] = [
    (
        "alpha",
        ["agency"],
        ["Alpha is headquartered in Port Avalor.", "Alpha is ISO 17100 certified since 2020."],
        [("XA", "hq", "extract")],
        [("iso17100", "strong", "ISO 17100 certified")],
    ),
    (
        "beta",
        ["agency"],
        ["Beta runs a sales office in Brennmark.", "Terms of service are published online."],
        [("XB", "sales", "extract")],
        [("public_terms", "standard", "Terms of service")],
    ),
    (
        "gamma",
        ["freelancer_network"],
        ["Gamma is a fully remote network; its terms are public."],
        [("XZ", "remote", "set")],
        [("public_terms", "weak", "terms")],
    ),
    (
        "delta",
        ["agency"],
        ["Delta operates a site in Calvessa.", "Delta mentions ISO alignment."],
        [("XC", "site", "extract"), ("XA", "field_service", "set")],
        [("iso17100", "weak", "ISO")],
    ),
    (
        "epsilon",
        ["agency"],
        ["Epsilon notes its certificate withdrawn last year."],
        [],
        [("iso17100", "red_flag", "certificate withdrawn")],
    ),
]


def seed_chunks(engine: Engine, entity_id: int, name: str, texts: list[str]) -> list[int]:
    """Record one snapshot for the entity and store ``texts`` as its chunks; return chunk ids."""
    url = f"https://{name}.example/about"
    body = "\n\n".join(texts)
    snapshot_id = record_snapshot(
        engine,
        entity_id=entity_id,
        url=url,
        result=FetchResult(url, 200, "text/html", body.encode(), fetched_at=FETCHED_AT),
        source_kind="example_seed",
    )
    chunks, start = [], 0
    for seq, text in enumerate(texts):
        chunks.append(Chunk(seq, text, None, (name,), start, start + len(text)))
        start += len(text) + 2
    store_chunks(engine, snapshot_id, chunks)
    with Session(engine) as session:
        return list(
            session.scalars(
                select(ChunkRow.id)
                .where(ChunkRow.snapshot_id == snapshot_id)
                .order_by(ChunkRow.seq)
            )
        )


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
        for name, segments, texts, _presences, assessments in ENTITIES:
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
            chunk_ids = seed_chunks(engine, ids[-1], name, texts)
            record_screen_result(
                engine,
                entity_id=ids[-1],
                run_id=run_id,
                result=ScreenResult(True, tuple(segments), 0.9, "example seed", False),
            )
            for criterion, status, quote in assessments:
                idx = next(i for i, t in enumerate(texts) if quote in t)
                chunk_id, offset = chunk_ids[idx], texts[idx].index(quote)
                record_assessment(
                    engine,
                    entity_id=ids[-1],
                    criterion_id=criterion,
                    extraction=Extraction(
                        status=status,
                        quotes=(
                            Quote(chunk_id, quote, True, "exact", (offset, offset + len(quote))),
                        ),
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


def landscape(args: list[str], env: dict[str, str], steps: list[dict[str, object]]) -> str:
    """Run one ``landscape`` CLI command; record it; abort on a nonzero exit; return stdout."""
    proc = subprocess.run(
        [*LANDSCAPE, *args], env=env, cwd=ROOT, capture_output=True, text=True, check=False
    )
    steps.append({"command": "landscape " + " ".join(args), "exit_code": proc.returncode})
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"step failed ({proc.returncode}): landscape {' '.join(args)}")
    return proc.stdout


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
    extracted = 0
    by_type: dict[str, list[int]] = {}
    for eid, (_name, _segs, _texts, presences, _a) in zip(ids, ENTITIES, strict=True):
        for country, ptype, via in presences:
            if via == "extract":
                by_type.setdefault(ptype, []).append(eid)
                continue
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
    for ptype, eids in by_type.items():
        args = ["presence-extract", *scope, "--type", ptype]
        for eid in sorted(set(eids)):
            args += ["--entity", str(eid)]
        stdout = landscape(args, env, steps)
        m = re.search(r"^extracted (\d+)$", stdout, re.MULTILINE)
        if m is None:
            raise SystemExit(f"no extracted count from: landscape {' '.join(args)}")
        extracted += int(m.group(1))
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
    print(
        json.dumps(
            {"steps": steps, "extracted": extracted, "lines": lines, "mosaic": mosaic},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
