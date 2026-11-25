"""``landscape`` command-line entry point (Typer).

Each pipeline stage (ARCHITECTURE.md §4) is a subcommand sharing the same options. Stage bodies
are unwired by design until their own todo.md rows; ``--dry-run`` prints the resolved plan.
``validate`` (SP-02) and ``schema`` (SP-03) are not stages: they load/summarize a scope pack and
export its JSON Schemas.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from core.config import load_settings
from core.scope_pack import ScopePackError, load_scope_pack, scope_pack_json_schemas

STAGES: tuple[str, ...] = (
    "discover",
    "resolve",
    "screen",
    "crawl",
    "parse",
    "index",
    "assess",
    "attributes",
    "verify",
    "aggregate",
    "synthesize",
    "publish",
)

_SECRET_FIELDS = frozenset({"cloud_api_key"})

app = typer.Typer(help="Domain Mapper pipeline stages.", no_args_is_help=True)

ScopeOpt = Annotated[Path, typer.Option("--scope", help="Scope pack directory.")]
LayerOpt = Annotated[str | None, typer.Option("--layer", help="Market layer to run.")]
LimitOpt = Annotated[int | None, typer.Option("--limit", min=1, help="Max entities to process.")]
DryRunOpt = Annotated[bool, typer.Option("--dry-run", help="Print the plan and exit.")]


def run_stage(stage: str, scope: Path, layer: str | None, limit: int | None, dry_run: bool) -> None:
    """Resolve the plan for ``stage``; print it on dry run, otherwise fail as not implemented."""
    if not scope.is_dir():
        typer.echo(f"error: scope directory not found: {scope}", err=True)
        raise typer.Exit(2)
    try:
        settings = load_settings()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    if not dry_run:
        typer.echo(f"error: stage '{stage}' is not implemented yet", err=True)
        raise typer.Exit(1)
    typer.echo(f"stage: {stage}")
    typer.echo(f"scope: {scope}")
    typer.echo(f"layer: {layer if layer is not None else '(all)'}")
    typer.echo(f"limit: {limit if limit is not None else '(none)'}")
    typer.echo("settings:")
    for f in dataclasses.fields(settings):
        if f.name not in _SECRET_FIELDS:
            value = getattr(settings, f.name)
            if f.name == "database_url":
                value = _mask_database_url(value)
            typer.echo(f"  {f.name}: {value}")


def _mask_database_url(url: str) -> str:
    """Render ``url`` with its password hidden; never echo an unparsable value."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except ArgumentError:
        return "(unparsable database URL; value hidden)"


def _register(stage: str) -> None:
    def command(
        scope: ScopeOpt,
        layer: LayerOpt = None,
        limit: LimitOpt = None,
        dry_run: DryRunOpt = False,
    ) -> None:
        run_stage(stage, scope, layer, limit, dry_run)

    app.command(name=stage, help=f"Run the {stage} stage.")(command)


for _stage in STAGES:
    _register(_stage)


@app.command(name="validate", help="Validate a scope pack and print a summary.")
def validate(scope: ScopeOpt) -> None:
    """Load ``scope`` with ``load_scope_pack``; exit 1 on an invalid pack, 2 if missing."""
    if not scope.is_dir():
        typer.echo(f"error: scope directory not found: {scope}", err=True)
        raise typer.Exit(2)
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    topics = pack.rubric.topics
    typer.echo(f"scope: {pack.scope.id}")
    typer.echo(f"segments: {len(pack.scope.segments)}")
    typer.echo(f"layers: {len(pack.scope.layers)}")
    typer.echo(f"topics: {len(topics)}")
    typer.echo(f"criteria: {sum(len(t.criteria) for t in topics)}")
    optional = {
        "vocabulary": len(pack.vocabulary.entries),
        "sources": len(pack.sources.sources),
        "attributes": len(pack.attributes.attributes),
        "profiles": len(pack.profiles.profiles),
        "views": len(pack.views.views),
    }
    for name, count in optional.items():
        present = (scope / f"{name}.yaml").is_file()
        typer.echo(f"{name}: {count if present else 'absent'}")


@app.command(name="schema", help="Export scope-pack JSON Schemas for editor autocompletion.")
def schema(
    out: Annotated[
        Path | None, typer.Option("--out", help="Write <file stem>.schema.json files here.")
    ] = None,
    file: Annotated[
        str, typer.Option("--file", help="Pack file to print when --out is not given.")
    ] = "scope.yaml",
) -> None:
    """Write every schema to ``out``, or print the one for ``file``; unknown file exits 2."""
    schemas = scope_pack_json_schemas()
    if out is None:
        if file not in schemas:
            known = ", ".join(schemas)
            typer.echo(f"error: unknown scope-pack file: {file} (known: {known})", err=True)
            raise typer.Exit(2)
        typer.echo(json.dumps(schemas[file], indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    for name, body in schemas.items():
        target = out / f"{Path(name).stem}.schema.json"
        target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        typer.echo(str(target))


def main() -> None:
    """Console-script entry point for ``landscape``."""
    app()
