"""``landscape`` command-line entry point (Typer).

Each pipeline stage (ARCHITECTURE.md §4) is a subcommand sharing the same options. Stage bodies
are unwired by design until their own todo.md rows; ``--dry-run`` prints the resolved plan.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from core.config import load_settings

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


def main() -> None:
    """Console-script entry point for ``landscape``."""
    app()
