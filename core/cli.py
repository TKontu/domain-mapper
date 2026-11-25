"""``landscape`` command-line entry point (Typer).

Each pipeline stage (ARCHITECTURE.md §4) is a subcommand sharing the same options. Stage bodies
are unwired by design until their own todo.md rows; ``--dry-run`` prints the resolved plan.
``validate`` (SP-02) and ``schema`` (SP-03) are not stages: they load/summarize a scope pack and
export its JSON Schemas.

``crawl`` (CL-01, CR-04, CR-05) runs for explicit ``--url`` values, or with none for the
homepage ``https://<primary_domain>/`` of each entity of the scope pack's id in id order (at most
``--limit`` entities, which counts entities rather than URLs), then that entity's
``crawl.priority_paths`` in listed order, stopping at ``crawl.max_pages`` URLs per entity
(default: homepage only). Entities are paged by :func:`core.entity_store.iter_entities` so one page
is in memory.
Each URL goes through the robots.txt check (CR-02), per-host rate limit,
:class:`~adapters.firecrawl.FirecrawlFetcher` over the stdlib transport, then
:func:`~core.ingest.ingest_page` with a :class:`~adapters.blob_store.FilesystemBlobStore` rooted at
``RAW_STORE_DIR`` and the entity's id as ``entity_id``. URLs run sequentially, one body in memory
at a time.

``web-search`` (DS-01) runs one query against SearXNG (``SEARXNG_URL``) and prints
``<rank>\t<url>\t<title>`` per result. With ``--save`` (DS-02) each result is also written as one
``candidate`` row (channel ``search``) via :func:`~core.candidate_store.record_candidates`.

``resolve`` (RV-01) is read-only: it groups unresolved ``candidate`` rows by normalised domain via
:func:`~core.resolve.group_candidates` and prints ``<domain>\t<count>\t<name>`` lines, then
``domains N candidates M`` and, when nonzero, ``skipped K``. With ``--apply`` (RV-02) it instead
links them to ``entity`` rows of the scope pack's id via :func:`~core.resolve.apply_resolution`
and prints ``created C linked L skipped K``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from itertools import chain, islice
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import typer
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from adapters.blob_store import FilesystemBlobStore, raw_store_root
from adapters.crawl_policy import HostRateLimiter, RobotsPolicy
from adapters.fetcher import DEFAULT_MAX_BYTES, Fetcher, FetchError
from adapters.firecrawl import FirecrawlFetcher
from adapters.html_parser import HtmlParser
from adapters.http_transport import get_robots, make_post_transport
from adapters.search import SearchError, SearchQuery
from adapters.searxng import SearxngSearch
from core.candidate_store import NewCandidate, record_candidates
from core.config import Settings, load_settings
from core.entity_store import EntityRef, iter_entities
from core.ingest import IngestResult, ingest_page
from core.links import extract_links, normalise_url
from core.resolve import apply_resolution, group_candidates
from core.runs import create_all
from core.scope_pack import CrawlPolicy, ScopePackError, load_scope_pack, scope_pack_json_schemas

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


UrlOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URL to crawl (repeatable; without it, each entity homepage is crawled).",
    ),
]

USER_AGENT = "domain-mapper/0.1 (polite research crawler)"
CRAWL_MIN_INTERVAL_S = 1.0
"""Minimum spacing between requests to one host; a robots ``Crawl-delay`` can only raise it."""
_HTTP_TIMEOUT_S = 60.0
_ROBOTS_TIMEOUT_S = 15.0
_SOURCE_KIND = "web"


def _load(scope: Path) -> Settings:
    if not scope.is_dir():
        typer.echo(f"error: scope directory not found: {scope}", err=True)
        raise typer.Exit(2)
    try:
        return load_settings()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


def run_stage(stage: str, scope: Path, layer: str | None, limit: int | None, dry_run: bool) -> None:
    """Resolve the plan for ``stage``; print it on dry run, otherwise fail as not implemented."""
    settings = _load(scope)
    if not dry_run:
        typer.echo(f"error: stage '{stage}' is not implemented yet", err=True)
        raise typer.Exit(1)
    _print_plan(stage, scope, layer, limit, settings)


def _print_plan(
    stage: str, scope: Path, layer: str | None, limit: int | None, settings: Settings
) -> None:
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


def run_crawl(
    scope: Path, layer: str | None, limit: int | None, dry_run: bool, urls: list[str]
) -> None:
    """Crawl ``urls`` or, without them, one homepage per entity of ``scope``'s id.

    One line per URL; exit 1 if any failed, 2 on bad setup.
    """
    settings = _load(scope)
    if dry_run:
        _print_plan("crawl", scope, layer, limit, settings)
        return
    if urls:
        _crawl_urls(settings, urls)
        return
    _crawl_entities(scope, settings, limit)


def _make_fetcher() -> FirecrawlFetcher:
    """Build the Firecrawl fetcher from the environment; a missing URL exits 2."""
    try:
        return FirecrawlFetcher.from_env(
            make_post_transport(
                max_bytes=2 * DEFAULT_MAX_BYTES + 64 * 1024, timeout_s=_HTTP_TIMEOUT_S
            )
        )
    except ValueError as exc:  # message names FIRECRAWL_URL, never its value
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


def _open_engine(settings: Settings) -> Engine:
    """Create the schema on the configured database; any failure exits 2, URL never echoed."""
    try:
        engine = create_engine(settings.database_url)
        create_all(engine)
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    return engine


def _crawl_urls(settings: Settings, urls: list[str]) -> None:
    """Crawl explicit URLs with ``entity_id`` None; exit 1 if any failed."""
    fetcher = _make_fetcher()
    engine = _open_engine(settings)
    try:
        failed = _crawl_targets(engine, fetcher, ((url, None) for url in urls))
    finally:
        engine.dispose()
    if failed:
        raise typer.Exit(1)


def _crawl_entities(scope: Path, settings: Settings, limit: int | None) -> None:
    """Crawl each entity of the scope pack id in id order, at most ``limit``, per its policy."""
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    policy = pack.scope.crawl
    engine = _open_engine(settings)
    try:
        entities = iter_entities(engine, pack.scope.id, limit=limit)
        try:
            first = next(entities)
        except StopIteration:
            typer.echo("entities 0")
            return
        # Built only once there is work: no entities needs no fetcher endpoint.
        fetcher = _make_fetcher()
        crawler = _Crawler(engine, fetcher)
        for entity in chain((first,), entities):
            _crawl_entity(crawler, entity, policy)
        failed = crawler.failed
    except SQLAlchemyError as exc:  # a page read failed: bad setup, URL never echoed
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    finally:
        engine.dispose()
    if failed:
        raise typer.Exit(1)


def _entity_targets(entity: EntityRef, policy: CrawlPolicy) -> Iterator[tuple[str, int]]:
    """Yield the homepage, then priority paths, of ``entity``; at most ``policy.max_pages``."""
    urls = chain(
        (_homepage(entity),),
        (f"https://{entity.primary_domain}{path}" for path in policy.priority_paths),
    )
    for url in islice(urls, policy.max_pages):
        yield url, entity.id


def _crawl_entity(crawler: _Crawler, entity: EntityRef, policy: CrawlPolicy) -> None:
    """Crawl one entity: homepage and priority paths, then (if enabled) discovered links.

    With ``discover_links`` the remaining ``max_pages`` budget is filled breadth-first, in
    document order, with same-host links from this entity's successfully fetched HTML pages.
    Every URL counts against the budget once, fetched or not, and none is attempted twice.
    State held: at most ``max_pages`` URLs in the seen set/frontier plus one page's links.
    """
    seen: set[str] = set()
    frontier: deque[str] = deque()
    for url, _ in _entity_targets(entity, policy):
        key = normalise_url(url) or url
        if key not in seen:
            seen.add(key)
            frontier.append(url)
    host = entity.primary_domain.lower()
    while frontier:
        url = frontier.popleft()
        result = crawler.crawl(url, entity.id)
        if not policy.discover_links or result is None or len(seen) >= policy.max_pages:
            continue
        if "html" not in result.content_type.lower():
            continue
        html = result.body_bytes.decode("utf-8", errors="replace")
        for link in extract_links(html, base_url=result.final_url, host=host):
            if len(seen) >= policy.max_pages:
                break
            if link not in seen:
                seen.add(link)
                frontier.append(link)


def _homepage(entity: EntityRef) -> str:
    """The homepage URL crawled first for every entity."""
    return f"https://{entity.primary_domain}/"


def _crawl_targets(
    engine: Engine, fetcher: Fetcher, targets: Iterable[tuple[str, int | None]]
) -> bool:
    """Fetch ``(url, entity_id)`` targets sequentially; return whether any failed."""
    crawler = _Crawler(engine, fetcher)
    for url, entity_id in targets:
        crawler.crawl(url, entity_id)
    return crawler.failed


class _Crawler:
    """Sequential fetches sharing one robots policy, per-host limiter, raw store and parser.

    Each URL goes through the same robots check, per-host wait and ``ingest_page`` call. A target
    failure is printed and recorded in :attr:`failed` and does not stop the rest.
    """

    def __init__(self, engine: Engine, fetcher: Fetcher) -> None:
        self.engine = engine
        self.fetcher = fetcher
        self.limiter = HostRateLimiter(CRAWL_MIN_INTERVAL_S, time.monotonic, time.sleep)
        limiter = self.limiter

        def fetch_robots(robots_url: str) -> tuple[int, bytes]:
            # Called only when RobotsPolicy actually fetches (not cached): one wait per real GET.
            limiter.wait(robots_url)
            return get_robots(robots_url, user_agent=USER_AGENT, timeout_s=_ROBOTS_TIMEOUT_S)

        self.robots = RobotsPolicy(fetch_robots, USER_AGENT)
        self.blob_store = FilesystemBlobStore(raw_store_root(os.environ))
        self.parser = HtmlParser()
        self.failed = False

    def crawl(self, url: str, entity_id: int | None) -> IngestResult | None:
        """Fetch and ingest ``url``; None if robots-skipped or failed."""
        try:
            host = urlsplit(url).hostname or ""
            if not self.robots.allowed(url):
                typer.echo(f"skipped {url} robots")
                return None
            delay = self.robots.crawl_delay(url)
            if delay is not None:
                self.limiter.set_interval(host, delay)
            self.limiter.wait(url)
            result = ingest_page(
                self.engine,
                fetcher=self.fetcher,
                parser=self.parser,
                url=url,
                entity_id=entity_id,
                source_kind=_SOURCE_KIND,
                blob_store=self.blob_store,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the next URL
            self.failed = True
            detail = str(exc) if isinstance(exc, FetchError) else ""
            typer.echo(f"error {url} {type(exc).__name__}{': ' + detail if detail else ''}")
            return None
        typer.echo(
            f"ok {url} snapshot={result.snapshot_id} chunks={result.chunks_written} "
            f"raw={result.raw_digest}"
        )
        return result


def run_resolve(scope: Path, apply: bool = False) -> None:
    """Print unresolved candidates grouped by normalised domain; exit 2 on a database error.

    With ``apply`` link them to entities of the scope pack id and print the write summary.
    """
    settings = _load(scope)
    if apply:
        try:
            scope_id = load_scope_pack(scope).scope.id
        except ScopePackError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
        try:
            engine = create_engine(settings.database_url)
            try:
                create_all(engine)
                applied = apply_resolution(engine, scope_id)
            finally:
                engine.dispose()
        except Exception as exc:  # noqa: BLE001 - never echo the database URL
            typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
            raise typer.Exit(2) from None
        typer.echo(f"created {applied.created} linked {applied.linked} skipped {applied.skipped}")
        return
    try:
        engine = create_engine(settings.database_url)
        try:
            create_all(engine)
            summary = group_candidates(engine)
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    for group in summary.groups:
        typer.echo(f"{group.domain}\t{group.count}\t{group.name}")
    typer.echo(f"domains {len(summary.groups)} candidates {summary.candidates}")
    if summary.skipped:
        typer.echo(f"skipped {summary.skipped}")


def _register(stage: str) -> None:
    if stage == "crawl":

        def crawl(
            scope: ScopeOpt,
            layer: LayerOpt = None,
            limit: LimitOpt = None,
            dry_run: DryRunOpt = False,
            url: UrlOpt = None,
        ) -> None:
            run_crawl(scope, layer, limit, dry_run, list(url or []))

        app.command(
            name=stage,
            help="Run the crawl stage for each --url, or per entity by its scope's crawl policy.",
        )(crawl)
        return

    def command(
        scope: ScopeOpt,
        layer: LayerOpt = None,
        limit: LimitOpt = None,
        dry_run: DryRunOpt = False,
    ) -> None:
        run_stage(stage, scope, layer, limit, dry_run)

    app.command(name=stage, help=f"Run the {stage} stage.")(command)


for _stage in STAGES:
    if _stage != "resolve":  # `resolve` has its own body below (RV-01)
        _register(_stage)


@app.command(name="resolve", help="Group unresolved candidates by normalised domain.")
def resolve(
    scope: ScopeOpt,
    apply: Annotated[
        bool, typer.Option("--apply", help="Create/link entity rows per normalised domain.")
    ] = False,
) -> None:
    """Group unresolved ``candidate`` rows by normalised domain; ``--apply`` links entities."""
    run_resolve(scope, apply)


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


@app.command(name="web-search", help="Run one SearXNG web search and print ranked results.")
def web_search(
    scope: ScopeOpt,
    query: Annotated[str, typer.Option("--query", help="Search text.")],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100, help="Max results.")] = 10,
    language: Annotated[str | None, typer.Option("--language", help="Language code.")] = None,
    site: Annotated[str | None, typer.Option("--site", help="Restrict to host.")] = None,
    save: Annotated[
        bool, typer.Option("--save", help="Persist results as candidate rows.")
    ] = False,
) -> None:
    """Print one ``rank\turl\ttitle`` line per result; exit 1 on SearchError, 2 on bad setup."""
    settings = _load(scope)
    if not query.strip():
        typer.echo("error: --query must be non-blank", err=True)
        raise typer.Exit(2)
    try:
        search = SearxngSearch.from_env()
    except ValueError as exc:  # names SEARXNG_URL, never its value
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    try:
        results = search.search(SearchQuery(text=query, limit=limit, language=language, site=site))
    except SearchError as exc:  # message omits base URL and query
        typer.echo(f"error {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from None
    for r in results:
        typer.echo(f"{r.rank}\t{r.url}\t{r.title}")
    if not save:
        return
    found_at = datetime.now(UTC)
    rows = [
        NewCandidate(name=r.title, url=r.url, channel="search", query=query, found_at=found_at)
        for r in sorted(results, key=lambda r: r.rank)
    ]
    try:
        engine = create_engine(settings.database_url)
        try:
            create_all(engine)
            saved = record_candidates(engine, rows)
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"saved {saved} candidates")


def main() -> None:
    """Console-script entry point for ``landscape``."""
    app()
