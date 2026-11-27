"""``landscape`` command-line entry point (Typer).

Each pipeline stage (ARCHITECTURE.md §4) is a subcommand sharing the same options. Stage bodies
are unwired by design until their own todo.md rows; ``--dry-run`` prints the resolved plan.
``validate`` (SP-02) and ``schema`` (SP-03) are not stages: they load/summarize a scope pack and
export its JSON Schemas.

``crawl`` (CL-01, CR-04, CR-05) runs for explicit ``--url`` values, or with none for the
homepage ``https://<primary_domain>/`` of each entity of the scope pack's id in id order (at most
``--limit`` entities, which counts entities rather than URLs), then that entity's
``crawl.priority_paths`` in listed order, stopping at ``crawl.max_pages`` URLs per entity
(default: homepage only). ``--entity-scheme http`` builds those entity URLs as ``http://``
instead, a test/live aid for loopback fakes; it exits 2 with ``--url`` and stays ``https`` by
default. Entities are paged by :func:`core.entity_store.iter_entities` so one page is in memory.
Each URL goes through the robots.txt check (CR-02), per-host rate limit,
:class:`~adapters.firecrawl.FirecrawlFetcher` over the stdlib transport, then
:func:`~core.ingest.ingest_page` with a :class:`~adapters.blob_store.FilesystemBlobStore` rooted at
``RAW_STORE_DIR`` and the entity's id as ``entity_id``. URLs run sequentially, one body in memory
at a time. A fetched response whose status is outside 200-299 writes nothing; it prints
``skipped <url> status=<code>``, is not a failure, and the crawl continues with the next URL.
A transient fetcher transport error prints ``retry <url> <ErrorName>: <detail>``, sleeps
:data:`CRAWL_RETRY_BACKOFF_S`, waits on the per-host limiter and fetches the URL once more; the
second outcome is handled as above (or as an error).

``web-search`` (DS-01) runs one query against SearXNG (``SEARXNG_URL``) and prints
``<rank>\t<url>\t<title>`` per result. With ``--save`` (DS-02) each result is also written as one
``candidate`` row (channel ``search``) via :func:`~core.candidate_store.record_candidates`.

``resolve`` (RV-01) is read-only: it groups unresolved ``candidate`` rows by normalised domain via
:func:`~core.resolve.group_candidates` and prints ``<domain>\t<count>\t<name>`` lines, then
``domains N candidates M`` and, when nonzero, ``skipped K``. With ``--apply`` (RV-02) it instead
links them to ``entity`` rows of the scope pack's id via :func:`~core.resolve.apply_resolution`
and prints ``created C linked L skipped K``.

``screen`` (SC-02) streams the scope's ``unscreened`` entities in id order and classifies each
with the small local model via :func:`~core.screen.screen_entity` over that entity's stored chunk
text. One ``screen_result`` row is written and the entity status moved to ``in_scope``,
``out_of_scope`` or ``borderline`` in one transaction per entity. It prints ``ok <id> <status>
<confidence> <segments>``, ``skipped <id> no-text`` for an entity without chunk text, and
``error <id> <ErrorName>: <message>`` for invalid model output or a failed call (the entity stays
``unscreened`` and the run exits 1). Spend is capped per run by ``DM_LLM_BUDGET_USD`` (LB-01):
``LlmBudgetExceeded`` stops screening, leaving that and later entities ``unscreened``; the last
line is ``screened S skipped K errors E``.

``index`` (IX-02, IX-05) embeds pending chunks: settings load, the embedder builds through the
test seam :func:`_make_embedder` (``build_embedder(settings, model)``), a ``run`` row (stage
``index``) opens via :func:`_open_run`, and :func:`~core.embed_store.embed_pending_chunks` runs
with the ``--model``/``--batch-size``/``--limit`` options. On success it prints ``embedded <n>
batches <m>``; on ``EmbeddingError``/``EmbedStoreError`` it prints ``error <ErrorName>: <message>``
on stderr, closes the run ``failed`` and exits 1. A ``ValueError`` building the embedder (an
unconfigured service) prints ``error: <message>`` on stderr and exits 2 before any run row exists;
any other error or interrupt after the run opens closes it ``failed`` before propagating.

``search`` (IX-04) prints one line per BM25 hit from :func:`~core.keyword_search.search_chunks`:
``<score> chunk <chunk_id> snapshot <snapshot_id> seq <seq>`` plus `` <heading > ...`` when the
chunk has a heading path, or ``no hits`` on stdout when it has none. Read-only: no ``run`` row and
nothing is written. Exit 2 on an empty or whitespace query or a ``--limit`` below 1, before the
database is opened.

``export`` (AS-12) writes the ``assessments`` current view (AS-11) to ``--out`` as JSON Lines:
one object per latest assessment with its evidence, in :func:`~core.assessment_store.
current_assessments` order and filtered the same way. Rows are written one at a time to
``<out>.tmp``, renamed to ``<out>`` on success, then ``exported N assessments to PATH`` prints
(N may be 0, exit 0). A format other than ``json``, a missing scope or an unwritable output
directory exits 2 and leaves no ``PATH``.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import functools
import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from itertools import chain, islice
from pathlib import Path
from typing import Annotated, Any, NoReturn
from urllib.parse import urlsplit

import typer
from sqlalchemy import func, or_, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session

from adapters.blob_store import FilesystemBlobStore, raw_store_root
from adapters.crawl_policy import HostRateLimiter, RobotsPolicy
from adapters.embedding import Embedder, EmbeddingError, Reranker, build_embedder, build_reranker
from adapters.fetcher import DEFAULT_MAX_BYTES, Fetcher, FetchError, TransientFetchError
from adapters.firecrawl import FirecrawlFetcher
from adapters.html_parser import HtmlParser
from adapters.http_transport import get_robots, make_post_transport
from adapters.llm import LLM, DbTarget, LlmBudgetExceeded, LlmError, build_llm
from adapters.search import SearchError, SearchQuery
from adapters.searxng import SearxngSearch
from app.mosaic import (
    entity_slug,
    filter_by_layers,
    filter_by_segments,
    render_entity_page,
    render_mosaic,
)
from core.aggregate import TopicScore, score_topics
from core.assess import AssessError, extract_criterion
from core.assessment_store import (
    CurrentAssessment,
    current_assessments,
    iter_current_assessments,
    record_assessment,
)
from core.attribute_extract import (
    AttributeExtractError,
    AttributeExtraction,
    attribute_query,
    extract_attribute,
    store_attribute_extraction,
)
from core.attribute_store import (
    attribute_evidence_many,
    current_attribute_value_rows_many,
    current_attribute_values,
    current_attribute_values_many,
    record_attribute_value,
)
from core.candidate_store import Candidate, NewCandidate, record_candidates
from core.chunk_store import ChunkRow
from core.config import Settings, load_settings
from core.embed_store import EmbedStats, EmbedStoreError, embed_pending_chunks
from core.entity_store import STATUS_UNSCREENED, Entity, EntityRef, iter_entities
from core.ingest import IngestResult, NonSuccessStatusError, ingest_page
from core.keyword_search import search_chunks
from core.links import crawl_key, extract_links
from core.llm_calls import CostSummaryRow, summarise_llm_calls
from core.log import configure_logging
from core.presence_extract import CompiledTerms, compile_terms, extract_presence
from core.presence_store import (
    PRESENCE_TYPES,
    check_presence,
    layer_ids_for,
    presence_evidence_many,
    presences_many,
    record_presence,
)
from core.resolve import ApplySummary, apply_resolution, group_candidates
from core.retrieve import _with_texts, retrieve_evidence
from core.runs import (
    FAILED,
    FINISHED,
    RUNNING,
    Run,
    StageRunner,
    create_all,
    create_db_engine,
)
from core.scope_pack import (
    Attribute,
    CrawlPolicy,
    Criterion,
    Rubric,
    ScopePack,
    ScopePackError,
    load_scope_pack,
    scope_pack_json_schemas,
)
from core.screen import ScreenError, ScreenResult, screen_entity
from core.screen_store import (
    DEFAULT_MAX_CHARS,
    STATUS_IN_SCOPE,
    latest_screen_segments,
    latest_screen_segments_many,
    load_entity_text,
    record_screen_result,
)
from core.snapshot_store import SnapshotRow
from core.source_catalogue import seed_sources
from core.verify import route, verify_extraction

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

_SECRET_FIELDS = frozenset({"cloud_api_key", "llm_gateway_api_key"})

app = typer.Typer(help="Domain Mapper pipeline stages.", no_args_is_help=True)

_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


@app.callback()
def _configure(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="DM_LOG_LEVEL",
            help="Root log level (CRITICAL, ERROR, WARNING, INFO, DEBUG).",
        ),
    ] = "INFO",
) -> None:
    """Configure structured one-line logging to stderr once per invocation (RT-02)."""
    level = log_level.strip().upper()
    if level not in _LOG_LEVELS:
        typer.echo(f"error: invalid log level {log_level}", err=True)
        raise typer.Exit(2)
    configure_logging(level)


ScopeOpt = Annotated[Path, typer.Option("--scope", help="Scope pack directory.")]
LayerOpt = Annotated[str | None, typer.Option("--layer", help="Market layer to run.")]
LimitOpt = Annotated[int | None, typer.Option("--limit", min=1, help="Max entities to process.")]
IndexModelOpt = Annotated[
    str, typer.Option("--model", help="Embedding model name passed to the embedding service.")
]
IndexBatchSizeOpt = Annotated[
    int, typer.Option("--batch-size", min=1, help="Chunks per embed call (default 64).")
]
IndexLimitOpt = Annotated[
    int | None, typer.Option("--limit", min=0, help="Max chunks to embed; 0 embeds none.")
]
DryRunOpt = Annotated[bool, typer.Option("--dry-run", help="Print the plan and exit.")]
JobsOpt = Annotated[
    int, typer.Option("--jobs", min=1, max=8, help="Concurrent screen LLM calls (1-8).")
]


UrlOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URL to crawl (repeatable; without it, each entity homepage is crawled).",
    ),
]


class EntityScheme(StrEnum):
    """Scheme for entity homepage and priority-path URLs (CR-13).

    ``http`` is a test/live aid for loopback fakes; remove the option once a live check can run
    against a real ``https`` host.
    """

    https = "https"
    http = "http"


EntitySchemeOpt = Annotated[
    EntityScheme | None,
    typer.Option(
        "--entity-scheme",
        help="URL scheme for entity targets; http only for loopback test fakes.",
        show_default="https",
    ),
]

USER_AGENT = "domain-mapper/0.1 (polite research crawler)"
CRAWL_MIN_INTERVAL_S = 1.0
"""Minimum spacing between requests to one host; a robots ``Crawl-delay`` can only raise it."""
CRAWL_RETRY_BACKOFF_S = 5.0
"""Seconds slept before the single retry of a transient fetcher transport error."""
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
    scope: Path,
    layer: str | None,
    limit: int | None,
    dry_run: bool,
    urls: list[str],
    entity_scheme: EntityScheme | None = None,
) -> None:
    """Crawl ``urls`` or, without them, one homepage per entity of ``scope``'s id.

    ``entity_scheme`` selects the entity URL scheme (default ``https``) and exits 2 when combined
    with ``urls``. One line per URL; exit 1 if any failed, 2 on bad setup.
    """
    if urls and entity_scheme is not None:
        typer.echo("error: --entity-scheme cannot be combined with --url", err=True)
        raise typer.Exit(2)
    scheme = (entity_scheme or EntityScheme.https).value
    settings = _load(scope)
    if dry_run:
        _print_plan("crawl", scope, layer, limit, settings)
        return
    if urls:
        _crawl_urls(settings, urls)
        return
    _crawl_entities(scope, settings, limit, scheme)


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
        engine = create_db_engine(settings.database_url)
        create_all(engine)
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    return engine


def _crawl_urls(settings: Settings, urls: list[str]) -> None:
    """Crawl explicit URLs with ``entity_id`` None; exit 1 if any failed, 2 on a database error."""
    fetcher = _make_fetcher()
    engine = _open_engine(settings)
    run_id: int | None = None
    try:
        run_id = _open_run(engine, "crawl")
        try:
            failed = _crawl_targets(engine, fetcher, ((url, None) for url in urls), run_id=run_id)
        except SQLAlchemyError:
            raise  # closed best-effort below
        except BaseException as exc:
            # An unexpected error or interrupt must not leave the run row open (CR-14); a
            # failed close names itself and never replaces the original exception (CR-17).
            _close_run_aborting(engine, run_id, exc)
            raise
        _close_run(engine, run_id, FAILED if failed else FINISHED)
    except SQLAlchemyError as exc:  # CR-16: like entity mode, URL never echoed
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()
    if failed:
        raise typer.Exit(1)


def _crawl_entities(
    scope: Path, settings: Settings, limit: int | None, scheme: str = "https"
) -> None:
    """Crawl each entity of the scope pack id in id order, at most ``limit``, per its policy."""
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    policy = pack.scope.crawl
    engine = _open_engine(settings)
    run_id: int | None = None
    skipped = 0
    try:
        entities = iter_entities(engine, pack.scope.id, limit=limit)
        try:
            first = next(entities)
        except StopIteration:
            typer.echo("entities 0")  # no work: no run row
            return
        # Built only once there is work: no entities needs no fetcher endpoint. The fetcher is
        # built before the run row opens, so a missing FIRECRAWL_URL exits 2 with no run row
        # (CR-14); the run row is opened before the crawler is constructed, so the snapshots the
        # crawler writes carry the run id (CR-15).
        fetcher = _make_fetcher()
        run_id = _open_run(engine, "crawl")
        crawler = _Crawler(engine, fetcher, run_id=run_id)
        try:
            skipped = _run_crawl_entity_units(
                engine, crawler, chain((first,), entities), pack.scope.id, policy, scheme, run_id
            )
        except SQLAlchemyError:
            raise  # closed best-effort below (CR-18)
        except BaseException as exc:
            # An unexpected error, interrupt or page-read failure must not leave the run
            # open; a failed close names itself and never replaces the original exception.
            _close_run_aborting(engine, run_id, exc)
            raise
        failed = crawler.failed
        _close_run(engine, run_id, FAILED if failed else FINISHED)
    except SQLAlchemyError as exc:  # a page read failed: bad setup, URL never echoed
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()
    if skipped:
        typer.echo(f"skipped {skipped}")
    if failed:
        raise typer.Exit(1)


class _EntityCrawlFailed(Exception):
    """Raised by the crawl unit handler so StageRunner records the entity's unit failed."""


def _run_crawl_entity_units(
    engine: Engine,
    crawler: _Crawler,
    entities: Iterable[EntityRef],
    scope_id: str,
    policy: CrawlPolicy,
    scheme: str,
    run_id: int,
) -> int:
    """RT-01: one :class:`core.runs.StageRunner` unit per entity under the crawl run ``run_id``.

    Stage ``crawl``, unit key ``"<entity_id>"``, input hash = sha256 over the scope id, the
    entity's primary domain, the URL scheme and the canonical JSON of the crawl policy. A unit
    already ``done`` for the same hash is skipped (nothing fetched). An entity with any failed
    URL marks its unit failed (retried next time) and sets ``crawler.failed``; the rest
    continue. Entities stream one at a time. Returns the skipped count.
    """
    policy_json = json.dumps(policy.model_dump(mode="json"), sort_keys=True)
    current: dict[str, EntityRef] = {}

    def units() -> Iterator[tuple[str, str]]:
        for entity in entities:
            key = str(entity.id)
            current.clear()
            current[key] = entity
            digest = hashlib.sha256(
                f"{scope_id}\0{entity.primary_domain}\0{scheme}\0{policy_json}".encode()
            ).hexdigest()
            yield key, digest

    any_failed = False

    def handle(key: str) -> None:
        nonlocal any_failed
        crawler.failed = False
        _crawl_entity(crawler, current[key], policy, scheme)
        if crawler.failed:
            any_failed = True
            raise _EntityCrawlFailed(f"entity {key} had a failed URL")

    try:
        summary = StageRunner(engine).run(
            "crawl", units(), handle, run_id=run_id, catch=(_EntityCrawlFailed,)
        )
    finally:
        crawler.failed = any_failed
    return summary.skipped


def _entity_targets(
    entity: EntityRef, policy: CrawlPolicy, scheme: str = "https"
) -> Iterator[tuple[str, int]]:
    """Yield the homepage, then priority paths, of ``entity``; at most ``policy.max_pages``."""
    urls = chain(
        (_homepage(entity, scheme),),
        (f"{scheme}://{entity.primary_domain}{path}" for path in policy.priority_paths),
    )
    for url in islice(urls, policy.max_pages):
        yield url, entity.id


def _crawl_entity(
    crawler: _Crawler, entity: EntityRef, policy: CrawlPolicy, scheme: str = "https"
) -> None:
    """Crawl one entity: homepage and priority paths, then (if enabled) discovered links.

    With ``discover_links`` the remaining ``max_pages`` budget is filled breadth-first, in
    document order, with same-host links from this entity's successfully fetched HTML pages.
    Every URL counts against the budget once, fetched or not, and none is attempted twice.
    State held: at most ``max_pages`` URLs in the seen set/frontier plus one page's links.
    """
    seen: set[str] = set()
    frontier: deque[str] = deque()
    for url, _ in _entity_targets(entity, policy, scheme):
        key = crawl_key(url) or url
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
            key = crawl_key(link) or link
            if key not in seen:
                seen.add(key)
                frontier.append(link)


def _homepage(entity: EntityRef, scheme: str = "https") -> str:
    """The homepage URL crawled first for every entity."""
    return f"{scheme}://{entity.primary_domain}/"


def _crawl_targets(
    engine: Engine,
    fetcher: Fetcher,
    targets: Iterable[tuple[str, int | None]],
    run_id: int | None = None,
) -> bool:
    """Fetch ``(url, entity_id)`` targets sequentially; return whether any failed."""
    crawler = _Crawler(engine, fetcher, run_id=run_id)
    for url, entity_id in targets:
        crawler.crawl(url, entity_id)
    return crawler.failed


class _Crawler:
    """Sequential fetches sharing one robots policy, per-host limiter, raw store and parser.

    ``run_id`` (CR-15), when given, is the id of the invocation's ``crawl`` run row and is
    passed to every ``ingest_page`` call, so each ``snapshot`` row written by this crawler
    carries it.

    Each URL goes through the same robots check, per-host wait and ``ingest_page`` call. A target
    failure is printed and recorded in :attr:`failed` and does not stop the rest. A non-2xx
    status is printed as a skip, writes nothing and is not recorded in :attr:`failed`. A
    transient fetcher transport error is retried once after :data:`CRAWL_RETRY_BACKOFF_S` and a
    fresh limiter wait; a second failure takes the normal error path. ``sleep`` is injected so
    tests never wait.
    """

    def __init__(
        self,
        engine: Engine,
        fetcher: Fetcher,
        *,
        sleep: Callable[[float], None] = time.sleep,
        run_id: int | None = None,
    ) -> None:
        self.engine = engine
        self.fetcher = fetcher
        self.run_id = run_id
        self._sleep = sleep
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
        """Fetch and ingest ``url``; None if robots-skipped, non-2xx or failed."""
        try:
            host = urlsplit(url).hostname or ""
            if not self.robots.allowed(url):
                typer.echo(f"skipped {url} robots")
                return None
            delay = self.robots.crawl_delay(url)
            if delay is not None:
                self.limiter.set_interval(host, delay)
            self.limiter.wait(url)
            try:
                result = self._ingest(url, entity_id)
            except TransientFetchError as exc:
                typer.echo(f"retry {url} {type(exc).__name__}: {exc}")
                self._sleep(CRAWL_RETRY_BACKOFF_S)
                self.limiter.wait(url)
                result = self._ingest(url, entity_id)
        except NonSuccessStatusError as exc:
            typer.echo(f"skipped {url} status={exc.status}")
            return None
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

    def _ingest(self, url: str, entity_id: int | None) -> IngestResult:
        return ingest_page(
            self.engine,
            fetcher=self.fetcher,
            parser=self.parser,
            url=url,
            entity_id=entity_id,
            source_kind=_SOURCE_KIND,
            blob_store=self.blob_store,
            run_id=self.run_id,
        )


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
            engine = create_db_engine(settings.database_url)
            try:
                create_all(engine)
                applied, skipped = _run_resolve_unit(engine, scope_id)
            finally:
                engine.dispose()
        except Exception as exc:  # noqa: BLE001 - never echo the database URL
            typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
            raise typer.Exit(2) from None
        typer.echo(f"created {applied.created} linked {applied.linked} skipped {applied.skipped}")
        if skipped:
            typer.echo(f"skipped {skipped}")
        return
    try:
        engine = create_db_engine(settings.database_url)
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


def _run_resolve_unit(engine: Engine, scope_id: str) -> tuple[ApplySummary, int]:
    """RT-01: run ``apply_resolution`` as one :class:`core.runs.StageRunner` unit under one run.

    Stage ``resolve``, unit key ``"<scope_id>"``, input hash = sha256 over the scope id and the
    candidate count + max candidate id (one aggregate query). A unit already ``done`` for the
    same hash is skipped (``apply_resolution`` not called, zero summary). The run row is
    FINISHED on success; any error fails the unit and the run, then propagates. Returns
    ``(summary, skipped)``.
    """
    run_id = _open_run(engine, "resolve")
    applied = ApplySummary(created=0, linked=0, skipped=0)

    def units() -> Iterator[tuple[str, str]]:
        with engine.connect() as conn:
            n, top = conn.execute(select(func.count(Candidate.id), func.max(Candidate.id))).one()
        yield scope_id, hashlib.sha256(f"{scope_id}\0{n}\0{top}".encode()).hexdigest()

    def handle(_key: str) -> None:
        nonlocal applied
        applied = apply_resolution(engine, scope_id)

    try:
        summary = StageRunner(engine).run("resolve", units(), handle, run_id=run_id, catch=())
    except BaseException as exc:
        _close_run_aborting(engine, run_id, exc)
        raise
    _close_run(engine, run_id, FINISHED)
    return applied, summary.skipped


def _make_screen_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the small local model used for screening; the test seam for a fake adapter.

    One budget per invocation (= per run): ``settings.llm_budget_usd`` caps the run's spend.
    """
    return build_llm(
        settings,
        "local",
        db=db,
        budget_usd=settings.llm_budget_usd,
        timeout=settings.llm_timeout_s,
        max_tokens=settings.local_max_tokens,
    )


def _open_run(engine: Engine, stage: str) -> int:
    """Insert one ``run`` row for this invocation and return its id."""
    with Session(engine) as session, session.begin():
        run = Run(stage=stage, status=RUNNING, started_at=datetime.now(UTC))
        session.add(run)
        session.flush()
        return run.id


def _close_run(engine: Engine, run_id: int, status: str) -> None:
    """Mark the invocation's run row ``finished``/``failed`` with its end time."""
    with Session(engine) as session, session.begin():
        run = session.get(Run, run_id)
        if run is None:  # pragma: no cover - the row was just created
            raise RuntimeError(f"run {run_id} disappeared")
        run.status = status
        run.finished_at = datetime.now(UTC)


def _exit_database_unavailable(
    engine: Engine, run_id: int | None, exc: SQLAlchemyError
) -> NoReturn:
    """Close ``run_id`` FAILED best-effort, print one ``database unavailable`` line, exit 2.

    A second database failure during the close stays silent so exactly one line is printed and
    the database URL is never echoed (CR-16, CR-18).
    """
    if run_id is not None:
        try:
            _close_run(engine, run_id, FAILED)
        except SQLAlchemyError:
            pass
    typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
    raise typer.Exit(2) from None


def _close_run_aborting(engine: Engine, run_id: int | None, exc: BaseException) -> None:
    """Best-effort FAILED close for a crawl that aborts with ``exc`` (CR-17).

    If the close raises ``SQLAlchemyError``, one stderr line names it (the database URL is
    never echoed); the caller then re-raises ``exc``, so a close failure never replaces the
    original exception.
    """
    if run_id is None:
        return
    try:
        _close_run(engine, run_id, FAILED)
    except SQLAlchemyError as close_exc:
        typer.echo(f"error: database unavailable ({type(close_exc).__name__})", err=True)


class _ScreenBudgetStop(Exception):
    """Raised by the screen handler so StageRunner records the unit failed and stops."""


def run_screen(
    scope: Path, layer: str | None, limit: int | None, dry_run: bool, jobs: int = 1
) -> None:
    """Screen each ``unscreened`` entity of ``scope``'s id with the small local model.

    Entities in id order, at most ``limit``. Prints one line per entity and the final
    ``screened S skipped K errors E``. Exits 1 if any entity errored, 2 on bad setup.

    With ``jobs`` > 1 (LM-04) up to ``jobs`` ``screen_entity`` calls run on a bounded thread pool,
    prefetched at most ``jobs`` units ahead of the StageRunner; results, output lines and unit
    records stay on the main thread in id order. A budget stop cancels not-yet-started calls;
    running ones finish and their results are discarded (those entities stay unscreened).
    """
    settings = _load(scope)
    if dry_run:
        _print_plan("screen", scope, layer, limit, settings)
        return
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    engine = _open_engine(settings)
    run_id: int | None = None
    try:
        try:
            llm = _make_screen_llm(settings, engine)
        except (ValueError, LlmError) as exc:  # unbuildable backend: bad setup, exit 2
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None

        run_id = _open_run(engine, "screen")
        screened = skipped = errors = 0
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs) if jobs > 1 else None
        # Units handed to the StageRunner, keyed by unit key: entity and a result getter.
        current: dict[str, tuple[EntityRef, Callable[[], ScreenResult]]] = {}
        # Prefetch window in id order: ``(entity, None)`` is a no-text skip.
        window: deque[tuple[EntityRef, tuple[str, str, Callable[[], ScreenResult]] | None]] = (
            deque()
        )
        futures: list[concurrent.futures.Future[ScreenResult]] = []

        def call(entity: EntityRef, text: str) -> ScreenResult:
            return screen_entity(
                llm,
                pack,
                name=entity.name,
                domain=entity.primary_domain,
                text=text,
                run_id=run_id,
                max_chars=DEFAULT_MAX_CHARS,
            )

        def units() -> Iterator[tuple[str, str]]:
            nonlocal skipped
            source = iter_entities(engine, pack.scope.id, limit=limit, status=STATUS_UNSCREENED)
            exhausted = False

            def fill() -> None:
                nonlocal exhausted
                while not exhausted and sum(1 for _, u in window if u) < max(jobs, 1):
                    entity = next(source, None)
                    if entity is None:
                        exhausted = True
                        return
                    text = load_entity_text(engine, entity.id)
                    if not text.strip():
                        window.append((entity, None))
                        continue
                    digest = hashlib.sha256(f"{pack.scope.id}\0{text}".encode()).hexdigest()
                    getter: Callable[[], ScreenResult]
                    if pool is None:
                        getter = functools.partial(call, entity, text)
                    else:
                        future = pool.submit(call, entity, text)
                        futures.append(future)
                        getter = future.result
                    window.append((entity, (str(entity.id), digest, getter)))

            while True:
                fill()
                if not window:
                    return
                entity, unit = window.popleft()
                if unit is None:
                    # Not recorded as a unit: it is retried once text arrives.
                    typer.echo(f"skipped {entity.id} no-text")
                    skipped += 1
                    continue
                key, digest, getter = unit
                current.clear()
                current[key] = (entity, getter)
                yield key, digest

        def handle(key: str) -> None:
            nonlocal screened, errors
            entity, getter = current.pop(key)
            try:
                result = getter()
            except LlmBudgetExceeded as exc:
                # The run's budget is spent: stop cleanly, the rest stay unscreened.
                typer.echo(f"error {entity.id} {type(exc).__name__}: {exc}")
                errors += 1
                raise _ScreenBudgetStop from exc
            except (ScreenError, LlmError) as exc:
                # Invalid output or a failed call leaves the entity unscreened for a retry.
                typer.echo(f"error {entity.id} {type(exc).__name__}: {exc}")
                errors += 1
                raise
            status = record_screen_result(engine, entity_id=entity.id, run_id=run_id, result=result)
            segments = ",".join(result.segments) if result.segments else "-"
            typer.echo(f"ok {entity.id} {status} {result.confidence:.2f} {segments}")
            screened += 1

        try:
            try:
                StageRunner(engine).run(
                    "screen", units(), handle, run_id=run_id, catch=(ScreenError, LlmError)
                )
            except _ScreenBudgetStop:
                pass
            finally:
                if pool is not None:
                    # Not-yet-started calls are cancelled; running ones finish, results dropped.
                    for future in futures:
                        future.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
        except SQLAlchemyError:
            raise  # the outer handler closes the run best-effort and prints one line (AB-01)
        except BaseException as exc:
            _close_run_aborting(engine, run_id, exc)
            raise
        _close_run(engine, run_id, FAILED if errors else FINISHED)
    except SQLAlchemyError as exc:  # a paged read or write failed: bad setup, URL never echoed
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()

    typer.echo(f"screened {screened} skipped {skipped} errors {errors}")
    if errors:
        raise typer.Exit(1)


def _make_embedder(settings: Settings, model: str) -> Embedder:
    """Build the embedder for the Index stage; the test seam for a fake adapter."""
    return build_embedder(settings, model)


def _make_reranker(settings: Settings, model: str) -> Reranker:
    """Build the Assess reranker (``DM_RERANKER_URL``); the test seam for a fake adapter."""
    return build_reranker(settings, model)


def _run_index_unit(
    engine: Engine,
    run_id: int,
    embedder: Embedder,
    model: str,
    batch_size: int,
    limit: int | None,
) -> tuple[EmbedStats, int]:
    """RT-01: run ``embed_pending_chunks`` as one :class:`core.runs.StageRunner` unit.

    Stage ``index``, unit key ``"<model>"``, input hash = sha256 over the model, the pending
    chunk count + max pending chunk id for that model (one aggregate query, the
    ``core.embed_store`` pending predicate), ``limit``, ``batch_size`` and, while any chunk is
    pending, the run id (so pending work never skips). A unit already
    ``done`` for the same hash is skipped (no embedding, zero stats). Errors fail the unit and
    propagate; the caller owns the run row. Returns ``(stats, skipped)``.
    """
    stats = EmbedStats(embedded=0, batches=0)

    def units() -> Iterator[tuple[str, str]]:
        pending = or_(ChunkRow.embedding_model.is_(None), ChunkRow.embedding_model != model)
        with engine.connect() as conn:
            n, top = conn.execute(
                select(func.count(ChunkRow.id), func.max(ChunkRow.id)).where(pending)
            ).one()
        # Pending chunks never match a done unit (a model switch can recreate an old pending
        # state); with none pending the hash is stable so an unchanged rerun skips.
        nonce = run_id if n else ""
        raw = f"{model}\0{n}\0{top}\0{limit}\0{batch_size}\0{nonce}"
        yield model, hashlib.sha256(raw.encode()).hexdigest()

    def handle(_key: str) -> None:
        nonlocal stats
        stats = embed_pending_chunks(
            engine, embedder, model=model, batch_size=batch_size, limit=limit
        )

    summary = StageRunner(engine).run("index", units(), handle, run_id=run_id, catch=())
    return stats, summary.skipped


def run_index(model: str, batch_size: int, limit: int | None) -> None:
    """Embed pending chunks via the configured embedding service inside a ``run`` row (IX-02).

    Prints ``embedded <n> batches <m>`` on success; ``EmbeddingError``/``EmbedStoreError`` print
    ``error <ErrorName>: <message>`` on stderr with the run ``failed`` and exit 1. The embedder is
    built before the run row opens, so an unbuildable backend (``ValueError``) exits 2 with
    ``error: <message>`` and no run row; any other error or interrupt after that closes the run
    ``failed`` before propagating (IX-05). A failed abort close prints one ``database unavailable``
    line without replacing the original exception; a ``SQLAlchemyError`` exits 2 with one such
    line and no database URL (AB-01).
    """
    if not model.strip():
        typer.echo("error: --model must be non-empty", err=True)
        raise typer.Exit(2)
    if batch_size < 1:
        typer.echo("error: --batch-size must be at least 1", err=True)
        raise typer.Exit(2)
    if limit is not None and limit < 0:
        typer.echo("error: --limit must be non-negative", err=True)
        raise typer.Exit(2)
    settings = load_settings()
    engine = _open_engine(settings)
    run_id: int | None = None
    try:
        try:
            embedder = _make_embedder(settings, model)
        except ValueError as exc:  # unbuildable backend: bad setup, exit 2, no run row
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
        run_id = _open_run(engine, "index")
        try:
            stats, skipped = _run_index_unit(engine, run_id, embedder, model, batch_size, limit)
        except (EmbeddingError, EmbedStoreError) as exc:
            typer.echo(f"error {type(exc).__name__}: {exc}", err=True)
            _close_run(engine, run_id, FAILED)
            raise typer.Exit(1) from None
        except SQLAlchemyError:
            raise  # the outer handler closes the run best-effort and prints one line (AB-01)
        except BaseException as exc:
            # An unexpected error or interrupt must not leave the run row open.
            _close_run_aborting(engine, run_id, exc)
            raise
        _close_run(engine, run_id, FINISHED)
        typer.echo(f"embedded {stats.embedded} batches {stats.batches}")
        if skipped:
            typer.echo(f"skipped {skipped}")
    except SQLAlchemyError as exc:  # URL never echoed; a failed close stays silent (AB-01)
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()


def run_search(query: str, entity_id: int | None, limit: int) -> None:
    """Print one line per BM25 hit of :func:`search_chunks`; read-only, exit 2 on bad input.

    Validation precedes :func:`_open_engine`. ``no hits`` is printed when there are none;
    ``Engine`` is disposed in ``finally`` whatever happens.
    """
    if not query.strip():
        typer.echo("error: query must be non-blank", err=True)
        raise typer.Exit(2)
    if limit < 1:
        typer.echo("error: --limit must be at least 1", err=True)
        raise typer.Exit(2)
    settings = load_settings()
    engine = _open_engine(settings)
    try:
        hits = search_chunks(engine, query, limit=limit, entity_id=entity_id)
    finally:
        engine.dispose()
    if not hits:
        typer.echo("no hits")
        return
    for hit in hits:
        line = f"{hit.score:.4f} chunk {hit.chunk_id} snapshot {hit.snapshot_id} seq {hit.seq}"
        if hit.heading_path:
            line = f"{line} {' > '.join(hit.heading_path)}"
        typer.echo(line)


def run_assessments(
    scope: Path, entity_id: int | None, criterion_id: str | None, rubric_version: str | None
) -> None:
    """Print the current view (AS-11): one line per latest assessment plus its evidence lines.

    Read-only via :func:`core.assessment_store.current_assessments`; ``no assessments`` when
    empty (exit 0); setup errors exit 2. ``scope`` is validated and loads settings; assessments
    carry no scope column, so the listing is not narrowed by it.
    """
    settings = _load(scope)
    engine = _open_engine(settings)
    try:
        rows = current_assessments(
            engine, entity_id=entity_id, criterion_id=criterion_id, rubric_version=rubric_version
        )
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    if not rows:
        typer.echo("no assessments")
        return
    for row in rows:
        typer.echo(
            f"assessment {row.id} entity {row.entity_id} criterion {row.criterion_id}"
            f" rubric {row.rubric_version} status {row.status} route {row.route_kind}"
            f" decided_by {row.decided_by} run {row.run_id}"
        )
        for ev in row.evidence:
            typer.echo(f"  evidence chunk {ev.chunk_id} grounded {'yes' if ev.grounded else 'no'}")


def _parse_attribute_text(attribute: Attribute, text: str) -> object:
    """Parse CLI ``text`` per ``attribute.type``; raise ``ValueError`` when it does not parse.

    int uses ``int()``; float uses ``float()``; bool takes ``true``/``false`` only; enum and date
    pass the string through for :func:`core.attribute_store.record_attribute_value` to validate.
    """
    kind = attribute.type
    if kind == "int":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"attribute {attribute.id!r} expects an int, got {text!r}") from None
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"attribute {attribute.id!r} expects a float, got {text!r}") from None
    if kind == "bool":
        if text in ("true", "false"):
            return text == "true"
        raise ValueError(f"attribute {attribute.id!r} expects true or false, got {text!r}")
    return text


def _load_pack_or_exit(scope: Path) -> ScopePack:
    try:
        return load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


def run_sources(scope: Path) -> None:
    """Seed the ``source`` catalogue from the pack's sources (AD-02, AD-15) and print counts.

    One transaction via ``seed_sources``; a pack error or a database error exits 2 with one
    ``error:`` line (the database URL is never echoed).
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    engine = _open_engine(settings)
    try:
        summary = seed_sources(engine, pack)
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    finally:
        engine.dispose()
    typer.echo(
        f"inserted {summary.inserted} updated {summary.updated} unchanged {summary.unchanged}"
    )


def run_attribute_set(scope: Path, entity_id: int, attribute_id: str, text: str) -> None:
    """Record one manual attribute value (AT-02) via ``record_attribute_value`` and commit.

    An attribute id not in the pack's attributes.yaml, an unparsable value or one that fails
    validation exits 2 with one line and writes nothing.
    Manual values carry no run (``run_id`` None).
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    attribute = next((a for a in pack.attributes.attributes if a.id == attribute_id), None)
    if attribute is None:
        typer.echo(f"error: unknown attribute {attribute_id!r}", err=True)
        raise typer.Exit(2)
    try:
        value = _parse_attribute_text(attribute, text)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    engine = _open_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            row_id = record_attribute_value(session, entity_id, attribute, value)
    except ValueError as exc:  # validation runs before the session is touched; nothing written
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    typer.echo(f"attribute_value {row_id} entity {entity_id} attribute {attribute_id}")


def _entity_in_scope(session: Session, entity_id: int, scope_id: str) -> bool:
    entity = session.get(Entity, entity_id)
    return entity is not None and entity.scope_id == scope_id


def run_presence_set(
    scope: Path, entity_id: int, country: str, presence_type: str, evidence_id: int | None = None
) -> None:
    """Record one manual market presence (MP-01) and commit, optionally citing a chunk (MP-02).

    An invalid type or country, an entity not in the scope, or an ``evidence_id`` chunk that is
    missing or whose snapshot belongs to another (or no) entity exits 2 with one line and writes
    nothing.
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    try:
        check_presence(country, presence_type)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    engine = _open_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            if not _entity_in_scope(session, entity_id, pack.scope.id):
                typer.echo(f"error: unknown entity {entity_id}", err=True)
                raise typer.Exit(2)
            if evidence_id is not None:
                owner = session.scalar(
                    select(SnapshotRow.entity_id)
                    .join(ChunkRow, ChunkRow.snapshot_id == SnapshotRow.id)
                    .where(ChunkRow.id == evidence_id)
                )
                if owner != entity_id:
                    typer.echo(
                        f"error: evidence chunk {evidence_id} is not a chunk of entity {entity_id}",
                        err=True,
                    )
                    raise typer.Exit(2)
            row_id = record_presence(session, entity_id, country, presence_type, evidence_id)
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    typer.echo(f"market_presence {row_id} entity {entity_id} {country} {presence_type}")


def run_presences(scope: Path, entity_id: int) -> None:
    """Print ``country type`` lines (sorted) then ``layers: a, b`` via ``layer_ids_for`` (MP-01).

    An entity not in the scope exits 2 with one line.
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    engine = _open_engine(settings)
    try:
        with Session(engine) as session:
            if not _entity_in_scope(session, entity_id, pack.scope.id):
                typer.echo(f"error: unknown entity {entity_id}", err=True)
                raise typer.Exit(2)
            rows = presence_evidence_many(session, [entity_id]).get(entity_id, [])
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    for country, presence_type, evidence_id in rows:
        suffix = f" evidence {evidence_id}" if evidence_id is not None else ""
        typer.echo(f"{country} {presence_type}{suffix}")
    layers = layer_ids_for(pack.scope.layers, {country for country, _, _ in rows})
    typer.echo(f"layers: {', '.join(layers)}")


def run_presence_extract(
    scope: Path, presence_type: str, entity_ids: list[int] | None, dry_run: bool
) -> None:
    """Record presence found by the pack's ``presence_terms`` in stored chunks (MP-03).

    Prints ``entity E country CC chunk N`` per match then ``extracted K``; ``--dry-run`` prints
    the same and writes nothing. An unknown type, or a pack without ``presence_terms``, exits 2
    with one line. Without ``--entity`` every entity of the scope is scanned. Without
    ``--dry-run`` each entity is one StageRunner unit (RT-01, see
    :func:`_run_presence_extract_units`); unchanged done units are skipped and a final
    ``skipped S`` line prints when S > 0.
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    if presence_type not in PRESENCE_TYPES:
        typer.echo(
            f"error: presence type must be one of {', '.join(PRESENCE_TYPES)}, "
            f"got {presence_type!r}",
            err=True,
        )
        raise typer.Exit(2)
    if not pack.scope.presence_terms:
        typer.echo("error: scope pack declares no presence_terms", err=True)
        raise typer.Exit(2)
    compiled = compile_terms(pack.scope.presence_terms)
    engine = _open_engine(settings)
    if not dry_run:
        try:
            ids: Iterable[int] | None = None
            if entity_ids:
                ids = sorted(set(entity_ids))
                with Session(engine) as session:
                    for entity_id in ids:
                        if not _entity_in_scope(session, entity_id, pack.scope.id):
                            typer.echo(f"error: unknown entity {entity_id}", err=True)
                            raise typer.Exit(2)
            extracted, skipped = _run_presence_extract_units(
                engine, pack, compiled, presence_type, ids
            )
        except SQLAlchemyError as exc:
            _exit_database_unavailable(engine, None, exc)
        finally:
            engine.dispose()
        typer.echo(f"extracted {extracted}")
        if skipped:
            typer.echo(f"skipped {skipped}")
        return
    try:
        with Session(engine) as session, session.begin():
            if entity_ids:
                ids = sorted(set(entity_ids))
                for entity_id in ids:
                    if not _entity_in_scope(session, entity_id, pack.scope.id):
                        typer.echo(f"error: unknown entity {entity_id}", err=True)
                        raise typer.Exit(2)
            else:
                ids = list(
                    session.scalars(
                        select(Entity.id)
                        .where(Entity.scope_id == pack.scope.id)
                        .order_by(Entity.id)
                    )
                )
            matches = extract_presence(session, compiled, ids)
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    for m in matches:
        typer.echo(f"entity {m.entity_id} country {m.country} chunk {m.chunk_id}")
    typer.echo(f"extracted {len(matches)}")


def _run_presence_extract_units(
    engine: Engine,
    pack: ScopePack,
    compiled: CompiledTerms,
    presence_type: str,
    entity_ids: Iterable[int] | None,
) -> tuple[int, int]:
    """RT-01: one :class:`core.runs.StageRunner` unit per entity under one run row.

    Stage ``presence-extract``, unit key ``"<entity_id>:<presence_type>"``, input hash = sha256
    over the scope id, the type, the canonical JSON of ``presence_terms`` and the entity's chunk
    count + max chunk id (one aggregate query per unit). Entity ids stream via
    :func:`iter_entities` when ``entity_ids`` is ``None``. A unit already ``done`` for the same
    hash is skipped; each executed unit extracts and records (idempotent) in its own transaction
    and prints its match lines. Returns ``(extracted, skipped)``.
    """
    terms = json.dumps(pack.scope.presence_terms, sort_keys=True)
    stream: Iterable[int] = (
        entity_ids
        if entity_ids is not None
        else (entity.id for entity in iter_entities(engine, pack.scope.id))
    )
    extracted = 0

    def units() -> Iterator[tuple[str, str]]:
        for entity_id in stream:
            with engine.connect() as conn:
                n, top = conn.execute(
                    select(func.count(ChunkRow.id), func.max(ChunkRow.id))
                    .join(SnapshotRow, SnapshotRow.id == ChunkRow.snapshot_id)
                    .where(SnapshotRow.entity_id == entity_id)
                ).one()
            digest = hashlib.sha256(
                f"{pack.scope.id}\0{presence_type}\0{terms}\0{n}\0{top}".encode()
            ).hexdigest()
            yield f"{entity_id}:{presence_type}", digest

    def handle(key: str) -> None:
        nonlocal extracted
        entity_id = int(key.split(":", 1)[0])
        with Session(engine) as session, session.begin():
            matches = extract_presence(session, compiled, [entity_id])
            for m in matches:
                record_presence(session, m.entity_id, m.country, presence_type, m.chunk_id)
        for m in matches:
            typer.echo(f"entity {m.entity_id} country {m.country} chunk {m.chunk_id}")
        extracted += len(matches)

    summary = StageRunner(engine).run("presence-extract", units(), handle, catch=())
    return extracted, summary.skipped


def run_attributes(scope: Path, entity_id: int) -> None:
    """Print ``<attr_id> <value json> <unit or ->`` per current value (AT-02), sorted by id.

    Read-only via ``current_attribute_values``; ``no attributes`` when empty (exit 0). The unit
    is the pack attribute's; an id no longer in the pack prints ``-``.
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    units = {a.id: a.unit for a in pack.attributes.attributes}
    engine = _open_engine(settings)
    try:
        with Session(engine) as session:
            values = current_attribute_values(session, entity_id)
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    if not values:
        typer.echo("no attributes")
        return
    for attribute_id in sorted(values):
        typer.echo(
            f"{attribute_id} {json.dumps(values[attribute_id])} {units.get(attribute_id) or '-'}"
        )


ATTRIBUTES_EXPORT_BATCH = 500


def run_attributes_export(scope: Path, out: Path) -> None:
    """Write one attributes object per entity of ``scope`` with current values to ``out`` (AT-03).

    Entities stream in ascending id via :func:`iter_entities` and are batched
    (``ATTRIBUTES_EXPORT_BATCH`` ids) into one
    :func:`core.attribute_store.current_attribute_value_rows_many` query and one
    :func:`core.attribute_store.attribute_evidence_many` query per batch (AT-04, AT-10, AT-11).
    Entities without values are skipped and counted; the rest are written one line at a time,
    attribute ids sorted, with an ``evidence`` object (attribute id -> current value's evidence
    in id order) and a ``units`` object (attribute id -> current row's stored unit, JSON null
    when it has none) to ``<out>.tmp``, renamed to ``out`` on success so a failure leaves no
    ``PATH``. Setup errors or an unwritable output directory exit 2. Production caller: the
    ``attributes-export`` command; consumer is the later static-site row.
    """
    settings = _load(scope)
    pack = _load_pack_or_exit(scope)
    tmp = Path(f"{out}.tmp")
    count = skipped = 0
    try:
        try:
            handle = tmp.open("w", encoding="utf-8")
        except OSError as exc:
            typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
            raise typer.Exit(2) from None
        with handle:
            engine = _open_engine(settings)
            try:
                batch: list[int] = []

                def flush() -> None:
                    nonlocal count, skipped
                    with Session(engine) as session:
                        many = current_attribute_value_rows_many(session, batch)
                        evidence = attribute_evidence_many(
                            session,
                            [row_id for rows in many.values() for row_id, _, _ in rows.values()],
                        )
                    for entity_id in batch:
                        values = many.get(entity_id)
                        if not values:
                            skipped += 1
                            continue
                        keys = sorted(values)
                        obj = {
                            "entity_id": entity_id,
                            "attributes": {key: values[key][1] for key in keys},
                            "evidence": {
                                key: [
                                    {
                                        "chunk_id": e.chunk_id,
                                        "quote": e.quote,
                                        "grounded": e.grounded,
                                    }
                                    for e in evidence.get(values[key][0], [])
                                ]
                                for key in keys
                            },
                            "units": {key: values[key][2] for key in keys},
                        }
                        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        count += 1
                    batch.clear()

                for entity in iter_entities(engine, pack.scope.id):
                    batch.append(entity.id)
                    if len(batch) >= ATTRIBUTES_EXPORT_BATCH:
                        flush()
                if batch:
                    flush()
            except SQLAlchemyError as exc:
                _exit_database_unavailable(engine, None, exc)
            finally:
                engine.dispose()
        os.replace(tmp, out)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
        raise typer.Exit(2) from None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    typer.echo(f"exported {count} entities to {out} (skipped {skipped} without attributes)")


def _fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _scores_object(
    entity_id: int, segments: tuple[str, ...], version: str, scores: Iterable[TopicScore]
) -> dict[str, object]:
    """The AG-03 per-entity JSON object; shared by ``scores --format json`` and scores-export."""
    return {
        "entity_id": entity_id,
        "segments": list(segments),
        "rubric_version": version,
        "topics": [
            {
                "topic_id": score.topic_id,
                "score": score.score,
                "coverage": score.coverage,
                "applicable": score.applicable,
                "scored": score.scored,
                "evidenced": score.evidenced,
            }
            for score in scores
        ],
    }


def run_scores(
    scope: Path, entity_id: int, segments: list[str] | None, format: str = "text"
) -> None:
    """Print per-topic score and coverage for one entity (AG-02) via ``score_topics``.

    ``format`` ``text`` (default) prints the AG-02 header plus one line per topic; ``json``
    (AG-03) prints one JSON object ``{"entity_id", "segments", "rubric_version", "topics"}``
    with the topics in rubric order and ``score``/``coverage`` as unrounded floats or ``null``.
    Any other value exits 2 before setup.

    Entity segments are ``segments`` when given (each must be a scope segment id), else the
    entity's latest screen result. Assessments are the current ones at the pack's version (the
    version ``landscape assess`` stores), consumed once. Setup errors exit 2; a ``ValueError``
    from ``score_topics`` exits 1.
    """
    if format not in {"text", "json"}:
        typer.echo(f"error: unknown format {format!r}", err=True)
        raise typer.Exit(2)
    settings = _load(scope)
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    if segments:
        known = {s.id for s in pack.scope.segments}
        for segment in segments:
            if segment not in known:
                typer.echo(f"error: unknown segment {segment!r}", err=True)
                raise typer.Exit(2)
    version = pack.scope.version
    engine = _open_engine(settings)
    try:
        entity_segments = tuple(segments) if segments else latest_screen_segments(engine, entity_id)
        if entity_segments is None:
            typer.echo(f"error: entity {entity_id} has no screen result; pass --segment", err=True)
            raise typer.Exit(2)
        assessments = iter_current_assessments(engine, entity_id=entity_id, rubric_version=version)
        try:
            scores = score_topics(pack.rubric, assessments, entity_segments)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    if format == "json":
        typer.echo(
            json.dumps(
                _scores_object(entity_id, entity_segments, version, scores), ensure_ascii=False
            )
        )
        return
    typer.echo(f"entity {entity_id} segments {','.join(entity_segments)} rubric {version}")
    for score in scores:
        typer.echo(
            f"topic {score.topic_id} score {_fmt_ratio(score.score)}"
            f" coverage {_fmt_ratio(score.coverage)} applicable {score.applicable}"
            f" scored {score.scored} evidenced {score.evidenced}"
        )


SCORES_EXPORT_BATCH = 500


SCORES_EXPORT_FORMATS = ("jsonl", "csv")
_CSV_TOPIC_FIELDS = ("score", "coverage", "applicable", "scored", "evidenced")


_CSV_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(value: object) -> str:
    """One XP-01 CSV cell: ``None`` is empty, lists join with ``;``, else ``str`` (unrounded).

    XP-02: a text cell (not int/float/bool) starting with ``= + - @`` TAB or CR gets a leading
    ``'`` so a spreadsheet does not evaluate it as a formula; numbers and booleans are unchanged.
    """
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return str(value)
    text = ";".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return f"'{text}" if text.startswith(_CSV_FORMULA_LEADS) else text


def _scores_csv_row(obj: Mapping[str, Any], topic_ids: Sequence[str]) -> list[str]:
    """Flatten one scores-export object into the XP-01 CSV row for ``topic_ids`` (rubric order)."""
    row = [_csv_cell(obj[key]) for key in ("entity_id", "segments", "layers", "rubric_version")]
    by_topic = {topic["topic_id"]: topic for topic in obj["topics"]}
    for topic_id in topic_ids:
        topic = by_topic.get(topic_id, {})
        row.extend(_csv_cell(topic.get(field)) for field in _CSV_TOPIC_FIELDS)
    return row


def run_scores_export(scope: Path, out: Path, format: str = "jsonl") -> None:
    """Write one AG-03 scores object per screened entity of ``scope`` to ``out`` (AG-04).

    ``format`` ``jsonl`` (default) writes one JSON object per line; ``csv`` (XP-01) writes one
    RFC 4180 row per entity under a header (always written) of ``entity_id, segments, layers,
    rubric_version`` then ``<topic_id>.<field>`` per rubric topic in rubric order, lists joined
    with ``;`` and ``None`` as an empty cell. Any other value exits 2 before setup.

    Entities stream in ascending id via :func:`iter_entities` and are batched
    (``SCORES_EXPORT_BATCH`` ids) into one
    :func:`core.screen_store.latest_screen_segments_many` query and one
    :func:`core.assessment_store.iter_current_assessments` query per batch (AG-05), plus one
    :func:`core.presence_store.presences_many` query per batch whose countries give each line's
    ``layers`` (FE-04; scope layer order, ``[]`` without presence). Entities
    without a screen result are skipped and counted; the rest are scored and written one line at
    a time to ``<out>.tmp``, renamed to ``out`` on success so a failure leaves no ``PATH``. Setup
    errors or an unwritable output directory exit 2; a ``ValueError`` from ``score_topics`` exits
    1. Production caller: the ``scores-export`` command; consumer is the later frontend row
    (JSON Lines) or an operator spreadsheet (CSV, XP-01).
    """
    if format not in SCORES_EXPORT_FORMATS:
        typer.echo(f"error: unknown format {format!r}", err=True)
        raise typer.Exit(2)
    settings = _load(scope)
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    version = pack.scope.version
    tmp = Path(f"{out}.tmp")
    count = skipped = 0
    try:
        try:
            handle = tmp.open("w", encoding="utf-8", newline="")
        except OSError as exc:
            typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
            raise typer.Exit(2) from None
        with handle:
            topic_ids = [topic.id for topic in pack.rubric.topics]
            writer = csv.writer(handle, lineterminator="\n") if format == "csv" else None
            if writer is not None:
                writer.writerow(
                    ["entity_id", "segments", "layers", "rubric_version"]
                    + [f"{tid}.{field}" for tid in topic_ids for field in _CSV_TOPIC_FIELDS]
                )
            engine = _open_engine(settings)
            try:
                batch: list[int] = []

                def flush() -> None:
                    nonlocal count, skipped
                    segments_by_entity = latest_screen_segments_many(engine, batch)
                    presences = presences_many(engine, batch)
                    grouped: dict[int, list[CurrentAssessment]] = {}
                    for assessment in iter_current_assessments(
                        engine, entity_ids=batch, rubric_version=version
                    ):
                        grouped.setdefault(assessment.entity_id, []).append(assessment)
                    for entity_id in batch:
                        segments = segments_by_entity.get(entity_id)
                        if segments is None:
                            skipped += 1
                            continue
                        try:
                            scores = score_topics(pack.rubric, grouped.get(entity_id, ()), segments)
                        except ValueError as exc:
                            typer.echo(f"error: entity {entity_id}: {exc}", err=True)
                            raise typer.Exit(1) from None
                        obj = _scores_object(entity_id, segments, version, scores)
                        countries = (c for c, _ in presences.get(entity_id, ()))
                        obj["layers"] = layer_ids_for(pack.scope.layers, countries)
                        if writer is None:
                            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        else:
                            writer.writerow(_scores_csv_row(obj, topic_ids))
                        count += 1
                    batch.clear()

                for entity in iter_entities(engine, pack.scope.id):
                    batch.append(entity.id)
                    if len(batch) >= SCORES_EXPORT_BATCH:
                        flush()
                if batch:
                    flush()
            except SQLAlchemyError as exc:
                _exit_database_unavailable(engine, None, exc)
            finally:
                engine.dispose()
        os.replace(tmp, out)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
        raise typer.Exit(2) from None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    typer.echo(f"exported {count} entities to {out} (skipped {skipped} without screen result)")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-blank line of ``path`` (streaming)."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{number}: not a JSON object")
                yield obj


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``<path>.tmp`` then rename; exit 2 on OSError."""
    tmp = Path(f"{path}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        typer.echo(f"error: cannot write {path}: {exc.strerror or exc}", err=True)
        raise typer.Exit(2) from None


def run_mosaic(
    scope: Path,
    scores: Path,
    export: Path,
    out: Path,
    entity_dir: Path | None = None,
    segments: list[str] | None = None,
    layers: list[str] | None = None,
) -> None:
    """Write the FE-01 mosaic page for ``scores`` and ``export`` JSON Lines to ``out``.

    Offline: no settings or database. Each page goes to ``<file>.tmp`` and is renamed on
    success. With ``entity_dir`` (FE-02) one page per scored entity is also written to
    ``entity_dir/<slug>.html`` and mosaic row headers link to it. A missing scope/input file,
    bad scope pack or unwritable output exits 2; a malformed JSON line exits 1. With ``segments``
    (FE-03) only scores lines whose ``segments`` include one of the ids are rendered (and get
    entity pages) and the page heading names the filter by label; an unknown id exits 2.
    ``layers`` (FE-04) filters the same way on each line's ``layers`` list (a line without one is
    dropped); with both, a line must match both. Layers have no label, so the id is shown.
    """
    if not scope.is_dir():
        typer.echo(f"error: scope directory not found: {scope}", err=True)
        raise typer.Exit(2)
    pack = _load_pack_or_exit(scope)
    heading: str | None = None
    parts: list[str] = []
    if segments:
        labels = {s.id: s.label for s in pack.scope.segments}
        for segment in segments:
            if segment not in labels:
                typer.echo(f"error: unknown segment id: {segment}", err=True)
                raise typer.Exit(2)
        parts.append("segment: " + ", ".join(labels[s] for s in dict.fromkeys(segments)))
    if layers:
        layer_ids = {la.id for la in pack.scope.layers}
        for layer in layers:
            if layer not in layer_ids:
                typer.echo(f"error: unknown layer id: {layer}", err=True)
                raise typer.Exit(2)
        parts.append("layer: " + ", ".join(dict.fromkeys(layers)))
    if parts:
        heading = "; ".join(parts)
    for path in (scores, export):
        if not path.is_file():
            typer.echo(f"error: input file not found: {path}", err=True)
            raise typer.Exit(2)

    def score_lines() -> Iterator[Mapping[str, Any]]:
        lines: Iterator[Mapping[str, Any]] = _iter_jsonl(scores)
        if segments:
            lines = filter_by_segments(lines, segments)
        if layers:
            lines = filter_by_layers(lines, layers)
        return lines

    try:
        if entity_dir is None:
            page = render_mosaic(pack, score_lines(), _iter_jsonl(export), heading=heading)
        else:
            score_rows = list(score_lines())
            export_rows = list(_iter_jsonl(export))
    except ValueError as exc:  # includes json.JSONDecodeError
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    if entity_dir is not None:
        rel_dir = Path(os.path.relpath(entity_dir.resolve(), out.resolve().parent)).as_posix()
        back = Path(os.path.relpath(out.resolve(), entity_dir.resolve())).as_posix()
        try:
            entity_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            typer.echo(f"error: cannot create {entity_dir}: {exc.strerror or exc}", err=True)
            raise typer.Exit(2) from None
        by_entity: dict[str, dict[str, dict[str, object]]] = {}
        for row in export_rows:
            by_entity.setdefault(str(row.get("entity_id")), {})[str(row.get("criterion_id"))] = row
        for line in score_rows:
            entity = str(line.get("entity_id"))
            text = render_entity_page(pack, line, by_entity.get(entity, {}), back)
            _write_atomic(entity_dir / f"{entity_slug(entity)}.html", text)
        page = render_mosaic(
            pack,
            score_rows,
            export_rows,
            lambda e: f"{rel_dir}/{entity_slug(e)}.html",
            heading=heading,
        )
    _write_atomic(out, page)
    typer.echo(f"wrote mosaic to {out}")


def _export_object(row: CurrentAssessment) -> dict[str, object]:
    """One JSON Lines object for a current assessment, with its evidence nested per row."""
    return {
        "id": row.id,
        "entity_id": row.entity_id,
        "criterion_id": row.criterion_id,
        "status": row.status,
        "confidence": row.confidence,
        "rationale": row.rationale,
        "rubric_version": row.rubric_version,
        "extractor_model": row.extractor_model,
        "verifier_model": row.verifier_model,
        "decided_by": row.decided_by,
        "route_kind": row.route_kind,
        "run_id": row.run_id,
        "created_at": row.created_at.isoformat(),
        "evidence": [
            {
                "chunk_id": ev.chunk_id,
                "quote": ev.quote,
                "grounded": ev.grounded,
                "snapshot_id": ev.snapshot_id,
                "url": ev.url,
                "fetched_at": ev.fetched_at.isoformat() if ev.fetched_at else None,
            }
            for ev in row.evidence
        ],
    }


def run_export(
    scope: Path,
    format: str,
    out: Path,
    entity_id: int | None,
    criterion_id: str | None,
    rubric_version: str | None,
) -> None:
    """Write the current view (AS-11) as JSON Lines to ``out``; read-only, exit 2 on bad setup.

    Rows go one at a time to ``<out>.tmp``, renamed to ``out`` on success, so a failed export
    leaves no ``PATH``. Bad input or an unwritable output directory exits 2; an empty view writes
    an empty file and prints ``exported 0 assessments to PATH``.
    """
    if format != "json":
        typer.echo(f"error: unsupported format {format} (known: json)", err=True)
        raise typer.Exit(2)
    settings = _load(scope)
    tmp = Path(f"{out}.tmp")
    count = 0
    try:
        try:
            handle = tmp.open("w", encoding="utf-8")
        except OSError as exc:
            typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
            raise typer.Exit(2) from None
        with handle:
            engine = _open_engine(settings)
            try:
                for row in iter_current_assessments(
                    engine,
                    entity_id=entity_id,
                    criterion_id=criterion_id,
                    rubric_version=rubric_version,
                ):
                    handle.write(json.dumps(_export_object(row), ensure_ascii=False) + "\n")
                    count += 1
            except SQLAlchemyError as exc:
                _exit_database_unavailable(engine, None, exc)
            finally:
                engine.dispose()
        os.replace(tmp, out)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        typer.echo(f"error: cannot write {out}: {exc.strerror or exc}", err=True)
        raise typer.Exit(2) from None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    typer.echo(f"exported {count} assessments to {out}")


def _make_assess_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the local extractor model for assess (§6 routing); the test seam for a fake adapter."""
    return build_llm(
        settings,
        "local",
        db=db,
        budget_usd=settings.llm_budget_usd,
        timeout=settings.llm_timeout_s,
        max_tokens=settings.local_max_tokens,
    )


def _make_verify_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the local verifier from ``settings.verifier_model`` (§6 routing); the test seam."""
    verifier_settings = dataclasses.replace(settings, local_model=settings.verifier_model)
    return build_llm(
        verifier_settings,
        "local",
        db=db,
        budget_usd=settings.llm_budget_usd,
        timeout=settings.llm_timeout_s,
        max_tokens=(
            settings.verifier_max_tokens
            if settings.verifier_max_tokens is not None
            else settings.local_max_tokens
        ),
    )


def _find_criterion(rubric: Rubric, criterion_id: str) -> Criterion | None:
    for topic in rubric.topics:
        for criterion in topic.criteria:
            if criterion.id == criterion_id:
                return criterion
    return None


def run_assess(
    scope: Path,
    entity_id: int | None,
    criterion_id: str | None,
    n: int,
    k: int,
    dry_run: bool,
    embedding_model: str | None = None,
    rerank_model: str | None = None,
) -> None:
    """Retrieve evidence and run the extractor for one entity × criterion inside an ``assess`` run.

    Prints ``status S checks pass|fail chunks N``, one ``quote ...`` line per quote, one
    ``failure ...`` line per failed check and ``rationale ...``. Exits 0 when checks pass, 1 when
    they fail or on ``AssessError``/``LlmError`` (run ``failed``), 2 on bad input or setup.
    Then runs the verifier (``_make_verify_llm``, ``verifier_model``; no call when silent) and
    :func:`core.verify.route`, printing ``route KIND confidence C|- reason R``; ``adjudicate``
    exits 1. Every reached outcome (accept, adjudicate or silent) appends one ``assessment`` row
    plus one ``evidence`` row per quote through :func:`core.assessment_store.record_assessment`
    before the run closes ``finished``, and prints ``assessment ID`` after the route line.
    ``AssessError``/``LlmError``/``EmbeddingError`` and setup errors write no assessment row.
    With ``embedding_model`` set (AS-07), an Embedder is built via :func:`_make_embedder` before
    the run row opens (``ValueError`` exits 2, no run row) and retrieval is hybrid with that
    stored model; ``EmbeddingError`` closes the run ``failed`` and exits 1.
    With ``rerank_model`` set (AS-09), a Reranker is built via :func:`_make_reranker` the same way
    and reranks the top-``k`` pool to ``n``; ``EmbeddingError`` or an out-of-range hit
    ``ValueError`` closes the run ``failed`` and exits 1.
    """
    settings = _load(scope)
    if dry_run:
        _print_plan("assess", scope, None, None, settings)
        return
    if entity_id is None or criterion_id is None:
        typer.echo("error: --entity and --criterion are required", err=True)
        raise typer.Exit(2)
    try:
        pack = load_scope_pack(scope)
    except ScopePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    criterion = _find_criterion(pack.rubric, criterion_id)
    if criterion is None:
        typer.echo(f"error: unknown criterion {criterion_id}", err=True)
        raise typer.Exit(2)
    if n < 1:
        typer.echo("error: --n must be at least 1", err=True)
        raise typer.Exit(2)
    if k < n:
        typer.echo("error: --k must be at least --n", err=True)
        raise typer.Exit(2)
    if embedding_model is not None and not embedding_model.strip():
        typer.echo("error: --embedding-model must be non-empty", err=True)
        raise typer.Exit(2)
    if rerank_model is not None and not rerank_model.strip():
        typer.echo("error: --rerank-model must be non-empty", err=True)
        raise typer.Exit(2)

    engine = _open_engine(settings)
    run_id: int | None = None
    try:
        try:
            llm = _make_assess_llm(settings, engine)
            verify_llm = _make_verify_llm(settings, engine)
            embedder = (
                _make_embedder(settings, embedding_model) if embedding_model is not None else None
            )
            reranker = _make_reranker(settings, rerank_model) if rerank_model is not None else None
        except (ValueError, LlmError) as exc:  # unbuildable backend: bad setup, exit 2
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
        run_id = _open_run(engine, "assess")
        try:
            if reranker is None and embedder is None:
                chunks = retrieve_evidence(engine, criterion, entity_id=entity_id, n=n, k=k)
            elif reranker is None:
                chunks = retrieve_evidence(
                    engine,
                    criterion,
                    entity_id=entity_id,
                    n=n,
                    k=k,
                    embedder=embedder,
                    model=embedding_model,
                )
            else:
                try:
                    chunks = retrieve_evidence(
                        engine,
                        criterion,
                        entity_id=entity_id,
                        n=n,
                        k=k,
                        embedder=embedder,
                        model=embedding_model,
                        reranker=reranker,
                    )
                except ValueError as exc:  # out-of-range rerank hit (AS-08)
                    typer.echo(f"error {type(exc).__name__}: {exc}", err=True)
                    _close_run(engine, run_id, FAILED)
                    raise typer.Exit(1) from None
            extraction = extract_criterion(llm, pack.rubric, criterion, chunks, run_id=run_id)
            verification = verify_extraction(
                verify_llm, pack.rubric, criterion, chunks, extraction, run_id=run_id
            )
            routed = route(extraction, verification)
            assessment_id = record_assessment(
                engine,
                entity_id=entity_id,
                criterion_id=criterion.id,
                extraction=extraction,
                route=routed,
                rubric_version=pack.scope.version,
                extractor_model=llm.model,
                verifier_model=verify_llm.model,
                run_id=run_id,
            )
        except (AssessError, LlmError, EmbeddingError) as exc:
            typer.echo(f"error {type(exc).__name__}: {exc}", err=True)
            _close_run(engine, run_id, FAILED)
            raise typer.Exit(1) from None
        except SQLAlchemyError:
            raise  # the outer handler closes the run best-effort and prints one line (AB-01)
        except BaseException as exc:
            _close_run_aborting(engine, run_id, exc)
            raise
        _close_run(engine, run_id, FINISHED)
    except SQLAlchemyError as exc:  # URL never echoed (AB-01)
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()

    checks = "pass" if extraction.checks_passed else "fail"
    typer.echo(f"status {extraction.status} checks {checks} chunks {len(chunks)}")
    for quote in extraction.quotes:
        grounded = "grounded" if quote.grounded else "ungrounded"
        typer.echo(f'quote chunk {quote.chunk_id} {grounded} {quote.method or "-"} "{quote.text}"')
    for failure in extraction.failures:
        typer.echo(f"failure {failure}")
    typer.echo(f"rationale {extraction.rationale}")
    typer.echo(f"route {routed.kind} confidence {routed.confidence or '-'} reason {routed.reason}")
    typer.echo(f"assessment {assessment_id}")
    if not extraction.checks_passed or routed.kind == "adjudicate":
        raise typer.Exit(1)


@dataclasses.dataclass(frozen=True)
class _PairExtraction:
    """One entity × attribute result shared by single and ``--all`` batch modes (AT-07/AT-08)."""

    entity_id: int
    attribute_id: str
    extraction: AttributeExtraction
    num_chunks: int
    value_id: int | None


def _extract_pair(
    engine: Engine, llm: LLM, run_id: int, entity_id: int, attribute: Attribute, n: int
) -> _PairExtraction:
    """Per-pair work shared by single and batch modes (one code path).

    BM25 retrieves up to ``n`` of the entity's chunks for :func:`attribute_query`; the extractor
    runs via :func:`extract_attribute`; the value plus its evidence is stored by
    :func:`store_attribute_extraction` in its own committed transaction (null or failed
    extractions write nothing). ``AttributeExtractError``/``LlmError`` propagate; a pair's
    transaction never rolls back an earlier pair. ``SQLAlchemyError`` propagates.
    """
    hits = search_chunks(engine, attribute_query(attribute), limit=n, entity_id=entity_id)
    chunks = _with_texts(engine, [hit.chunk_id for hit in hits])
    extraction = extract_attribute(llm, attribute, chunks, run_id=run_id)
    with Session(engine) as session, session.begin():
        value_id = store_attribute_extraction(
            session, entity_id, attribute, extraction, run_id=run_id
        )
    return _PairExtraction(entity_id, attribute.id, extraction, len(chunks), value_id)


ATTRIBUTE_EXTRACT_PAGE = 500  # keyset page size; one current-values query per page (AT-09)


def _run_attributes_extract_batch(
    engine: Engine,
    llm: LLM,
    run_id: int,
    pack: ScopePack,
    attributes: Sequence[Attribute],
    n: int,
    limit: int | None,
    *,
    skip_existing: bool = False,
) -> tuple[int, int, int, int, int, int]:
    """AT-08 batch: every ``in_scope`` entity × each attribute, sequentially.

    Entities stream in ascending id via :func:`iter_entities` keyset paging (one page in memory,
    at most ``limit`` entities); per-pair ``AttributeExtractError``/``LlmError`` is recorded and
    the batch continues. Prints one line per pair. Returns ``(pairs, stored, null, failed,
    errors, skipped)``. One committed transaction per pair; earlier pairs are never rolled back.

    AT-09 ``skip_existing``: per keyset page (<= ``ATTRIBUTE_EXTRACT_PAGE`` ids) current values
    are loaded once with :func:`core.attribute_store.current_attribute_values_many` (one extra
    query per page); an entity × attribute pair whose attribute id already has a current value
    (a stored row; null extractions are never stored, so a current value is never a stored
    null) is skipped: no retrieval or LLM call, prints ``entity E attribute A skipped existing``,
    counted in ``skipped`` and ``pairs`` (``pairs == stored + null + failed + errors +
    skipped``), and never affects the exit code.

    RT-01: every non-skipped pair runs through :class:`core.runs.StageRunner` under ``run_id``
    (unit key ``"<entity_id>:<attribute_id>"``; input hash over the scope id, the attribute's
    canonical JSON, ``n`` and a sha256 over the entity's chunk ids and texts, streamed per page).
    A pair already ``done`` for the same hash is skipped silently and counted in ``skipped``; a
    pair whose checks fail is still ``done``; an errored pair is a ``failed`` unit, retried later.
    """
    pairs = stored = null = failed = errors = skipped = 0
    stream = iter_entities(
        engine,
        pack.scope.id,
        status=STATUS_IN_SCOPE,
        limit=limit,
        batch=ATTRIBUTE_EXTRACT_PAGE,
    )

    definitions = {
        attribute.id: json.dumps(attribute.model_dump(mode="json"), sort_keys=True)
        for attribute in attributes
    }
    in_flight: dict[str, tuple[int, Attribute]] = {}  # the one pair the runner is handling

    def units(entities: list[EntityRef]) -> Iterator[tuple[str, str]]:
        nonlocal pairs, skipped
        current: dict[int, dict[str, Any]] = {}
        if skip_existing:
            with Session(engine) as session:
                current = current_attribute_values_many(session, [entity.id for entity in entities])
        fingerprints = _chunk_fingerprints(engine, [entity.id for entity in entities])
        for entity in entities:
            for attribute in attributes:
                pairs += 1
                if skip_existing and attribute.id in current.get(entity.id, {}):
                    skipped += 1
                    typer.echo(f"entity {entity.id} attribute {attribute.id} skipped existing")
                    continue
                chunks = fingerprints.get(entity.id, "")
                key = f"{entity.id}:{attribute.id}"
                digest = hashlib.sha256(
                    f"{pack.scope.id}\0{definitions[attribute.id]}\0{n}\0{chunks}".encode()
                ).hexdigest()
                in_flight.clear()
                in_flight[key] = (entity.id, attribute)
                yield key, digest

    def handle(key: str) -> None:
        nonlocal stored, null, failed, errors
        entity_id, attribute = in_flight.pop(key)
        try:
            result = _extract_pair(engine, llm, run_id, entity_id, attribute, n)
        except (AttributeExtractError, LlmError) as exc:
            errors += 1
            typer.echo(
                f"entity {entity_id} attribute {attribute.id} error {type(exc).__name__}: {exc}"
            )
            raise
        checks = "pass" if result.extraction.checks_passed else "fail"
        attribute_value = result.value_id if result.value_id is not None else "-"
        typer.echo(
            f"entity {result.entity_id} attribute {result.attribute_id} "
            f"value {json.dumps(result.extraction.value)} checks {checks} "
            f"chunks {result.num_chunks} attribute_value {attribute_value}"
        )
        if result.value_id is not None:
            stored += 1
        elif not result.extraction.checks_passed:
            failed += 1
        else:
            null += 1

    def pages() -> Iterator[list[EntityRef]]:
        page: list[EntityRef] = []
        for entity in stream:
            page.append(entity)
            if len(page) >= ATTRIBUTE_EXTRACT_PAGE:
                yield page
                page = []
        if page:
            yield page

    # A pair already done for the same input is skipped by the runner (RT-01).
    summary = StageRunner(engine).run(
        "attributes_extract",
        chain.from_iterable(units(page) for page in pages()),
        handle,
        run_id=run_id,
        catch=(AttributeExtractError, LlmError),
    )
    skipped += summary.skipped
    return pairs, stored, null, failed, errors, skipped


def _chunk_fingerprints(engine: Engine, entity_ids: list[int]) -> dict[int, str]:
    """``entity_id -> sha256 over its chunks' (id, text)`` for one page (RT-01).

    Rows stream ordered by entity id, chunk id (one row in memory at a time), so a chunk whose
    text is edited in place without its id changing still changes the entity's fingerprint.
    """
    stmt = (
        select(SnapshotRow.entity_id, ChunkRow.id, ChunkRow.text)
        .join(ChunkRow, ChunkRow.snapshot_id == SnapshotRow.id)
        .where(SnapshotRow.entity_id.in_(entity_ids))
        .order_by(SnapshotRow.entity_id, ChunkRow.id)
        .execution_options(yield_per=500)
    )
    digests: dict[int, Any] = {}
    with engine.connect() as conn:
        for eid, chunk_id, text in conn.execute(stmt):
            if eid is None:
                continue
            h = digests.get(eid)
            if h is None:
                h = digests[eid] = hashlib.sha256()
            body = (text or "").encode()
            h.update(f"{chunk_id}\0{len(body)}\0".encode())
            h.update(body)
    return {eid: h.hexdigest() for eid, h in digests.items()}


def run_attributes_extract(
    scope: Path,
    entity_id: int | None,
    attribute_id: str | None,
    n: int,
    dry_run: bool,
    *,
    all_entities: bool = False,
    limit: int | None = None,
    skip_existing: bool = False,
) -> None:
    """Extract attribute values inside one ``attributes_extract`` run (AT-07, AT-08, AT-09).

    Single mode (AT-07, ``--entity E --attribute A``): unchanged output and behaviour.
    ``--all`` (AT-08, no ``--entity``): every ``in_scope`` entity of the scope, every pack
    attribute or only ``--attribute A``, ``--limit`` caps entities; one ``attributes_extract``
    run covers the whole batch. Per entity × attribute: BM25 retrieves up to ``n`` chunks for
    :func:`attribute_query`; the extractor (``_make_assess_llm``) runs via :func:`extract_attribute`
    and :func:`store_attribute_extraction` stores in its own committed transaction; per-pair
    ``AttributeExtractError``/``LlmError`` is recorded and the batch continues. Batch prints one
    ``entity E attribute A value V checks pass|fail chunks N attribute_value ID|-`` (or ``entity E
    attribute A error <Class>: msg``) line per pair then ``pairs P stored S null U failed F errors
    X skipped K``.

    Single mode prints ``value V checks pass|fail chunks N``, ``quote ...``, ``failure ...`` and
    ``attribute_value ID|-``. Exit 0 when checks pass (single) / no pair failed checks or errored
    (batch), 1 otherwise (run ``finished``); single mode also exits 1 on
    ``AttributeExtractError``/``LlmError`` (run ``failed``); 2 on bad input or setup (unknown
    attribute, ``--n < 1``, ``--limit < 1``, ``--all`` with ``--entity``, neither ``--all`` nor
    ``--entity``, ``--skip-existing`` without ``--all``; no run row is opened).

    AT-09 ``--skip-existing`` (``--all`` only; with ``--entity`` exits 2, no run row): per keyset
    page, current values are loaded once with
    :func:`core.attribute_store.current_attribute_values_many`; an entity × attribute pair whose
    attribute id already has a current value is skipped (no retrieval or LLM call), prints
    ``entity E attribute A skipped existing``, adds to ``skipped K``; ``pairs P`` counts every
    entity × attribute pair considered (``pairs == stored + null + failed + errors + skipped``),
    skipped pairs do not affect the exit code; without the flag ``K`` is 0.
    """
    settings = _load(scope)
    if skip_existing and not all_entities:
        typer.echo("error: --skip-existing requires --all", err=True)
        raise typer.Exit(2)
    if dry_run:
        _print_plan("attributes-extract", scope, None, None, settings)
        return
    if all_entities and entity_id is not None:
        typer.echo("error: --all cannot be combined with --entity", err=True)
        raise typer.Exit(2)
    if not all_entities and entity_id is None:
        typer.echo("error: --all or --entity is required", err=True)
        raise typer.Exit(2)
    if n < 1:
        typer.echo("error: --n must be at least 1", err=True)
        raise typer.Exit(2)
    if limit is not None and limit < 1:
        typer.echo("error: --limit must be at least 1", err=True)
        raise typer.Exit(2)
    pack = _load_pack_or_exit(scope)
    attribute: Attribute | None = None
    if attribute_id is not None:
        attribute = next((a for a in pack.attributes.attributes if a.id == attribute_id), None)
        if attribute is None:
            typer.echo(f"error: unknown attribute {attribute_id!r}", err=True)
            raise typer.Exit(2)
    elif not all_entities:
        typer.echo("error: --entity and --attribute are required", err=True)
        raise typer.Exit(2)

    engine = _open_engine(settings)
    run_id: int | None = None
    try:
        try:
            llm = _make_assess_llm(settings, engine)
        except (ValueError, LlmError) as exc:  # unbuildable backend: bad setup, exit 2
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
        run_id = _open_run(engine, "attributes_extract")

        if all_entities:
            attributes = [attribute] if attribute is not None else list(pack.attributes.attributes)
            try:
                pairs, stored, null, failed, errors, skipped = _run_attributes_extract_batch(
                    engine, llm, run_id, pack, attributes, n, limit, skip_existing=skip_existing
                )
            except SQLAlchemyError:
                raise  # the outer handler closes the run best-effort and prints one line
            except BaseException as exc:
                _close_run_aborting(engine, run_id, exc)
                raise
            _close_run(engine, run_id, FINISHED)
            typer.echo(
                f"pairs {pairs} stored {stored} null {null} failed {failed} "
                f"errors {errors} skipped {skipped}"
            )
            if failed or errors:
                raise typer.Exit(1)
            return

        # single-entity mode (AT-07): unchanged output
        assert entity_id is not None and attribute is not None
        try:
            pair = _extract_pair(engine, llm, run_id, entity_id, attribute, n)
        except (AttributeExtractError, LlmError) as exc:
            typer.echo(f"error {type(exc).__name__}: {exc}", err=True)
            _close_run(engine, run_id, FAILED)
            raise typer.Exit(1) from None
        except SQLAlchemyError:
            raise  # the outer handler closes the run best-effort and prints one line
        except BaseException as exc:
            _close_run_aborting(engine, run_id, exc)
            raise
        _close_run(engine, run_id, FINISHED)
    except SQLAlchemyError as exc:  # URL never echoed
        _exit_database_unavailable(engine, run_id, exc)
    finally:
        engine.dispose()

    extraction = pair.extraction
    checks = "pass" if extraction.checks_passed else "fail"
    typer.echo(f"value {json.dumps(extraction.value)} checks {checks} chunks {pair.num_chunks}")
    for quote in extraction.quotes:
        grounded = "grounded" if quote.grounded else "ungrounded"
        typer.echo(f'quote chunk {quote.chunk_id} {grounded} "{quote.text}"')
    for failure in extraction.failures:
        typer.echo(f"failure {failure}")
    typer.echo(f"attribute_value {pair.value_id if pair.value_id is not None else '-'}")
    if not extraction.checks_passed:
        raise typer.Exit(1)


def _register(stage: str) -> None:
    if stage == "crawl":

        def crawl(
            scope: ScopeOpt,
            layer: LayerOpt = None,
            limit: LimitOpt = None,
            dry_run: DryRunOpt = False,
            url: UrlOpt = None,
            entity_scheme: EntitySchemeOpt = None,
        ) -> None:
            run_crawl(scope, layer, limit, dry_run, list(url or []), entity_scheme)

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
    # resolve, screen, index and assess have their own bodies below
    if _stage not in ("resolve", "screen", "index", "assess"):
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


@app.command(name="screen", help="Screen unscreened entities with the small local model.")
def screen(
    scope: ScopeOpt,
    layer: LayerOpt = None,
    limit: LimitOpt = None,
    dry_run: DryRunOpt = False,
    jobs: JobsOpt = 1,
) -> None:
    """Classify each ``unscreened`` entity of the scope from its stored chunk text."""
    run_screen(scope, layer, limit, dry_run, jobs)


@app.command(name="assess", help="Assess one entity against one rubric criterion.")
def assess(
    scope: ScopeOpt,
    entity_id: Annotated[int | None, typer.Option("--entity", help="Entity id.")] = None,
    criterion: Annotated[str | None, typer.Option("--criterion", help="Criterion id.")] = None,
    n: Annotated[int, typer.Option("--n", help="Evidence chunks sent to the extractor.")] = 6,
    k: Annotated[int, typer.Option("--k", help="BM25 candidates considered (>= --n).")] = 24,
    dry_run: DryRunOpt = False,
    embedding_model: Annotated[
        str | None,
        typer.Option("--embedding-model", help="Stored embedding model; enables hybrid retrieval."),
    ] = None,
    rerank_model: Annotated[
        str | None,
        typer.Option("--rerank-model", help="Reranker model (DM_RERANKER_URL) over the top-k."),
    ] = None,
) -> None:
    """Retrieve evidence for the entity and run the local extractor for the criterion."""
    run_assess(scope, entity_id, criterion, n, k, dry_run, embedding_model, rerank_model)


@app.command(name="index", help="Embed pending chunks with the configured embedding service.")
def index(
    model: IndexModelOpt,
    batch_size: IndexBatchSizeOpt = 64,
    limit: IndexLimitOpt = None,
) -> None:
    """Embed pending chunks via ``embed_pending_chunks``; prints ``embedded <n> batches <m>``."""
    run_index(model, batch_size, limit)


@app.command(name="search", help="Search stored chunks with BM25 and print one line per hit.")
def search(
    query: Annotated[str, typer.Argument(help="Search text.")],
    entity_id: Annotated[
        int | None, typer.Option("--entity", help="Restrict hits to this entity id.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max hits to print.")] = 10,
) -> None:
    """Print ``score chunk C snapshot S seq N`` per hit, plus the heading tail when nonempty."""
    run_search(query, entity_id, limit)


@app.command(name="assessments", help="List the current assessment per entity and criterion.")
def assessments(
    scope: ScopeOpt,
    entity_id: Annotated[int | None, typer.Option("--entity", help="Entity id.")] = None,
    criterion: Annotated[str | None, typer.Option("--criterion", help="Criterion id.")] = None,
    rubric_version: Annotated[
        str | None, typer.Option("--rubric-version", help="Rubric version.")
    ] = None,
) -> None:
    """Print the latest assessment per (entity, criterion, rubric version) and its evidence."""
    run_assessments(scope, entity_id, criterion, rubric_version)


@app.command(name="attribute-set", help="Record one manual attribute value for an entity.")
def attribute_set(
    scope: ScopeOpt,
    entity_id: Annotated[int, typer.Option("--entity", help="Entity id.")],
    attribute: Annotated[str, typer.Option("--attribute", help="Pack attribute id.")],
    value: Annotated[str, typer.Option("--value", help="Value text, parsed per the type.")],
) -> None:
    """Parse, validate and store one attribute value; print its row id."""
    run_attribute_set(scope, entity_id, attribute, value)


@app.command(name="sources", help="Seed the source catalogue from the scope pack.")
def sources(scope: ScopeOpt) -> None:
    """Upsert the pack's sources into the ``source`` table and print the counts."""
    run_sources(scope)


@app.command(name="presence-set", help="Record one manual market presence for an entity.")
def presence_set(
    scope: ScopeOpt,
    entity_id: Annotated[int, typer.Option("--entity", help="Entity id.")],
    country: Annotated[str, typer.Option("--country", help="ISO-3166 alpha-2, uppercase.")],
    presence_type: Annotated[
        str, typer.Option("--type", help="hq, site, field_service, sales or remote.")
    ],
    evidence_id: Annotated[
        int | None, typer.Option("--evidence", help="Chunk id of this entity citing it.")
    ] = None,
) -> None:
    """Validate and store one presence row; print its id."""
    run_presence_set(scope, entity_id, country, presence_type, evidence_id)


@app.command(name="presence-extract", help="Extract market presence from stored chunks.")
def presence_extract(
    scope: ScopeOpt,
    presence_type: Annotated[
        str, typer.Option("--type", help="hq, site, field_service, sales or remote.")
    ],
    entity_ids: Annotated[
        list[int] | None, typer.Option("--entity", help="Entity id (repeatable).")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print matches only.")] = False,
) -> None:
    """Match the pack's presence_terms in each entity's chunks and record presence."""
    run_presence_extract(scope, presence_type, entity_ids, dry_run)


@app.command(name="presences", help="List an entity's market presence and derived layers.")
def presences(
    scope: ScopeOpt,
    entity_id: Annotated[int, typer.Option("--entity", help="Entity id.")],
) -> None:
    """Print presence rows and the layers they place the entity in."""
    run_presences(scope, entity_id)


@app.command(name="attributes", help="List the current attribute values of one entity.")
def attributes(
    scope: ScopeOpt,
    entity_id: Annotated[int, typer.Option("--entity", help="Entity id.")],
) -> None:
    """Print the latest value per attribute for one entity."""
    run_attributes(scope, entity_id)


@app.command(
    name="attributes-export", help="Export current attribute values per entity as JSON Lines."
)
def attributes_export(
    scope: ScopeOpt,
    out: Annotated[Path, typer.Option("--out", help="Output file path.")],
) -> None:
    """Write one attributes object per entity of the scope to ``--out``."""
    run_attributes_export(scope, out)


@app.command(
    name="attributes-extract", help="Extract one attribute value for an entity with evidence."
)
def attributes_extract(
    scope: ScopeOpt,
    entity_id: Annotated[int | None, typer.Option("--entity", help="Entity id.")] = None,
    attribute: Annotated[str | None, typer.Option("--attribute", help="Pack attribute id.")] = None,
    n: Annotated[int, typer.Option("--n", help="Evidence chunks sent to the extractor.")] = 6,
    dry_run: DryRunOpt = False,
    all_entities: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Extract every pack attribute for every in-scope entity (not with --entity).",
        ),
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Max in-scope entities to process (--all).")
    ] = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help=(
                "Skip entity x attribute pairs that already have a current value "
                "(batch --all only; one query per page)."
            ),
        ),
    ] = False,
) -> None:
    """Retrieve BM25 evidence and extract typed values; ``--all`` batches over in-scope entities."""
    run_attributes_extract(
        scope,
        entity_id,
        attribute,
        n,
        dry_run,
        all_entities=all_entities,
        limit=limit,
        skip_existing=skip_existing,
    )


@app.command(name="scores", help="Print per-topic score and coverage for one entity.")
def scores(
    scope: ScopeOpt,
    entity_id: Annotated[int, typer.Option("--entity", help="Entity id.")],
    segment: Annotated[
        list[str] | None,
        typer.Option("--segment", help="Entity segment id (repeatable); default: latest screen."),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", help="Output format: text (default) or json.")
    ] = "text",
) -> None:
    """Score each rubric topic for one entity over its current assessments."""
    run_scores(scope, entity_id, segment, format)


@app.command(
    name="scores-export", help="Export topic scores per screened entity as JSON Lines or CSV."
)
def scores_export(
    scope: ScopeOpt,
    out: Annotated[Path, typer.Option("--out", help="Output file path.")],
    format: Annotated[
        str, typer.Option("--format", help="Output format: jsonl (default) or csv.")
    ] = "jsonl",
) -> None:
    """Write one AG-03 scores object per screened entity of the scope to ``--out``."""
    run_scores_export(scope, out, format)


@app.command(name="mosaic", help="Write a static HTML mosaic page from scores and export.")
def mosaic(
    scope: ScopeOpt,
    scores: Annotated[Path, typer.Option("--scores", help="scores-export JSON Lines file.")],
    export: Annotated[Path, typer.Option("--export", help="export JSON Lines file.")],
    out: Annotated[Path, typer.Option("--out", help="Output HTML file path.")],
    entity_dir: Annotated[
        Path | None,
        typer.Option("--entity-dir", help="Also write one HTML page per entity into DIR."),
    ] = None,
    segment: Annotated[
        list[str] | None,
        typer.Option("--segment", help="Keep only entities in this segment id (repeatable)."),
    ] = None,
    layer: Annotated[
        list[str] | None,
        typer.Option("--layer", help="Keep only entities in this layer id (repeatable)."),
    ] = None,
) -> None:
    """Write one self-contained mosaic HTML page (FE-01) to ``--out``; entity pages (FE-02)."""
    run_mosaic(scope, scores, export, out, entity_dir, segment, layer)


@app.command(name="export", help="Export the current assessment view as JSON Lines.")
def export(
    scope: ScopeOpt,
    format: Annotated[str, typer.Option("--format", help="Output format; only json is defined.")],
    out: Annotated[Path, typer.Option("--out", help="Output file path.")],
    entity_id: Annotated[int | None, typer.Option("--entity", help="Entity id.")] = None,
    criterion: Annotated[str | None, typer.Option("--criterion", help="Criterion id.")] = None,
    rubric_version: Annotated[
        str | None, typer.Option("--rubric-version", help="Rubric version.")
    ] = None,
) -> None:
    """Write one JSON Lines object per current assessment (with evidence) to ``--out``."""
    run_export(scope, format, out, entity_id, criterion, rubric_version)


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
        engine = create_db_engine(settings.database_url)
        try:
            create_all(engine)
            saved = record_candidates(engine, rows)
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - never echo the database URL
        typer.echo(f"error: database unavailable ({type(exc).__name__})", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"saved {saved} candidates")


def _cost_line(row: CostSummaryRow) -> str:
    return (
        f"calls {row.calls} ok {row.ok} error {row.error} tokens_in {row.tokens_in}"
        f" tokens_out {row.tokens_out} cost {row.cost:.6f}"
        f" mean_latency_ms {row.mean_latency_ms:.1f} max_latency_ms {row.max_latency_ms}"
    )


def run_cost_report(run_id: int | None) -> None:
    """Print one line per (stage, purpose, model) from ``llm_call`` plus a ``total`` line (CO-01).

    Aggregation happens in SQL via :func:`core.llm_calls.summarise_llm_calls`; an unknown
    ``run_id`` exits 2 with ``error: run <ID> not found``; database errors exit 2.
    """
    try:
        settings = load_settings()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    engine = _open_engine(settings)
    try:
        with Session(engine) as session:
            if run_id is not None and session.get(Run, run_id) is None:
                typer.echo(f"error: run {run_id} not found", err=True)
                raise typer.Exit(2)
            rows = summarise_llm_calls(session, run_id=run_id)
    except SQLAlchemyError as exc:
        _exit_database_unavailable(engine, None, exc)
    finally:
        engine.dispose()
    for row in rows:
        typer.echo(f"stage {row.stage} purpose {row.purpose} model {row.model} {_cost_line(row)}")
    calls = sum(r.calls for r in rows)
    total = CostSummaryRow(
        stage="",
        purpose="",
        model="",
        calls=calls,
        ok=sum(r.ok for r in rows),
        error=sum(r.error for r in rows),
        tokens_in=sum(r.tokens_in for r in rows),
        tokens_out=sum(r.tokens_out for r in rows),
        cost=sum(r.cost for r in rows),
        mean_latency_ms=(sum(r.mean_latency_ms * r.calls for r in rows) / calls if calls else 0.0),
        max_latency_ms=max((r.max_latency_ms for r in rows), default=0),
    )
    typer.echo(f"total {_cost_line(total)}")


@app.command(name="cost-report", help="Print LLM call cost and latency per stage, purpose, model.")
def cost_report(
    run_id: Annotated[int | None, typer.Option("--run", help="Only this run id.")] = None,
) -> None:
    """Aggregate ``llm_call`` rows per (run.stage, purpose, model); read-only (CO-01)."""
    run_cost_report(run_id)


def main() -> None:
    """Console-script entry point for ``landscape``."""
    app()
