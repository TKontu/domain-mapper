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
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from itertools import chain, islice
from pathlib import Path
from typing import Annotated, NoReturn
from urllib.parse import urlsplit

import typer
from sqlalchemy import create_engine
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
from core.assess import AssessError, extract_criterion
from core.assessment_store import record_assessment
from core.candidate_store import NewCandidate, record_candidates
from core.config import Settings, load_settings
from core.embed_store import EmbedStoreError, embed_pending_chunks
from core.entity_store import STATUS_UNSCREENED, EntityRef, iter_entities
from core.ingest import IngestResult, NonSuccessStatusError, ingest_page
from core.keyword_search import search_chunks
from core.links import crawl_key, extract_links
from core.resolve import apply_resolution, group_candidates
from core.retrieve import retrieve_evidence
from core.runs import FAILED, FINISHED, RUNNING, Run, create_all
from core.scope_pack import (
    CrawlPolicy,
    Criterion,
    Rubric,
    ScopePackError,
    load_scope_pack,
    scope_pack_json_schemas,
)
from core.screen import ScreenError, screen_entity
from core.screen_store import DEFAULT_MAX_CHARS, load_entity_text, record_screen_result
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

_SECRET_FIELDS = frozenset({"cloud_api_key"})

app = typer.Typer(help="Domain Mapper pipeline stages.", no_args_is_help=True)

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
        engine = create_engine(settings.database_url)
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
            for entity in chain((first,), entities):
                _crawl_entity(crawler, entity, policy, scheme)
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
    if failed:
        raise typer.Exit(1)


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


def _make_screen_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the small local model used for screening; the test seam for a fake adapter.

    One budget per invocation (= per run): ``settings.llm_budget_usd`` caps the run's spend.
    """
    return build_llm(settings, "local", db=db, budget_usd=settings.llm_budget_usd)


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


def run_screen(scope: Path, layer: str | None, limit: int | None, dry_run: bool) -> None:
    """Screen each ``unscreened`` entity of ``scope``'s id with the small local model.

    One entity at a time in id order, at most ``limit``. Prints one line per entity and the final
    ``screened S skipped K errors E``. Exits 1 if any entity errored, 2 on bad setup.
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
        try:
            entities = iter_entities(engine, pack.scope.id, limit=limit, status=STATUS_UNSCREENED)
            for entity in entities:
                text = load_entity_text(engine, entity.id)
                if not text.strip():
                    typer.echo(f"skipped {entity.id} no-text")
                    skipped += 1
                    continue
                try:
                    result = screen_entity(
                        llm,
                        pack,
                        name=entity.name,
                        domain=entity.primary_domain,
                        text=text,
                        run_id=run_id,
                        max_chars=DEFAULT_MAX_CHARS,
                    )
                except LlmBudgetExceeded as exc:
                    # The run's budget is spent: stop cleanly, the rest stay unscreened.
                    typer.echo(f"error {entity.id} {type(exc).__name__}: {exc}")
                    errors += 1
                    break
                except (ScreenError, LlmError) as exc:
                    # Invalid output or a failed call leaves the entity unscreened for a retry.
                    typer.echo(f"error {entity.id} {type(exc).__name__}: {exc}")
                    errors += 1
                    continue
                status = record_screen_result(
                    engine, entity_id=entity.id, run_id=run_id, result=result
                )
                segments = ",".join(result.segments) if result.segments else "-"
                typer.echo(f"ok {entity.id} {status} {result.confidence:.2f} {segments}")
                screened += 1
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
            stats = embed_pending_chunks(
                engine,
                embedder,
                model=model,
                batch_size=batch_size,
                limit=limit,
            )
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


def _make_assess_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the local extractor model for assess (§6 routing); the test seam for a fake adapter."""
    return build_llm(settings, "local", db=db, budget_usd=settings.llm_budget_usd)


def _make_verify_llm(settings: Settings, db: DbTarget) -> LLM:
    """Build the local verifier from ``settings.verifier_model`` (§6 routing); the test seam."""
    verifier_settings = dataclasses.replace(settings, local_model=settings.verifier_model)
    return build_llm(verifier_settings, "local", db=db, budget_usd=settings.llm_budget_usd)


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
) -> None:
    """Classify each ``unscreened`` entity of the scope from its stored chunk text."""
    run_screen(scope, layer, limit, dry_run)


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
