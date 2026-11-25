"""Crawl politeness policy: robots.txt and per-host rate limiting (CR-02).

ARCHITECTURE.md §4 (Crawl & snapshot) and §9 (Legal & ethical constraints)
require a polite crawler: respect ``robots.txt``, keep a per-host rate limit
and identify with a stable user agent. This module is the policy half; the
fetch itself stays behind :mod:`adapters.fetcher`.

:class:`RobotsPolicy` answers ``allowed(url)`` and reports the site's
``Crawl-delay``. It performs no I/O: the caller injects
``fetch_robots(url) -> (status, body)``, so the same policy drives the real
fetcher and the offline tests. Robots handling is deliberately conservative:

* one parse is cached per scheme + host + port for the life of the policy
  object (one crawl run); the cache is never evicted (residual: it grows by one
  entry per host crawled);
* ``404``/``410`` mean "no robots.txt" → everything on that host is allowed;
* ``401``/``403`` → nothing on that host is allowed;
* any other non-2xx status (5xx, redirects, other 4xx, …) and any exception
  raised by ``fetch_robots`` → nothing on that host is allowed, and that
  decision is cached like a parse;
* at most :data:`MAX_ROBOTS_BYTES` of the body is decoded and parsed; rules
  beyond the cap are ignored rather than buffered. Invalid UTF-8 is replaced,
  never raised.

``Crawl-delay`` is parsed from the same capped body as the rules, because
:class:`urllib.robotparser.RobotFileParser` accepts integer seconds only.
Integer and decimal seconds are accepted (``1`` and ``1.5``); invalid,
negative and non-finite values read as ``None``. Groups follow the stdlib
parser's matching: a specific group named for the agent wins over the ``*``
group, and a matching specific group without a delay does not fall back.

:class:`HostRateLimiter` spaces requests by host. It keeps only the last
request time per host: two calls for the same host are at least
``max(min_interval, per-host override)`` apart, different hosts never wait for
each other, and the first call for a host never sleeps. Clock and sleep are
injected; production passes :func:`time.monotonic` and :func:`time.sleep`.

Thread safety is out of scope (residual): one policy and one limiter per crawl
run.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import SplitResult, unquote, urlsplit
from urllib.robotparser import RobotFileParser

#: Largest robots.txt body that is decoded and parsed (512 KiB).
MAX_ROBOTS_BYTES: Final = 512 * 1024

#: Path appended to an origin to form the robots.txt URL.
ROBOTS_PATH: Final = "/robots.txt"

_HTTP_SCHEMES: Final = frozenset({"http", "https"})
_ALLOW_ALL_STATUSES: Final = frozenset({404, 410})
_DISALLOW_ALL_STATUSES: Final = frozenset({401, 403})

#: ``fetch_robots``: ``(robots_url) -> (status, body)``; may raise on failure.
RobotsFetch = Callable[[str], tuple[int, bytes]]


class CrawlPolicyError(ValueError):
    """A crawl-policy input is invalid: a non-http URL, host or interval."""


def _split_http(url: str) -> SplitResult:
    """Parse an http(s) URL, rejecting anything else."""
    if not isinstance(url, str):
        raise CrawlPolicyError(f"not an http(s) URL: {url!r}")
    parts = urlsplit(url)
    if parts.scheme not in _HTTP_SCHEMES or not parts.hostname:
        raise CrawlPolicyError(f"not an http(s) URL: {url!r}")
    return parts


def _port(parts: SplitResult, url: str) -> int | None:
    try:
        return parts.port
    except ValueError as exc:  # unparsable or out-of-range port
        raise CrawlPolicyError(f"invalid port in URL: {url!r}") from exc


def _origin(url: str) -> tuple[str, str, int | None, str]:
    """Return ``(scheme, host, port, robots_url)`` for an http(s) URL.

    The robots URL is rebuilt from scheme + host + port, so userinfo, query and
    fragment in the crawled URL never leak into it.
    """
    parts = _split_http(url)
    assert parts.hostname is not None  # guaranteed by _split_http
    host = parts.hostname
    port = _port(parts, url)
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return parts.scheme, host, port, f"{parts.scheme}://{netloc}{ROBOTS_PATH}"


_DELAY_RE: Final = re.compile(r"\d+(?:\.\d+)?")


def _parse_delay(value: str) -> float | None:
    """Parse a ``Crawl-delay`` value as non-negative finite seconds.

    Only plain decimal notation is accepted: ``2``, ``1.5``, ``0.25``.
    Scientific notation, signs, underscores and non-finite spellings are
    invalid, as are values that overflow ``float``.
    """
    text = value.strip()
    if not _DELAY_RE.fullmatch(text):
        return None
    seconds = float(text)
    return seconds if math.isfinite(seconds) else None


@dataclass(frozen=True)
class _DelayEntry:
    """One robots.txt group's user agents and optional ``Crawl-delay``."""

    user_agents: tuple[str, ...]
    delay: float | None = None
    is_default: bool = False

    def applies_to(self, user_agent: str) -> bool:
        name = user_agent.split("/")[0].lower()
        return any(agent == "*" or agent.lower() in name for agent in self.user_agents)


@dataclass(frozen=True)
class _DelayRules:
    """``Crawl-delay`` values by group, matched like the stdlib parser."""

    entries: tuple[_DelayEntry, ...] = ()

    def crawl_delay(self, user_agent: str) -> float | None:
        for entry in self.entries:
            if not entry.is_default and entry.applies_to(user_agent):
                return entry.delay
        for entry in self.entries:
            if entry.is_default:
                return entry.delay
        return None


def _parse_crawl_delays(lines: Iterable[str]) -> _DelayRules:
    """Collect ``Crawl-delay`` per group, mirroring the stdlib grouping.

    The state machine is the one in :class:`urllib.robotparser.RobotFileParser`
    (start, user-agent seen, rule seen), so a group means the same thing to
    the policy's delay as it does to the parser's allow/deny rules.
    """
    entries: list[_DelayEntry] = []
    user_agents: list[str] = []
    delay: float | None = None
    state = 0

    def add_entry() -> None:
        nonlocal user_agents, delay
        entries.append(_DelayEntry(tuple(user_agents), delay, "*" in user_agents))
        user_agents, delay = [], None

    for line in lines:
        if not line:
            if state == 1:
                user_agents, delay = [], None
                state = 0
            elif state == 2:
                add_entry()
                state = 0
            continue
        comment = line.find("#")
        if comment >= 0:
            line = line[:comment]
        line = line.strip()
        if not line:
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        field = field.strip().lower()
        if field == "user-agent":
            if state == 2:
                add_entry()
            user_agents.append(unquote(value.strip()))
            state = 1
        elif field in ("allow", "disallow", "crawl-delay", "request-rate"):
            if state == 0:
                continue
            if field == "crawl-delay":
                parsed = _parse_delay(value)
                if parsed is not None:
                    delay = parsed
            state = 2
    if state == 2:
        add_entry()
    return _DelayRules(tuple(entries))


@dataclass(frozen=True)
class _HostRules:
    """One origin's robots.txt outcome: a parsed file, or a blanket decision.

    ``parser`` is ``None`` for the blanket cases ("no robots.txt" → everything
    allowed, any failure → nothing allowed), which carry no crawl delay.
    ``delays`` holds the same file's ``Crawl-delay`` groups, parsed by
    :func:`_parse_crawl_delays` because the stdlib parser drops fractions.
    """

    parser: RobotFileParser | None = None
    delays: _DelayRules | None = None
    allowed: bool = True

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if self.parser is None:
            return self.allowed
        return self.parser.can_fetch(user_agent, url)

    def crawl_delay(self, user_agent: str) -> float | None:
        if self.delays is None:
            return None
        return self.delays.crawl_delay(user_agent)


def _allow_all() -> _HostRules:
    return _HostRules(allowed=True)


def _disallow_all() -> _HostRules:
    return _HostRules(allowed=False)


def _robots_text(body: object) -> str:
    """Decode at most :data:`MAX_ROBOTS_BYTES` of a robots.txt body."""
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise TypeError(f"robots.txt body must be bytes, not {type(body).__name__}")
    return bytes(body[:MAX_ROBOTS_BYTES]).decode("utf-8", errors="replace")


class RobotsPolicy:
    """``robots.txt`` decisions for one user agent, cached per origin.

    At most ``max_origins`` origins are cached; the least recently used one is
    evicted and its robots.txt is fetched again on its next use. Not thread-safe.
    """

    def __init__(
        self, fetch_robots: RobotsFetch, user_agent: str, *, max_origins: int = 1024
    ) -> None:
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise CrawlPolicyError("user_agent must be a non-empty string")
        if not callable(fetch_robots):
            raise CrawlPolicyError("fetch_robots must be callable")
        if isinstance(max_origins, bool) or not isinstance(max_origins, int) or max_origins < 1:
            raise CrawlPolicyError(f"max_origins must be an int >= 1: {max_origins!r}")
        self._max_origins = max_origins
        self._fetch_robots = fetch_robots
        self._user_agent = user_agent
        self._cache: OrderedDict[tuple[str, str, int | None], _HostRules] = OrderedDict()

    @property
    def user_agent(self) -> str:
        return self._user_agent

    def allowed(self, url: str) -> bool:
        """Whether ``user_agent`` may fetch ``url`` under its host's robots.txt."""
        return self._rules_for(url).can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """The host's ``Crawl-delay`` for ``user_agent``, in seconds, if any."""
        return self._rules_for(url).crawl_delay(self._user_agent)

    def _rules_for(self, url: str) -> _HostRules:
        scheme, host, port, robots_url = _origin(url)
        key = (scheme, host, port)
        rules = self._cache.get(key)
        if rules is None:
            rules = self._load(robots_url)
            self._cache[key] = rules
            if len(self._cache) > self._max_origins:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return rules

    def _load(self, robots_url: str) -> _HostRules:
        try:
            status, body = self._fetch_robots(robots_url)
        except Exception:  # noqa: BLE001 - any transport failure is "disallow"
            return _disallow_all()
        if isinstance(status, bool) or not isinstance(status, int):
            return _disallow_all()
        if status in _ALLOW_ALL_STATUSES:
            return _allow_all()
        if status in _DISALLOW_ALL_STATUSES or not 200 <= status < 300:
            return _disallow_all()
        try:
            lines = _robots_text(body).splitlines()
            parser = RobotFileParser()
            parser.parse(lines)
            delays = _parse_crawl_delays(lines)
        except Exception:  # noqa: BLE001 - an unparsable body is "disallow"
            return _disallow_all()
        return _HostRules(parser=parser, delays=delays)


def _check_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrawlPolicyError(f"{name} must be a finite number >= 0: {value!r}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise CrawlPolicyError(f"{name} must be a finite number >= 0: {value!r}")
    return seconds


def _host_key(host: object) -> str:
    """Normalise a bare hostname used as an override key."""
    if not isinstance(host, str):
        raise CrawlPolicyError(f"host must be a hostname string: {host!r}")
    key = host.strip().lower()
    if not key or any(ch in key for ch in "/:@?#") or any(ch.isspace() for ch in key):
        raise CrawlPolicyError(f"host must be a bare hostname without a port: {host!r}")
    return key


class HostRateLimiter:
    """Keeps requests to one host at least ``min_interval`` seconds apart.

    ``clock`` returns seconds from a monotonic source and ``sleep(seconds)``
    blocks; both are injected so tests never sleep. ``set_interval`` raises the
    spacing for one host (e.g. from its robots ``Crawl-delay``); it can never
    shorten it below ``min_interval``.
    """

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._min_interval = _check_interval(min_interval, "min_interval")
        if not callable(clock) or not callable(sleep):
            raise CrawlPolicyError("clock and sleep must be callable")
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self._overrides: dict[str, float] = {}

    @property
    def min_interval(self) -> float:
        return self._min_interval

    def set_interval(self, host: str, seconds: float) -> None:
        """Set the spacing for ``host`` (a bare hostname, as in ``urlsplit``)."""
        self._overrides[_host_key(host)] = _check_interval(seconds, "seconds")

    def wait(self, url: str) -> None:
        """Sleep just long enough for this host's spacing, then record the call."""
        host = _origin(url)[1]
        interval = max(self._min_interval, self._overrides.get(host, 0.0))
        now = float(self._clock())
        last = self._last.get(host)
        if last is not None and now - last < interval:
            self._sleep(interval - (now - last))
            now = float(self._clock())
        self._last[host] = now
