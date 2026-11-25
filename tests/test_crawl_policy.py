"""Tests for the crawl politeness policy (CR-02).

robots.txt is reached only through the injected ``fetch_robots`` callable and
the rate limiter through an injected clock and sleep, so this suite never
touches the network and never really sleeps. Fixtures use example.com /
example.org only (D-003).
"""

from __future__ import annotations

import pytest

from adapters.crawl_policy import (
    MAX_ROBOTS_BYTES,
    CrawlPolicyError,
    HostRateLimiter,
    RobotsPolicy,
)

ROBOTS_URL = "https://example.com/robots.txt"
USER_AGENT = "PoliteBot/1.0"

ALLOW_PRIVATE_ROBOTS = b"User-agent: *\nDisallow: /private\n"


class FakeRobots:
    """Canned ``(status, body)`` robots.txt responses keyed by URL.

    Every requested URL is recorded in :attr:`calls`; an unknown URL answers
    ``404`` (no robots.txt). A response that is an exception instance is
    raised, standing in for a transport failure. A parameterised response may
    also be a :class:`list` of responses, consumed in order.
    """

    def __init__(self, responses: dict[str, object] | None = None) -> None:
        self._responses = dict(responses or {})
        self.calls: list[str] = []

    def __call__(self, url: str) -> tuple[int, bytes]:
        self.calls.append(url)
        response = self._responses.get(url, (404, b""))
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, tuple)
        return response


class FakeClock:
    """Monotonic-looking clock whose ``sleep`` advances time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_policy(
    body: bytes = ALLOW_PRIVATE_ROBOTS,
    *,
    status: int = 200,
    user_agent: str = USER_AGENT,
    url: str = ROBOTS_URL,
) -> tuple[RobotsPolicy, FakeRobots]:
    fetch = FakeRobots({url: (status, body)})
    return RobotsPolicy(fetch, user_agent), fetch


def make_limiter(min_interval: float = 1.0) -> tuple[HostRateLimiter, FakeClock]:
    clock = FakeClock()
    return HostRateLimiter(min_interval, clock.clock, clock.sleep), clock


# --- robots.txt -----------------------------------------------------------


def test_disallows_private_path_and_allows_public_path() -> None:
    policy, fetch = make_policy()
    assert policy.allowed("https://example.com/public/page") is True
    assert policy.allowed("https://example.com/private/secret") is False
    assert fetch.calls == [ROBOTS_URL]


def test_user_agent_section_is_used() -> None:
    body = b"User-agent: PoliteBot\nDisallow: /bot-only\n\nUser-agent: *\nDisallow: /everyone\n"
    polite, _ = make_policy(body, user_agent="PoliteBot/1.0")
    other, _ = make_policy(body, user_agent="OtherBot/1.0")
    assert polite.allowed("https://example.com/bot-only") is False
    assert polite.allowed("https://example.com/everyone") is True
    assert other.allowed("https://example.com/bot-only") is True
    assert other.allowed("https://example.com/everyone") is False


def test_one_fetch_per_scheme_and_host() -> None:
    fetch = FakeRobots({ROBOTS_URL: (200, ALLOW_PRIVATE_ROBOTS)})
    policy = RobotsPolicy(fetch, USER_AGENT)
    policy.allowed("https://example.com/a")
    policy.allowed("https://example.com/b")
    assert fetch.calls == [ROBOTS_URL]

    policy.allowed("https://example.org/a")
    assert fetch.calls == [ROBOTS_URL, "https://example.org/robots.txt"]

    policy.allowed("http://example.com/a")
    assert fetch.calls[-1] == "http://example.com/robots.txt"
    assert len(fetch.calls) == 3


def test_port_is_part_of_the_cache_key_and_the_robots_url() -> None:
    url = "https://example.com:8443/robots.txt"
    fetch = FakeRobots({url: (200, b"User-agent: *\nDisallow: /\n")})
    policy = RobotsPolicy(fetch, USER_AGENT)
    assert policy.allowed("https://example.com:8443/anything") is False
    assert policy.allowed("https://example.com:8443/other") is False
    assert fetch.calls == [url]


@pytest.mark.parametrize("status", [404, 410])
def test_missing_robots_allows_everything(status: int) -> None:
    policy, _ = make_policy(b"User-agent: *\nDisallow: /\n", status=status)
    assert policy.allowed("https://example.com/anything") is True


@pytest.mark.parametrize("status", [401, 403])
def test_unauthorised_robots_disallows_everything(status: int) -> None:
    policy, _ = make_policy(b"User-agent: *\nAllow: /\n", status=status)
    assert policy.allowed("https://example.com/anything") is False


@pytest.mark.parametrize("status", [301, 500, 503])
def test_other_failure_statuses_disallow_and_are_cached(status: int) -> None:
    policy, fetch = make_policy(b"", status=status)
    assert policy.allowed("https://example.com/anything") is False
    assert policy.allowed("https://example.com/other") is False
    assert fetch.calls == [ROBOTS_URL]


@pytest.mark.parametrize(
    "error",
    [OSError("connection reset"), TimeoutError(), RuntimeError("boom")],
)
def test_fetch_exception_disallows_and_is_cached(error: BaseException) -> None:
    fetch = FakeRobots({ROBOTS_URL: error})
    policy = RobotsPolicy(fetch, USER_AGENT)
    assert policy.allowed("https://example.com/anything") is False
    assert policy.allowed("https://example.com/other") is False
    assert fetch.calls == [ROBOTS_URL]


def test_rules_within_the_body_cap_are_honoured() -> None:
    body = b"User-agent: *\nDisallow: /early\n" + b"# " + b"x" * MAX_ROBOTS_BYTES
    policy, _ = make_policy(body)
    assert policy.allowed("https://example.com/early") is False


def test_rules_beyond_the_body_cap_are_ignored() -> None:
    body = b"User-agent: *\n# " + b"x" * MAX_ROBOTS_BYTES + b"\nDisallow: /late\n"
    oversized, _ = make_policy(body)
    assert oversized.allowed("https://example.com/late") is True

    # Control: the very same rule is honoured when the body fits under the cap.
    small, _ = make_policy(b"User-agent: *\nDisallow: /late\n")
    assert small.allowed("https://example.com/late") is False


def test_oversized_invalid_utf8_body_does_not_crash() -> None:
    body = b"User-agent: *\n" + b"\xff\xfe" * (MAX_ROBOTS_BYTES // 2 + 16)
    policy, _ = make_policy(body)
    assert policy.allowed("https://example.com/anything") is True


def test_body_cap_is_512_kib() -> None:
    assert MAX_ROBOTS_BYTES == 512 * 1024


def test_crawl_delay_is_parsed_from_robots() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 2\n")
    delay = policy.crawl_delay("https://example.com/page")
    assert delay == 2.0
    assert isinstance(delay, float)


def test_crawl_delay_is_none_when_absent_or_robots_missing() -> None:
    absent, _ = make_policy(ALLOW_PRIVATE_ROBOTS)
    assert absent.crawl_delay("https://example.com/page") is None
    missing, _ = make_policy(b"", status=404)
    assert missing.crawl_delay("https://example.com/page") is None


def test_crawl_delay_accepts_a_decimal_value() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 1.5\n")
    delay = policy.crawl_delay("https://example.com/page")
    assert delay == 1.5
    assert isinstance(delay, float)


def test_zero_crawl_delay_is_kept() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 0\n")
    assert policy.crawl_delay("https://example.com/page") == 0.0


def test_decimal_crawl_delay_does_not_disturb_allow_rules() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 0.25\nDisallow: /private\n")
    assert policy.allowed("https://example.com/private/secret") is False
    assert policy.allowed("https://example.com/public") is True
    assert policy.crawl_delay("https://example.com/public") == 0.25


def test_crawl_delay_tolerates_space_and_a_trailing_comment() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay:   0.75   # politely\n")
    assert policy.crawl_delay("https://example.com/page") == 0.75


def test_specific_agent_delay_wins_over_the_star_fallback() -> None:
    body = b"User-agent: PoliteBot\nCrawl-delay: 4.5\n\nUser-agent: *\nCrawl-delay: 1\n"
    polite, _ = make_policy(body, user_agent="PoliteBot/1.0")
    other, _ = make_policy(body, user_agent="OtherBot/1.0")
    assert polite.crawl_delay("https://example.com/page") == 4.5
    assert other.crawl_delay("https://example.com/page") == 1.0


def test_matching_specific_group_without_delay_does_not_fall_back_to_star() -> None:
    # The stdlib stops at the first matching specific group even when that group
    # carries no Crawl-delay; the ``*`` group is only a fallback.
    body = b"User-agent: PoliteBot\nDisallow: /bot\n\nUser-agent: *\nCrawl-delay: 3\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") is None


def test_crawl_delay_applies_to_every_agent_in_a_group() -> None:
    body = b"User-agent: ABot\nUser-agent: PoliteBot\nCrawl-delay: 2.5\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") == 2.5


def test_first_matching_specific_group_wins() -> None:
    body = b"User-agent: PoliteBot\nCrawl-delay: 2.5\n\nUser-agent: PoliteBot\nCrawl-delay: 9\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") == 2.5


def test_request_rate_alone_still_defines_a_specific_group() -> None:
    # The stdlib treats request-rate as starting a group, so this group still
    # shadows the ``*`` delay even though it carries no Crawl-delay.
    body = b"User-agent: PoliteBot\nRequest-rate: 1/5\n\nUser-agent: *\nCrawl-delay: 3\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") is None


def test_an_invalid_later_delay_keeps_an_earlier_valid_one() -> None:
    # Parity with the stdlib: an unparsable value leaves the group's delay as-is.
    body = b"User-agent: *\nCrawl-delay: 1.5\nCrawl-delay: nope\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") == 1.5


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "-1",
        "-0.5",
        "+1.5",
        "inf",
        "-inf",
        "Infinity",
        "nan",
        "NaN",
        "1e3",
        "1.5.2",
        "1,5",
        "1.5s",
        "1_0",
        "9" * 400,  # overflows float() to inf
    ],
)
def test_invalid_crawl_delay_values_are_none(value: str) -> None:
    policy, _ = make_policy(f"User-agent: *\nCrawl-delay: {value}\n".encode())
    assert policy.crawl_delay("https://example.com/page") is None


@pytest.mark.parametrize("status", [404, 500])
def test_crawl_delay_is_none_for_blanket_rules(status: int) -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 1.5\n", status=status)
    assert policy.crawl_delay("https://example.com/page") is None


def test_crawl_delay_is_none_when_the_fetch_fails() -> None:
    fetch = FakeRobots({ROBOTS_URL: OSError("connection reset")})
    policy = RobotsPolicy(fetch, USER_AGENT)
    assert policy.crawl_delay("https://example.com/page") is None


def test_crawl_delay_beyond_the_body_cap_is_ignored() -> None:
    body = b"User-agent: *\n# " + b"x" * MAX_ROBOTS_BYTES + b"\nCrawl-delay: 5\n"
    policy, _ = make_policy(body)
    assert policy.crawl_delay("https://example.com/page") is None


@pytest.mark.parametrize(
    "url",
    ["", "example.com/page", "ftp://example.com/x", "https:///path", "mailto:a@example.com"],
)
def test_policy_rejects_non_http_urls(url: str) -> None:
    policy, fetch = make_policy()
    with pytest.raises(CrawlPolicyError):
        policy.allowed(url)
    with pytest.raises(CrawlPolicyError):
        policy.crawl_delay(url)
    assert fetch.calls == []


def test_policy_rejects_an_unparsable_port() -> None:
    policy, _ = make_policy()
    with pytest.raises(CrawlPolicyError):
        policy.allowed("https://example.com:notaport/x")


@pytest.mark.parametrize("agent", ["", "   "])
def test_policy_requires_a_user_agent(agent: str) -> None:
    with pytest.raises(CrawlPolicyError):
        RobotsPolicy(FakeRobots(), agent)


# --- per-host rate limiter ------------------------------------------------


def test_first_call_for_a_host_never_sleeps() -> None:
    limiter, clock = make_limiter(2.0)
    limiter.wait("https://example.com/a")
    assert clock.sleeps == []


def test_same_host_calls_are_spaced_by_the_minimum_interval() -> None:
    limiter, clock = make_limiter(2.0)
    limiter.wait("https://example.com/a")
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(2.0)]
    limiter.wait("https://example.com/c")
    assert clock.sleeps == [pytest.approx(2.0), pytest.approx(2.0)]
    assert clock.now == pytest.approx(1004.0)


def test_different_hosts_never_wait_for_each_other() -> None:
    limiter, clock = make_limiter(5.0)
    limiter.wait("https://example.com/a")
    limiter.wait("https://example.org/a")
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(5.0)]


def test_no_sleep_when_the_interval_already_elapsed() -> None:
    limiter, clock = make_limiter(1.0)
    limiter.wait("https://example.com/a")
    clock.now += 10.0
    limiter.wait("https://example.com/b")
    assert clock.sleeps == []


def test_override_sets_a_longer_per_host_interval() -> None:
    limiter, clock = make_limiter(1.0)
    limiter.wait("https://example.com/a")
    limiter.set_interval("example.com", 7.5)
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(7.5)]

    limiter.wait("https://example.org/a")
    assert clock.sleeps == [pytest.approx(7.5)]
    limiter.wait("https://example.org/b")
    assert clock.sleeps == [pytest.approx(7.5), pytest.approx(1.0)]


def test_override_never_shortens_below_the_minimum() -> None:
    limiter, clock = make_limiter(5.0)
    limiter.wait("https://example.com/a")
    limiter.set_interval("example.com", 0.25)
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(5.0)]


def test_override_host_key_is_case_insensitive() -> None:
    limiter, clock = make_limiter(1.0)
    limiter.wait("https://example.com/a")
    limiter.set_interval("EXAMPLE.com", 3.0)
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(3.0)]


def test_zero_min_interval_never_sleeps() -> None:
    limiter, clock = make_limiter(0.0)
    limiter.wait("https://example.com/a")
    limiter.wait("https://example.com/b")
    assert clock.sleeps == []


@pytest.mark.parametrize("interval", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_rate_limiter_rejects_invalid_min_interval(interval: object) -> None:
    clock = FakeClock()
    with pytest.raises(CrawlPolicyError):
        HostRateLimiter(interval, clock.clock, clock.sleep)  # type: ignore[arg-type]


@pytest.mark.parametrize("seconds", [-0.5, float("nan"), float("inf"), True, "2"])
def test_set_interval_rejects_invalid_values(seconds: object) -> None:
    limiter, _ = make_limiter()
    with pytest.raises(CrawlPolicyError):
        limiter.set_interval("example.com", seconds)  # type: ignore[arg-type]


@pytest.mark.parametrize("host", ["", "   ", "https://example.com", "example.com/path", "a@b"])
def test_set_interval_rejects_non_host_keys(host: str) -> None:
    limiter, _ = make_limiter()
    with pytest.raises(CrawlPolicyError):
        limiter.set_interval(host, 1.0)


@pytest.mark.parametrize("url", ["", "example.com", "ftp://example.com/x"])
def test_rate_limiter_rejects_non_http_urls(url: str) -> None:
    limiter, clock = make_limiter()
    with pytest.raises(CrawlPolicyError):
        limiter.wait(url)
    assert clock.sleeps == []


def test_crawl_delay_can_feed_the_host_override() -> None:
    policy, _ = make_policy(b"User-agent: *\nCrawl-delay: 3\n")
    limiter, clock = make_limiter(0.5)
    delay = policy.crawl_delay("https://example.com/a")
    assert delay == 3.0
    limiter.set_interval("example.com", delay)
    limiter.wait("https://example.com/a")
    limiter.wait("https://example.com/b")
    assert clock.sleeps == [pytest.approx(3.0)]
