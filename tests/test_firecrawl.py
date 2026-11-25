"""Tests for the Firecrawl implementation of the Fetcher protocol (AD-13). No network."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pytest

from adapters.fetcher import DEFAULT_MAX_BYTES, Fetcher, FetchError
from adapters.firecrawl import DEFAULT_TIMEOUT_MS, FirecrawlFetcher

BASE = "http://firecrawl.invalid"
TARGET = "https://www.example.com/page"
RAW_HTML = "<html><body><p>Example</p></body></html>"


class FakeTransport:
    """Records POST calls and returns one canned ``(status, body)`` pair."""

    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.status = status
        self.body = envelope() if body is None else body
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str]) -> tuple[int, bytes]:
        self.calls.append((url, body, dict(headers)))
        return self.status, self.body


def envelope(
    raw_html: Any = RAW_HTML,
    *,
    include_raw_html: bool = True,
    include_status_code: bool = True,
    status_code: Any = 200,
    url: str | None = None,
    source_url: str | None = None,
    content_type: str | None = None,
) -> bytes:
    """A Firecrawl-looking ``/v1/scrape`` success payload."""
    metadata: dict[str, Any] = {}
    if include_status_code:
        metadata["statusCode"] = status_code
    if url is not None:
        metadata["url"] = url
    if source_url is not None:
        metadata["sourceURL"] = source_url
    if content_type is not None:
        metadata["contentType"] = content_type
    data: dict[str, Any] = {"metadata": metadata}
    if include_raw_html:
        data["rawHtml"] = raw_html
    return json.dumps({"success": True, "data": data}).encode()


def run(
    body: bytes | None = None, *, status: int = 200, url: str = TARGET, **kwargs: Any
) -> tuple[FetchResult, FakeTransport]:
    transport = FakeTransport(status, body)
    result = FirecrawlFetcher(BASE, transport, **kwargs).fetch(url)
    return result, transport


def test_satisfies_protocol() -> None:
    assert isinstance(FirecrawlFetcher(BASE, FakeTransport()), Fetcher)

    class NoFetch:
        pass

    assert not isinstance(NoFetch(), Fetcher)


def test_request_url_body_and_headers() -> None:
    _, transport = run()
    assert len(transport.calls) == 1
    url, body, headers = transport.calls[0]
    assert url == BASE + "/v1/scrape"
    assert json.loads(body) == {
        "url": TARGET,
        "formats": ["rawHtml"],
        "timeout": DEFAULT_TIMEOUT_MS,
    }
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"


def test_trailing_slash_in_base_url() -> None:
    transport = FakeTransport()
    FirecrawlFetcher(BASE + "/", transport).fetch(TARGET)
    assert transport.calls[0][0] == BASE + "/v1/scrape"


def test_custom_timeout_is_sent_in_ms() -> None:
    _, transport = run(timeout_ms=1234)
    assert json.loads(transport.calls[0][1])["timeout"] == 1234


def test_happy_path_maps_every_field() -> None:
    result, _ = run(
        envelope(
            url="https://www.example.com/final",
            content_type="text/html; charset=utf-8",
        )
    )
    assert result.final_url == "https://www.example.com/final"
    assert result.status == 200
    assert result.content_type == "text/html; charset=utf-8"
    assert result.body_bytes == RAW_HTML.encode("utf-8")
    assert isinstance(result.fetched_at, datetime)
    assert result.fetched_at.tzinfo is not None
    assert result.fetched_at.utcoffset() is not None
    assert result.headers == {}


def test_source_url_used_when_url_absent() -> None:
    result, _ = run(envelope(source_url="https://www.example.com/moved"))
    assert result.final_url == "https://www.example.com/moved"


def test_url_preferred_over_source_url() -> None:
    result, _ = run(
        envelope(url=" https://www.example.com/url ", source_url="https://www.example.com/source")
    )
    assert result.final_url == "https://www.example.com/url"


def test_final_url_falls_back_to_request_url() -> None:
    result, _ = run(envelope())
    assert result.final_url == TARGET


def test_content_type_from_metadata() -> None:
    result, _ = run(envelope(content_type="application/xhtml+xml"))
    assert result.content_type == "application/xhtml+xml"


def test_content_type_default() -> None:
    result, _ = run(envelope())
    assert result.content_type == "text/html; charset=utf-8"


def test_status_from_metadata_status_code() -> None:
    result, _ = run(envelope(status_code=404), status=200)
    assert result.status == 404


def test_missing_status_code_falls_back_to_transport_status() -> None:
    result, _ = run(envelope(include_status_code=False), status=202)
    assert result.status == 202


def test_sha256_is_hash_of_raw_html_bytes() -> None:
    result, _ = run(envelope())
    assert result.sha256 == hashlib.sha256(RAW_HTML.encode("utf-8")).hexdigest()


def test_render_js_does_not_change_the_request() -> None:
    transport = FakeTransport()
    fetcher = FirecrawlFetcher(BASE, transport)
    fetcher.fetch(TARGET)
    fetcher.fetch(TARGET, render_js=False)
    fetcher.fetch(TARGET, render_js=True)
    bodies = {json.dumps(json.loads(call[1]), sort_keys=True) for call in transport.calls}
    assert len(bodies) == 1
    assert len(transport.calls) == 3


def test_empty_raw_html_is_accepted_as_empty_body() -> None:
    result, _ = run(envelope(""))
    assert result.body_bytes == b""
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize(
    "url", ["ftp://example.com/", "file:///etc/passwd", "example.com", "", "http://"]
)
def test_non_http_url_rejected(url: str) -> None:
    transport = FakeTransport()
    with pytest.raises(FetchError, match="http"):
        FirecrawlFetcher(BASE, transport).fetch(url)
    assert transport.calls == []


@pytest.mark.parametrize("status", [301, 400, 404, 500, 503])
def test_non_2xx_transport_status_rejected(status: int) -> None:
    with pytest.raises(FetchError, match=str(status)):
        run(status=status)


def test_success_false_includes_error_text() -> None:
    body = json.dumps({"success": False, "error": "site blocked scraping"}).encode()
    with pytest.raises(FetchError, match="site blocked scraping"):
        run(body)


def test_success_false_error_text_is_truncated() -> None:
    long_error = "x" * 5000
    with pytest.raises(FetchError) as info:
        run(json.dumps({"success": False, "error": long_error}).encode())
    message = str(info.value)
    assert long_error not in message
    assert len(message) < 1000
    assert "x" * 100 in message


def test_success_false_non_string_error_still_fails() -> None:
    body = json.dumps({"success": False, "error": {"code": 1}}).encode()
    with pytest.raises(FetchError):
        run(body)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b"{}",
        b'{"success": true}',
        b'{"success": true, "data": []}',
        b'{"success": "yes", "data": {}}',
        b'{"success": false}',
    ],
)
def test_malformed_response_rejected(body: bytes) -> None:
    with pytest.raises(FetchError):
        run(body)


@pytest.mark.parametrize(
    "body",
    [
        envelope(include_raw_html=False),
        envelope(raw_html=None),
        envelope(raw_html=42),
    ],
)
def test_missing_raw_html_rejected(body: bytes) -> None:
    with pytest.raises(FetchError, match="rawHtml"):
        run(body)


def test_lone_surrogate_raw_html_rejected_without_content() -> None:
    # json.dumps escapes the lone surrogate, so the response bytes are valid JSON
    # and json.loads hands back a str that cannot be encoded as UTF-8.
    body = envelope(raw_html="<p>\ud800</p>")
    with pytest.raises(FetchError) as info:
        run(body)
    message = str(info.value)
    assert "rawHtml" in message
    assert "<p>" not in message
    assert "\ud800" not in message


def test_deeply_nested_json_rejected_without_content() -> None:
    depth = 100_000
    body = b"[" * depth + b"]" * depth
    with pytest.raises(FetchError) as info:
        run(body)
    message = str(info.value)
    assert len(message) < 200
    assert "nested" in message
    assert "[" not in message


def test_response_bytes_over_cap_rejected() -> None:
    with pytest.raises(FetchError, match="response"):
        run(b" " * 200_000, max_bytes=1)


def test_raw_html_over_max_bytes_rejected() -> None:
    with pytest.raises(FetchError, match="max_bytes"):
        run(envelope(), max_bytes=len(RAW_HTML.encode()) - 1)


def test_raw_html_at_max_bytes_accepted() -> None:
    result, _ = run(envelope(), max_bytes=len(RAW_HTML.encode()))
    assert result.body_bytes == RAW_HTML.encode()


def test_transport_exception_is_wrapped_without_message() -> None:
    def boom(url: str, body: bytes, headers: Mapping[str, str]) -> tuple[int, bytes]:
        raise OSError("secret https://internal.example detail")

    with pytest.raises(FetchError, match="OSError") as info:
        FirecrawlFetcher(BASE, boom).fetch(TARGET)
    assert "secret" not in str(info.value)


def test_from_env_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_URL", raising=False)
    with pytest.raises(ValueError, match="FIRECRAWL_URL"):
        FirecrawlFetcher.from_env(FakeTransport())


def test_from_env_blank_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_URL", "   ")
    with pytest.raises(ValueError, match="FIRECRAWL_URL"):
        FirecrawlFetcher.from_env(FakeTransport())


def test_from_env_uses_environment_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_URL", BASE)
    transport = FakeTransport()
    FirecrawlFetcher.from_env(transport).fetch(TARGET)
    assert transport.calls[0][0] == BASE + "/v1/scrape"


def test_from_env_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_URL", BASE + "/")
    transport = FakeTransport()
    FirecrawlFetcher.from_env(transport).fetch(TARGET)
    assert transport.calls[0][0] == BASE + "/v1/scrape"


@pytest.mark.parametrize("base", ["", "firecrawl.invalid", "ftp://firecrawl.invalid", 5])
def test_base_url_validation(base: Any) -> None:
    with pytest.raises(ValueError):
        FirecrawlFetcher(base, FakeTransport())


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5, "10"])
def test_max_bytes_validation(max_bytes: Any) -> None:
    with pytest.raises(ValueError):
        FirecrawlFetcher(BASE, FakeTransport(), max_bytes=max_bytes)


@pytest.mark.parametrize("timeout_ms", [0, -1, True, 1.5, "1000"])
def test_timeout_validation(timeout_ms: Any) -> None:
    with pytest.raises(ValueError):
        FirecrawlFetcher(BASE, FakeTransport(), timeout_ms=timeout_ms)


def test_default_max_bytes_is_the_fetcher_default() -> None:
    assert DEFAULT_MAX_BYTES > 0


def test_repr_hides_base_url() -> None:
    assert "firecrawl.invalid" not in repr(FirecrawlFetcher(BASE, FakeTransport()))


def test_no_network_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    result, _ = run()
    assert result.status == 200
