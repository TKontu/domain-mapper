"""Same-host link extraction from fetched HTML (CR-06); stdlib ``html.parser`` only.

Only ``<a href>`` is read. Each href is resolved against the page's final URL; only ``http``/
``https`` links on exactly ``host`` are kept, with fragment and query dropped, obvious non-HTML
extensions skipped and duplicates removed in document order. At most :data:`MAX_HREFS_PER_PAGE`
hrefs are examined per page.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

MAX_HREFS_PER_PAGE = 500
_SKIP_EXTENSIONS = frozenset(
    {
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".zip", ".gz",
        ".tar", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".mov", ".doc", ".docx", ".xls",
        ".xlsx", ".ppt", ".pptx", ".css", ".js", ".xml", ".json", ".csv", ".exe", ".dmg",
    }
)  # fmt: skip


class _HrefCollector(HTMLParser):
    def __init__(self, cap: int) -> None:
        super().__init__(convert_charrefs=True)
        self.cap = cap
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or len(self.hrefs) >= self.cap:
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)
                return


def normalise_url(url: str) -> str | None:
    """Lower-cased ``scheme://host[:port]/path``, no query or fragment; None if unusable."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in ("http", "https") or not host:
        return None
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", "", ""))


def crawl_key(url: str) -> str | None:
    """Seen-set key for one entity crawl (CR-10): :func:`normalise_url` with the scheme set to
    ``https`` and one trailing ``/`` dropped from a non-root path; None if unusable."""
    norm = normalise_url(url)
    if norm is None:
        return None
    parts = urlsplit(norm)
    path = parts.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit(("https", parts.netloc, path, "", ""))


def extract_links(
    html: str, *, base_url: str, host: str, cap: int = MAX_HREFS_PER_PAGE
) -> list[str]:
    """Same-host page links in ``html`` resolved against ``base_url``, in document order."""
    collector = _HrefCollector(cap)
    try:
        collector.feed(html)
        collector.close()
    except Exception:  # noqa: BLE001 - malformed markup: keep what was collected
        pass
    want = host.lower()
    seen: set[str] = set()
    links: list[str] = []
    for href in collector.hrefs:
        try:
            absolute = urljoin(base_url, href.strip())
        except ValueError:
            continue
        url = normalise_url(absolute)
        if url is None or urlsplit(url).hostname != want:
            continue
        last = urlsplit(url).path.rsplit("/", 1)[-1].lower()
        if "." in last and last[last.rindex(".") :] in _SKIP_EXTENSIONS:
            continue
        if url not in seen:
            seen.add(url)
            links.append(url)
    return links
