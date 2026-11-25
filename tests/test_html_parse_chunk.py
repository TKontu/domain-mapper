"""AD-14: `#` lines inside an HTML `pre` stay code through parse_and_chunk."""

from __future__ import annotations

import pytest

from adapters.html_parser import HtmlParser
from core.parse_chunk import parse_and_chunk

URL = "https://example.test/page"


def chunks_of(html: str):
    return list(
        parse_and_chunk(
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
            url=URL,
            parser=HtmlParser(),
        )
    )


@pytest.mark.parametrize(
    "pre_line",
    ["# not a heading", "## x", "###### deep"],
)
def test_hash_line_in_pre_is_not_a_heading(pre_line: str) -> None:
    html = (
        "<html><body><h1>Guide</h1><p>Intro text.</p>"
        f"<pre>setup\n{pre_line}\nmore code</pre>"
        "<h2>Next</h2><p>After.</p></body></html>"
    )
    chunks = chunks_of(html)
    paths = {c.heading_path for c in chunks}
    assert paths == {("Guide",), ("Guide", "Next")}
    guide = [c for c in chunks if c.heading_path == ("Guide",)]
    assert len(guide) == 1
    assert f"\n{pre_line}\n" in guide[0].text
    assert guide[0].text.count("```") == 2


def test_pre_with_backtick_run_keeps_hash_line_in_fence() -> None:
    html = (
        "<html><body><h1>Guide</h1>"
        "<pre>```\n# inside\n````\n## also inside</pre>"
        "<h3>Tail</h3><p>End.</p></body></html>"
    )
    chunks = chunks_of(html)
    assert {c.heading_path for c in chunks} == {("Guide",), ("Guide", "Tail")}
    guide = next(c for c in chunks if c.heading_path == ("Guide",))
    assert "\n# inside\n" in guide.text
    assert "\n## also inside" in guide.text
    assert "`````" in guide.text  # fence longer than the 4-backtick inner run
