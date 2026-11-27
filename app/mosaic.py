"""Static mosaic page (FE-01): one self-contained HTML file from scores and export JSON Lines.

Rows are entities (from ``scores-export`` lines), columns are rubric criteria grouped under their
topic in rubric order. Each topic header cell shows the entity's topic score and coverage; each
criterion cell is coloured and labelled from the rubric ``status_scale`` and opens (``<details>``,
no JavaScript) to status, confidence, rationale and evidence quotes with url and fetched_at.
All JSONL text is HTML-escaped. No external assets; the only ``http(s)`` references are evidence
links. Export lines are folded into a dict keyed ``(entity_id, criterion_id)``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from html import escape
from typing import Any

from core.scope_pack import ScopePack

_STYLE = """
body{font-family:sans-serif;font-size:13px;margin:1em}
table{border-collapse:collapse}
th,td{border:1px solid #ccc;padding:2px 4px;vertical-align:top}
th.topic{background:#eee}
td.topic{background:#f6f6f6;white-space:nowrap}
td.cell{min-width:5em}
td.cell details summary{cursor:pointer;list-style:none}
.hatched{background:repeating-linear-gradient(45deg,#ddd 0 4px,#fff 4px 8px)}
.pop{max-width:30em;white-space:normal}
"""


def _e(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _ratio(value: object) -> str:
    return "-" if not isinstance(value, int | float) else f"{float(value):.2f}"


def _link(url: object) -> str:
    text = _e(url)
    if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
        return f'<a href="{text}" rel="noopener noreferrer">{text}</a>'
    return text


def _cell(row: Mapping[str, Any] | None, scale: Mapping[str, tuple[str, str]]) -> str:
    if row is None:
        return '<td class="cell empty"></td>'
    status = str(row.get("status"))
    label, color = scale.get(status, (status, ""))
    if color == "hatched":
        attrs = ' class="cell hatched"'
    else:
        attrs = f' class="cell" style="background:{_e(color)}"' if color else ' class="cell"'
    parts = [
        f"<p>status: {_e(label)} ({_e(status)})</p>",
        f"<p>confidence: {_e(row.get('confidence'))}</p>",
        f"<p>rationale: {_e(row.get('rationale'))}</p>",
    ]
    evidence = row.get("evidence") or []
    if evidence:
        parts.append("<ul>")
        for ev in evidence:
            parts.append(
                f"<li><q>{_e(ev.get('quote'))}</q><br>{_link(ev.get('url'))}"
                f" <span>fetched {_e(ev.get('fetched_at'))}</span></li>"
            )
        parts.append("</ul>")
    return (
        f'<td{attrs} data-status="{_e(status)}" data-color="{_e(color)}">'
        f'<details><summary>{_e(label)}</summary><div class="pop">{"".join(parts)}</div>'
        "</details></td>"
    )


def render_mosaic(
    pack: ScopePack,
    scores_lines: Iterable[Mapping[str, Any]],
    export_lines: Iterable[Mapping[str, Any]],
    entity_href: Callable[[str], str] | None = None,
    heading: str | None = None,
) -> str:
    """Return the mosaic page as one HTML string; production caller is ``landscape mosaic``.

    ``entity_href`` (FE-02) maps an entity id to a relative link for its row header; without it
    the output is unchanged from FE-01. ``heading`` (FE-03) adds an escaped ``<h2>`` line, e.g.
    the active segment filter; ``None`` leaves the output unchanged.
    """
    scale = {s.id: (s.label, s.color) for s in pack.rubric.status_scale}
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in export_lines:
        cells[(str(row.get("entity_id")), str(row.get("criterion_id")))] = row
    topics = pack.rubric.topics
    out = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        f"<title>Mosaic {_e(pack.scope.id)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>{_e(pack.scope.id)} {_e(pack.scope.version)}</h1>",
        *([f'<h2 class="filter">{_e(heading)}</h2>'] if heading is not None else []),
        '<table><thead><tr><th rowspan="2">entity</th>',
    ]
    for topic in topics:
        out.append(
            f'<th class="topic" colspan="{len(topic.criteria) + 1}"'
            f' data-topic="{_e(topic.id)}">{_e(topic.label)}</th>'
        )
    out.append("</tr><tr>")
    for topic in topics:
        out.append('<th class="topic">score / coverage</th>')
        for c in topic.criteria:
            out.append(f'<th data-criterion="{_e(c.id)}" title="{_e(c.question)}">{_e(c.id)}</th>')
    out.append("</tr></thead><tbody>")
    for line in scores_lines:
        entity = str(line.get("entity_id"))
        by_topic = {str(t.get("topic_id")): t for t in line.get("topics") or []}
        head = _e(entity)
        if entity_href is not None:
            head = f'<a href="{_e(entity_href(entity))}">{head}</a>'
        out.append(f'<tr data-entity="{_e(entity)}"><th>{head}</th>')
        for topic in topics:
            ts = by_topic.get(topic.id, {})
            out.append(
                f'<td class="topic">{_ratio(ts.get("score"))} / {_ratio(ts.get("coverage"))}</td>'
            )
            for c in topic.criteria:
                out.append(_cell(cells.get((entity, c.id)), scale))
        out.append("</tr>")
    out.append("</tbody></table></body></html>\n")
    return "\n".join(out)


def filter_by_segments(
    scores_lines: Iterable[Mapping[str, Any]], segment_ids: Collection[str]
) -> Iterator[Mapping[str, Any]]:
    """Yield, in input order, scores lines whose ``segments`` list contains any of ``segment_ids``.

    Streaming (FE-03); a line without a ``segments`` list never matches.
    """
    wanted = set(segment_ids)
    for line in scores_lines:
        segments = line.get("segments")
        if isinstance(segments, list) and wanted.intersection(str(s) for s in segments):
            yield line


def filter_by_layers(
    scores_lines: Iterable[Mapping[str, Any]], layer_ids: Collection[str]
) -> Iterator[Mapping[str, Any]]:
    """Yield, in input order, scores lines whose ``layers`` list contains any of ``layer_ids``.

    Streaming (FE-04); a line without a ``layers`` list never matches.
    """
    wanted = set(layer_ids)
    for line in scores_lines:
        layers = line.get("layers")
        if isinstance(layers, list) and wanted.intersection(str(la) for la in layers):
            yield line


_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def entity_slug(entity_id: str) -> str:
    """Return a filename-safe, collision-resistant slug: ``[A-Za-z0-9_-]`` plus a short digest."""
    digest = hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:8]
    return f"{_UNSAFE.sub('_', entity_id)[:64]}-{digest}"


def render_entity_page(
    pack: ScopePack,
    scores_line: Mapping[str, Any],
    cells_for_entity: Mapping[str, Mapping[str, Any]],
    mosaic_href: str,
) -> str:
    """Return one entity's page (FE-02): topic score/coverage, then every criterion in rubric order.

    ``cells_for_entity`` maps criterion id to its export line; a criterion without one is shown as
    a "not assessed" row. ``mosaic_href`` is the relative link back to the mosaic page.
    """
    scale = {s.id: (s.label, s.color) for s in pack.rubric.status_scale}
    entity = str(scores_line.get("entity_id"))
    by_topic = {str(t.get("topic_id")): t for t in scores_line.get("topics") or []}
    out = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        f"<title>Entity {_e(entity)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f'<p><a href="{_e(mosaic_href)}">back to mosaic</a></p>',
        f'<h1 data-entity="{_e(entity)}">{_e(entity)}</h1>',
        "<table><thead><tr><th>topic</th><th>criterion</th><th>status</th><th>confidence</th>"
        "<th>rationale</th><th>evidence</th></tr></thead><tbody>",
    ]
    for topic in pack.rubric.topics:
        ts = by_topic.get(topic.id, {})
        out.append(
            f'<tr data-topic="{_e(topic.id)}"><th class="topic" colspan="6">{_e(topic.label)}: '
            f"{_ratio(ts.get('score'))} / {_ratio(ts.get('coverage'))}</th></tr>"
        )
        for c in topic.criteria:
            row = cells_for_entity.get(c.id)
            first = f'<td>{_e(topic.id)}</td><td title="{_e(c.question)}">{_e(c.id)}</td>'
            if row is None:
                out.append(
                    f'<tr data-criterion="{_e(c.id)}">{first}'
                    '<td class="cell empty">not assessed</td><td></td><td></td><td></td></tr>'
                )
                continue
            status = str(row.get("status"))
            label, color = scale.get(status, (status, ""))
            if color == "hatched":
                attrs = ' class="cell hatched"'
            else:
                attrs = (
                    f' class="cell" style="background:{_e(color)}"' if color else ' class="cell"'
                )
            evidence = "".join(
                f"<li><q>{_e(ev.get('quote'))}</q><br>{_link(ev.get('url'))}"
                f" <span>fetched {_e(ev.get('fetched_at'))}</span></li>"
                for ev in row.get("evidence") or []
            )
            out.append(
                f'<tr data-criterion="{_e(c.id)}">{first}'
                f'<td{attrs} data-status="{_e(status)}" data-color="{_e(color)}">{_e(label)}</td>'
                f"<td>{_e(row.get('confidence'))}</td><td>{_e(row.get('rationale'))}</td>"
                f"<td>{f'<ul>{evidence}</ul>' if evidence else ''}</td></tr>"
            )
    out.append("</tbody></table></body></html>\n")
    return "\n".join(out)
