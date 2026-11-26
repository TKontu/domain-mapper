"""Static mosaic page (FE-01): one self-contained HTML file from scores and export JSON Lines.

Rows are entities (from ``scores-export`` lines), columns are rubric criteria grouped under their
topic in rubric order. Each topic header cell shows the entity's topic score and coverage; each
criterion cell is coloured and labelled from the rubric ``status_scale`` and opens (``<details>``,
no JavaScript) to status, confidence, rationale and evidence quotes with url and fetched_at.
All JSONL text is HTML-escaped. No external assets; the only ``http(s)`` references are evidence
links. Export lines are folded into a dict keyed ``(entity_id, criterion_id)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
) -> str:
    """Return the mosaic page as one HTML string; production caller is ``landscape mosaic``."""
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
        out.append(f'<tr data-entity="{_e(entity)}"><th>{_e(entity)}</th>')
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
