"""Blind gold-set review pages (GS-03, ARCHITECTURE.md §7).

:func:`write_gold_pages` streams a GS-01 ``cells.jsonl`` and writes a GS-02 server root:
``pages/<cell_id>.html`` (self-contained, inline CSS/JS, no external requests), ``pages.json``
(cell id -> title) and ``INSTRUCTIONS.md``. Only whitelisted blind fields are read from each line
(cell id, primary domain, criterion id/text, evidence chunk text/url/fetched_at); any status,
confidence or assessment field on the line is ignored, and ``cells.key.jsonl`` is never opened.
All text is HTML-escaped and only ``http(s)`` URLs become links. Output is deterministic.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from core.scope_pack import ScopePack

PAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")  # mirrors app.review_server._PAGE_ID


@dataclass(frozen=True)
class _Evidence:
    chunk_text: str
    url: str
    fetched_at: str


@dataclass(frozen=True)
class _BlindCell:
    cell_id: str
    primary_domain: str
    criterion_id: str
    criterion: str
    evidence: tuple[_Evidence, ...]


def _str(obj: dict[str, object], key: str) -> str:
    value = obj.get(key, "")
    return value if isinstance(value, str) else ""


def _read_cells(path: Path) -> Iterator[_BlindCell]:
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            cid = _str(obj, "cell_id")
            if not PAGE_ID.fullmatch(cid):
                raise ValueError(f"{path}:{n}: invalid cell_id {cid!r}")
            raw_ev = obj.get("evidence")
            evidence = tuple(
                _Evidence(_str(e, "chunk_text"), _str(e, "url"), _str(e, "fetched_at"))
                for e in (raw_ev if isinstance(raw_ev, list) else [])
                if isinstance(e, dict)
            )
            yield _BlindCell(
                cid,
                _str(obj, "primary_domain"),
                _str(obj, "criterion_id"),
                _str(obj, "criterion"),
                evidence,
            )


def _safe_href(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() in ("http", "https") and parts.netloc:
        return url
    return None


_CSS = """
body{font-family:system-ui,sans-serif;max-width:60rem;margin:1rem auto;padding:0 1rem;}
.crit{font-size:1.2rem;font-weight:600;}
.ev{border:1px solid #ccc;padding:.5rem;margin:.5rem 0;white-space:pre-wrap;}
.meta{color:#555;font-size:.85rem;}
textarea{width:100%;min-height:4rem;}
#state{font-weight:600;margin-left:1rem;}
kbd{border:1px solid #999;padding:0 .2rem;border-radius:3px;}
"""

_JS = """
const CELL = __CELL__;
const radios = Array.from(document.querySelectorAll('input[name=status]'));
const state = document.getElementById('state');
function submitAnswer() {
  const sel = document.querySelector('input[name=status]:checked');
  if (!sel) { state.textContent = 'pick a status first'; return; }
  const status = sel.value;
  const quote = document.getElementById('quote').value;
  const note = document.getElementById('note').value;
  state.textContent = 'saving...';
  fetch('/api/answer/' + CELL, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cell_id: CELL, status: status, quote: quote, note: note})
  }).then(function (r) {
    state.textContent = r.ok ? 'saved' : 'failed (' + r.status + ')';
  }).catch(function () { state.textContent = 'failed (network)'; });
}
document.getElementById('form').addEventListener('submit', function (e) {
  e.preventDefault(); submitAnswer();
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submitAnswer(); return; }
  const t = e.target;
  if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT') && t.type !== 'radio') return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= radios.length) { radios[n - 1].checked = true; radios[n - 1].focus(); return; }
  if (e.key === 'q') {
    const s = String(window.getSelection() || '').trim();
    if (s) { e.preventDefault(); document.getElementById('quote').value = s; }
  }
});
"""


def _page(cell: _BlindCell, pack: ScopePack, next_id: str | None) -> str:
    e = escape
    scale = "\n".join(
        f'<label><input type="radio" name="status" value="{e(s.id)}"> '
        f"<kbd>{i}</kbd> {e(s.label)} <span class=meta>({e(s.id)})</span></label><br>"
        for i, s in enumerate(pack.rubric.status_scale, 1)
    )
    evidence = []
    for i, ev in enumerate(cell.evidence, 1):
        href = _safe_href(ev.url)
        link = f'<a href="{e(href)}" rel="noreferrer">{e(ev.url)}</a>' if href else e(ev.url)
        evidence.append(
            f'<div class="meta">[{i}] {link} &middot; snapshot {e(ev.fetched_at)}</div>'
            f'<div class="ev">{e(ev.chunk_text)}</div>'
        )
    ev_html = "\n".join(evidence) or "<p><em>No evidence frozen for this cell.</em></p>"
    nxt = (
        f'<a href="/p/{e(next_id)}">Next page &rarr;</a>'
        if next_id
        else '<a href="/">Last page &mdash; back to index</a>'
    )
    js = _JS.replace("__CELL__", json.dumps(cell.cell_id))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{e(cell.cell_id)}</title>
<style>{_CSS}</style></head><body>
<p class="meta">Cell {e(cell.cell_id)} &middot; {e(cell.primary_domain)} &middot; \
{e(cell.criterion_id)}</p>
<p class="crit">{e(cell.criterion)}</p>
<h2>Evidence</h2>
{ev_html}
<h2>Your answer</h2>
<form id="form">
<fieldset><legend>Status (keys 1&ndash;{len(pack.rubric.status_scale)})</legend>
{scale}
</fieldset>
<p><label for="quote">Supporting quote (select evidence text, press <kbd>q</kbd>)</label><br>
<textarea id="quote"></textarea></p>
<p><label for="note">Note</label><br><textarea id="note"></textarea></p>
<p><button type="submit">Submit (<kbd>Ctrl</kbd>+<kbd>Enter</kbd>)</button>\
<span id="state"></span></p>
</form>
<p>{nxt}</p>
<script>{js}</script>
</body></html>
"""


INSTRUCTIONS = """# Gold-set labelling instructions (blind protocol)

1. Open the index page and work through the cells in order; each page links to the next.
2. Each page shows one criterion, the rubric's status scale and the frozen evidence (chunk
   text, source URL and snapshot date). Judge **only** from that evidence; do not browse the
   live site and do not consult model output, `cells.key.jsonl` or other labellers.
3. Pick a status (number keys select it). If the evidence does not address the criterion,
   choose the "not stated" status; if the criterion does not apply, choose "does not apply".
4. Copy the sentence that supports your choice into the quote box (select it and press `q`),
   and add a short note when the call is borderline.
5. Submit with the button or Ctrl+Enter and wait for "saved". Re-submitting overwrites the
   earlier answer for that cell (every submission is kept in history).
6. Do not discuss individual cells with anyone until labelling is closed.
"""


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_gold_pages(pack: ScopePack, cells_path: Path, root: Path) -> dict[str, str]:
    """Write pages, ``pages.json`` and ``INSTRUCTIONS.md`` under ``root``; return the registry.

    Cells stream line by line; one page is held back so each page can link to the next.
    Raises ``ValueError`` on a malformed line, an invalid cell id or a duplicate id.
    """
    pages_dir = root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, str] = {}
    prev: _BlindCell | None = None
    for cell in _read_cells(cells_path):
        if cell.cell_id in registry or (prev and prev.cell_id == cell.cell_id):
            raise ValueError(f"duplicate cell_id {cell.cell_id!r}")
        if prev is not None:
            _atomic_write(pages_dir / f"{prev.cell_id}.html", _page(prev, pack, cell.cell_id))
            registry[prev.cell_id] = f"{prev.primary_domain} / {prev.criterion_id}"
        prev = cell
    if prev is not None:
        _atomic_write(pages_dir / f"{prev.cell_id}.html", _page(prev, pack, None))
        registry[prev.cell_id] = f"{prev.primary_domain} / {prev.criterion_id}"
    _atomic_write(root / "INSTRUCTIONS.md", INSTRUCTIONS)
    _atomic_write(root / "pages.json", json.dumps(registry, indent=2, sort_keys=False) + "\n")
    return registry
