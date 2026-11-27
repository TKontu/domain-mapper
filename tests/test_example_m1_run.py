"""EX-01: the offline M1 example run over scopes/_example drives the real CLI end to end."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "example_m1_run.py"


def _rows(page: str) -> list[str]:
    return re.findall(r'<tr data-entity="([^"]+)">', page)


def test_example_m1_run_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert all(step["exit_code"] == 0 for step in summary["steps"])
    scores = [
        json.loads(x) for x in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    layers_of = {str(s["entity_id"]): set(s.get("layers") or []) for s in scores}
    assert any(not v for v in layers_of.values()), "fixture needs an entity without presence"

    for layer in ("all", "home", "region", "global"):
        page_path = out / f"mosaic-{layer}.html"
        assert page_path.is_file()
        page = page_path.read_text(encoding="utf-8")
        rows = _rows(page)
        assert rows, layer
        if layer == "all":
            assert sorted(rows) == sorted(layers_of)
        else:
            expected = [e for e, ls in layers_of.items() if layer in ls]
            assert sorted(rows) == sorted(expected), layer
        assert summary["mosaic"][layer]["rows"] == len(rows)
        ent_dir = out / f"entities-{layer}"
        hrefs = re.findall(r'<tr data-entity="[^"]+"><th><a href="([^"]+)">', page)
        assert len(hrefs) == len(rows), layer
        for href in hrefs:
            target = (page_path.parent / href).resolve()
            assert target.parent == ent_dir.resolve() and target.is_file(), href
            ent = target.read_text(encoding="utf-8")
            assert "<ul><li>" in ent, href  # nonempty evidence section
