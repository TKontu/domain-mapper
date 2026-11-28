#!/usr/bin/env python3
"""Merge one reviewed PR under the standing delegation D-002 (docs/decisions.md).

Usage: python3 scripts/merge_guard.py PR --head FULL_REVIEWED_SHA

The only merge path for planners: raw `gh pr merge` stays denied in .claude/settings.json.
It merges only when the PR is open, not a draft, mergeable, targets the default branch, still
has exactly the reviewed head, and touches no operator-owned path. GitHub re-checks the head
at merge time (--match-head-commit), and the merge is confirmed by re-reading the PR.
Local tests decide (MG-01): it also refuses unless the round ledger (.agent-runs/, written by
scripts/round.py) holds a result for exactly the reviewed head whose checks all passed and together
cover pytest, ruff check, ruff format --check and mypy. A CI check that concluded failed also
refuses; pending, in-progress or absent CI does not block. Review itself stays the planner's
/review-round duty; this script cannot see it, nor whether a PR decides an [H] item.
Records-only PRs (MG-02): when every changed file is a reconciliation record (RECORD_PATHS), a
passing complete result stored by `scripts/round.py reconcile-result` for exactly that head, in
any round, also satisfies the local-gate rule. Any other file falls back to the ledger rule.
"""

import argparse
import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
# D-002: these stay the operator's to merge.
OPERATOR_PATHS = (
    "docs/decisions.md",
    "ARCHITECTURE.md",
    ".agent/config.toml",
    ".claude/*",
    "scripts/merge_guard.py",
)
# MG-02: reconciliation records; a PR touching only these may merge on reconcile evidence.
RECORD_PATHS = ("todo.md", "docs/items/*", "docs/runs/*", "docs/reconcile*")
LEDGER = ROOT / ".agent-runs"
# Gate components a recorded result must cover, matched on the joined command text.
GATE_COMPONENTS = ("pytest", "ruff check", "ruff format --check", "mypy")
FAILED_CI = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}
Run = Callable[[Sequence[str]], tuple[int, str]]


def _run(cmd: Sequence[str]) -> tuple[int, str]:
    done = subprocess.run(list(cmd), capture_output=True, text=True, timeout=120, cwd=ROOT)
    return done.returncode, (done.stdout + done.stderr).strip()


def default_branch() -> str:
    try:
        with (ROOT / ".agent" / "config.toml").open("rb") as fh:
            return tomllib.load(fh).get("project", {}).get("default_branch", "main")
    except OSError:
        return "main"


def view(pr: int, run: Run) -> dict:
    fields = "state,isDraft,mergeable,headRefOid,baseRefName,files,statusCheckRollup"
    rc, out = run(["gh", "pr", "view", str(pr), "--json", fields])
    if rc:
        raise RuntimeError(f"gh pr view {pr} failed: {out[:300]}")
    return json.loads(out)


def problems(record: dict, head: str, branch: str) -> list[str]:
    found = []
    if record.get("state") != "OPEN":
        found.append(f"state is {record.get('state')}, not OPEN")
    if record.get("isDraft"):
        found.append("PR is a draft")
    if record.get("mergeable") != "MERGEABLE":
        found.append(f"mergeable is {record.get('mergeable')}")
    if record.get("baseRefName") != branch:
        found.append(f"base is {record.get('baseRefName')}, not {branch}")
    if record.get("headRefOid") != head:
        found.append(f"head is {record.get('headRefOid')}, not the reviewed {head}")
    owned = sorted(
        f["path"]
        for f in record.get("files") or []
        if any(fnmatch.fnmatch(f["path"], pattern) for pattern in OPERATOR_PATHS)
    )
    if owned:
        found.append("operator-owned paths (D-002), leave to the operator: " + ", ".join(owned))
    return found


def _round_module():
    spec = importlib.util.spec_from_file_location("round_ledger", ROOT / "scripts" / "round.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger_results(ledger: Path) -> list[dict]:
    """Every result recorded in any round manifest under the ledger root."""
    ledger_io = _round_module()
    found = []
    for manifest_path in sorted(ledger.glob("*/round.json")):
        root = manifest_path.parent
        try:
            manifest = ledger_io.read(manifest_path)
            for assignment in (manifest.get("assignments") or {}).values():
                for relative in assignment.get("results") or []:
                    found.append(ledger_io.read(ledger_io.inside(root, relative)))
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    return found


def _result_gaps(result: dict) -> list[str]:
    checks = result.get("checks") or []
    gaps = [
        f"check {' '.join(c.get('command') or [])!r} is {c.get('status')}"
        for c in checks
        if c.get("status") != "passed"
    ]
    text = " ".join(" ".join(c.get("command") or []) for c in checks)
    text = " ".join(text.split())
    gaps += [f"missing {name}" for name in GATE_COMPONENTS if name not in text]
    return gaps


def _ledger_problems(head: str, ledger: Path) -> list[str]:
    """A passing complete result at the exact head suffices, even if others at it did not pass."""
    at_head = [r for r in _ledger_results(ledger) if r.get("head_sha") == head]
    if not at_head:
        return [f"no local-gate result recorded in {ledger} for head {head}"]
    gaps = [_result_gaps(result) for result in at_head]
    if any(not gap for gap in gaps):
        return []
    return [f"no passing complete local-gate result at {head}: " + "; ".join(gap) for gap in gaps]


def records_only(record: dict) -> bool:
    paths = [f["path"] for f in record.get("files") or []]
    return bool(paths) and all(
        any(fnmatch.fnmatchcase(path, pattern) for pattern in RECORD_PATHS) for path in paths
    )


def _reconcile_passes(head: str, ledger: Path) -> bool:
    """A stored reconcile result for exactly this head, in any round, with a complete gate."""
    for path in sorted(ledger.glob(f"*/reconcile-{head[:12]}.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(result, dict) and result.get("head_sha") == head and not _result_gaps(result):
            return True
    return False


def _ci_problems(record: dict) -> list[str]:
    failed = []
    for entry in record.get("statusCheckRollup") or []:
        if entry.get("__typename") == "StatusContext" or "state" in entry:
            outcome = (entry.get("state") or "").upper()
            name = entry.get("context")
            bad = outcome in {"FAILURE", "ERROR"}
        else:
            outcome = (entry.get("conclusion") or "").upper()
            name = entry.get("name")
            bad = outcome in FAILED_CI
        if bad:
            failed.append(f"CI check {name} concluded {outcome}")
    return failed


def guard(
    pr: int, head: str, *, run: Run = _run, branch: str | None = None, ledger: Path | None = None
) -> int:
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        print("refusing: --head must be the full 40-character reviewed SHA", file=sys.stderr)
        return 2
    branch = branch or default_branch()
    record = view(pr, run)
    found = problems(record, head, branch)
    ledger = ledger or LEDGER
    if not (records_only(record) and _reconcile_passes(head, ledger)):
        found += _ledger_problems(head, ledger)
    found += _ci_problems(record)
    if found:
        for problem in found:
            print(f"refusing PR #{pr}: {problem}", file=sys.stderr)
        return 1
    rc, out = run(["gh", "pr", "merge", str(pr), "--merge", "--match-head-commit", head])
    if rc:
        print(f"gh pr merge {pr} failed: {out[:300]}", file=sys.stderr)
        return 1
    state = view(pr, run).get("state")
    if state != "MERGED":
        print(f"PR #{pr} state is {state} after merge; not merged", file=sys.stderr)
        return 1
    print(f"merged PR #{pr} at reviewed head {head}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr", type=int)
    parser.add_argument("--head", required=True, help="full reviewed head SHA")
    args = parser.parse_args(argv)
    return guard(args.pr, args.head)


if __name__ == "__main__":
    sys.exit(main())
