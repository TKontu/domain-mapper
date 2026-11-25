#!/usr/bin/env python3
"""Link private scope checkouts (D-003) from a source checkout into a lane worktree.

Every directory ``<source>/scopes/<name>/`` that git does not track (so not ``_example``) is
symlinked to ``<worktree>/scopes/<name>``; nothing is copied. ``scopes/*`` is gitignored, so the
links stay untracked in the worktree. One line per scope name is printed:

- ``linked`` / ``would link``: created (or planned with ``--dry-run``);
- ``ok``: the correct link already exists and is left alone;
- ``conflict``: a different path exists there; it is not touched and the exit code is 1;
- with ``--unlink``: ``unlinked`` / ``would unlink`` for links this tool would create,
  ``kept (not our link)`` for anything else, ``absent`` when nothing is there.

``--source`` defaults to the main checkout of the worktree's repository. Caller: the planner
when provisioning a lane worktree (docs/items/LW-01.md). Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXCLUDED = {"_example"}


def main_checkout(worktree: Path) -> Path:
    common = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(common).parent


def tracked(source: Path, name: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(source), "ls-files", "--", f"scopes/{name}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(out.strip())


def private_scopes(source: Path) -> list[str]:
    root = source / "scopes"
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.is_symlink() and p.name not in EXCLUDED and not tracked(source, p.name)
    )


def is_our_link(dest: Path, src: Path) -> bool:
    return dest.is_symlink() and Path(os.path.realpath(dest)) == Path(os.path.realpath(src))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worktree", required=True, type=Path)
    ap.add_argument("--source", type=Path, help="checkout holding scopes/<name>/ (default: main)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    ap.add_argument("--unlink", action="store_true", help="remove links this tool would create")
    args = ap.parse_args(argv)

    worktree = args.worktree.resolve()
    source = (args.source or main_checkout(worktree)).resolve()
    if source == worktree:
        print("link_scopes: source and worktree are the same checkout", file=sys.stderr)
        return 2

    rc = 0
    for name in private_scopes(source):
        src = source / "scopes" / name
        dest = worktree / "scopes" / name
        if args.unlink:
            if is_our_link(dest, src):
                if not args.dry_run:
                    dest.unlink()
                print(f"{name}: {'would unlink' if args.dry_run else 'unlinked'}")
            elif dest.exists() or dest.is_symlink():
                print(f"{name}: kept (not our link)")
            else:
                print(f"{name}: absent")
            continue
        if is_our_link(dest, src):
            print(f"{name}: ok")
        elif dest.exists() or dest.is_symlink():
            print(f"{name}: conflict ({dest} exists and is not a link to {src})")
            print(f"link_scopes: refusing to overwrite {dest}", file=sys.stderr)
            rc = 1
        else:
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(src, target_is_directory=True)
            print(f"{name}: {'would link' if args.dry_run else 'linked'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
