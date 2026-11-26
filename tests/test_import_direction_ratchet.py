"""GV-01: shrink-only import-direction ratchet (core <-> adapters)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_import_direction_ratchet.py"
FIXTURE = ROOT / "tests" / "fixtures" / "import_direction_ratchet.toml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


class SyntheticTree:
    def __init__(self, files: dict[str, str], allowlist: str) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for rel, src in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src))
        self.allow = self.root / "allow.toml"
        self.allow.write_text(textwrap.dedent(allowlist))

    def check(self) -> subprocess.CompletedProcess[str]:
        return _run("--root", str(self.root), "--allowlist", str(self.allow))

    def close(self) -> None:
        self._tmp.cleanup()


class RealTreeTest(unittest.TestCase):
    def test_tree_matches_allowlist(self) -> None:
        r = _run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_print_round_trips_to_fixture(self) -> None:
        r = _run("--print")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(tomllib.loads(r.stdout), tomllib.loads(FIXTURE.read_text()))


class SyntheticTest(unittest.TestCase):
    def _tree(self, files: dict[str, str], allow: str = "") -> SyntheticTree:
        t = SyntheticTree(files, allow)
        self.addCleanup(t.close)
        return t

    def test_clean_tree_passes(self) -> None:
        t = self._tree({"core/a.py": "import os\n", "adapters/b.py": "import json\n"})
        self.assertEqual(t.check().returncode, 0)

    def test_new_top_level_import_fails(self) -> None:
        t = self._tree({"core/a.py": "from adapters.llm import X\n"})
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("core -> adapters", r.stdout)

    def test_dotted_import_form_fails(self) -> None:
        t = self._tree({"adapters/b.py": "import core.models\n"})
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("adapters/b.py", r.stdout)
        self.assertIn("adapters -> core", r.stdout)

    def test_new_function_local_import_fails(self) -> None:
        allow = """
        ["core/a.py".adapters]
        module = 1
        local = 0
        """
        t = self._tree(
            {"core/a.py": "import adapters\n\ndef f():\n    from adapters import llm\n"}, allow
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("local", r.stdout)

    def test_stale_entry_fails(self) -> None:
        allow = """
        ["core/a.py".adapters]
        module = 2
        local = 0
        """
        t = self._tree({"core/a.py": "import adapters\n"}, allow)
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("stale", r.stdout)
        self.assertIn("core -> adapters", r.stdout)

    def test_stale_entry_for_removed_file_fails(self) -> None:
        allow = """
        ["core/gone.py".adapters]
        module = 1
        local = 0
        """
        r = self._tree({"core/a.py": "import os\n"}, allow).check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/gone.py", r.stdout)

    def test_relative_and_lookalike_imports_ignored(self) -> None:
        t = self._tree(
            {"core/a.py": "from . import adapters\nimport adapters_extra\nimport coreutils\n"}
        )
        self.assertEqual(t.check().returncode, 0)


if __name__ == "__main__":
    unittest.main()
