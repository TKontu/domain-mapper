"""GV-02: shrink-only module reachability ratchet."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_reachability_ratchet.py"
FIXTURE = ROOT / "tests" / "fixtures" / "reachability_ratchet.toml"
SEED_REASON = "unwired at seed (GV-02)"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


class SyntheticTree:
    def __init__(self, files: dict[str, str], allowlist: str = "") -> None:
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

    def print_(self) -> subprocess.CompletedProcess[str]:
        return _run("--root", str(self.root), "--allowlist", str(self.allow), "--print")

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
        # Negative control: with no public symbols there is nothing to ratchet.
        t = self._tree({"core/a.py": "import os\n", "adapters/b.py": "import json\n"})
        self.assertEqual(t.check().returncode, 0, t.check().stdout)

    def test_new_unreferenced_symbol_fails(self) -> None:
        t = self._tree({"core/a.py": "def lonely():\n    return 1\n"})
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("lonely", r.stdout)

    def test_unreferenced_class_and_assign_fail(self) -> None:
        t = self._tree(
            {
                "core/a.py": """
                class Widget:
                    pass

                LIMIT = 3
                COUNT: int = 2
                """,
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        for name in ("Widget", "LIMIT", "COUNT"):
            self.assertIn(name, r.stdout)

    def test_test_only_reference_fails(self) -> None:
        t = self._tree(
            {
                "core/a.py": "def helper():\n    return 1\n",
                "tests/test_a.py": "from core.a import helper\n\nhelper()\n",
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("helper", r.stdout)

    def test_production_reference_passes(self) -> None:
        t = self._tree(
            {
                "core/a.py": "def helper():\n    return 1\n",
                "adapters/b.py": "from core.a import helper\n\nhelper()\n",
            }
        )
        self.assertEqual(t.check().returncode, 0, t.check().stdout)

    def test_same_named_reference_counts(self) -> None:
        # Deliberately coarse: a same-named symbol elsewhere is a caller.
        t = self._tree(
            {
                "core/a.py": "def shared():\n    return 1\n",
                "core/b.py": "shared()\n",
            }
        )
        self.assertEqual(t.check().returncode, 0, t.check().stdout)

    def test_stale_entry_for_referenced_symbol_fails(self) -> None:
        allow = """
        "core/a.py::helper" = "wired elsewhere"
        """
        t = self._tree(
            {
                "core/a.py": "def helper():\n    return 1\n",
                "core/b.py": "from core.a import helper\n\nhelper()\n",
            },
            allow,
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("stale", r.stdout)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("helper", r.stdout)

    def test_stale_entry_for_missing_symbol_fails(self) -> None:
        allow = """
        "core/a.py::gone" = "wired elsewhere"
        """
        t = self._tree({"core/a.py": "import os\n"}, allow)
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("stale", r.stdout)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("gone", r.stdout)

    def test_private_names_ignored(self) -> None:
        t = self._tree({"core/a.py": "def _hidden():\n    return 1\n\n_TABLE = 2\n"})
        self.assertEqual(t.check().returncode, 0, t.check().stdout)

    def test_dunder_all_alone_does_not_count(self) -> None:
        t = self._tree({"core/a.py": '__all__ = ["exported"]\n\ndef exported():\n    return 1\n'})
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exported", r.stdout)

    def test_project_script_entry_point_counts(self) -> None:
        t = self._tree(
            {
                "pyproject.toml": '[project.scripts]\nentry = "app.cli:main"\n',
                "app/cli.py": "def main():\n    return 0\n",
            }
        )
        self.assertEqual(t.check().returncode, 0, t.check().stdout)

    def test_call_decorator_rooted_at_production_name_counts_as_reference(self) -> None:
        # Framework registration (e.g. @app.command(...)) is a production caller.
        t = self._tree(
            {
                "app/cli.py": (
                    'app = Typer()\n\n@app.command("run")\ndef run() -> int:\n    return 1\n'
                ),
            }
        )
        r = t.check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_attribute_decorator_rooted_at_production_name_counts_as_reference(self) -> None:
        t = self._tree(
            {
                "app/cli.py": ("app = Typer()\n\n@app.command\ndef run() -> int:\n    return 1\n"),
            }
        )
        r = t.check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_registered_class_counts_as_reference(self) -> None:
        t = self._tree(
            {
                "app/cli.py": ('app = Typer()\n\n@app.command("run")\nclass Runner:\n    pass\n'),
            }
        )
        r = t.check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_dataclass_decorator_alone_does_not_count_as_reference(self) -> None:
        # Bare-name pure wrappers are not production registration.
        t = self._tree(
            {
                "core/a.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass\n"
                    "class Widget:\n"
                    "    x: int = 1\n"
                ),
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("core/a.py", r.stdout)
        self.assertIn("Widget", r.stdout)

    def test_dataclass_call_decorator_alone_does_not_count_as_reference(self) -> None:
        t = self._tree(
            {
                "core/a.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class Widget:\n"
                    "    x: int = 1\n"
                ),
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Widget", r.stdout)

    def test_stdlib_attribute_decorator_does_not_count_as_reference(self) -> None:
        t = self._tree(
            {
                "core/a.py": (
                    "import functools\n\n@functools.cache\ndef compute() -> int:\n    return 1\n"
                ),
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("compute", r.stdout)

    def test_bare_name_decorator_does_not_count_as_reference(self) -> None:
        # Even a production name does not count unless the expression is a call or attribute.
        t = self._tree(
            {
                "core/a.py": (
                    "def register(fn):\n"
                    "    return fn\n\n"
                    "@register\n"
                    "def handler() -> int:\n"
                    "    return 1\n"
                ),
            }
        )
        r = t.check()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("handler", r.stdout)

    def test_print_emits_seeded_toml(self) -> None:
        t = self._tree({"core/a.py": "def lonely():\n    return 1\n"})
        r = t.print_()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(tomllib.loads(r.stdout), {"core/a.py::lonely": SEED_REASON})


if __name__ == "__main__":
    unittest.main()
