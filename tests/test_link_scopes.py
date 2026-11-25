"""LW-01: link private scope checkouts from a source checkout into a lane worktree."""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("link_scopes", ROOT / "scripts" / "link_scopes.py")
link_scopes = importlib.util.module_from_spec(_spec)
sys.modules["link_scopes"] = link_scopes
_spec.loader.exec_module(link_scopes)


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


class LinkScopesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.source = base / "main"
        self.worktree = base / "wt"
        scopes = self.source / "scopes"
        for name in ("alpha", "beta", "_example", "tracked"):
            (scopes / name).mkdir(parents=True)
            (scopes / name / "scope.yaml").write_text("name: x\n")
        (scopes / "README.md").write_text("readme\n")
        (scopes / "loose.txt").write_text("not a dir\n")
        _git(self.source, "init", "-q")
        _git(self.source, "add", "scopes/README.md", "scopes/_example", "scopes/tracked")
        _git(self.source, "commit", "-q", "-m", "init")
        self.worktree.mkdir()

    def run_cli(self, *extra):
        out = io.StringIO()
        err = io.StringIO()
        argv = ["--worktree", str(self.worktree), "--source", str(self.source), *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = link_scopes.main(argv)
        return rc, out.getvalue().splitlines(), err.getvalue()

    def target(self, name):
        return self.worktree / "scopes" / name

    def test_links_only_untracked_scope_directories(self):
        rc, lines, _ = self.run_cli()
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha: linked", "beta: linked"])
        for name in ("alpha", "beta"):
            self.assertTrue(self.target(name).is_symlink())
            self.assertEqual(self.target(name).resolve(), (self.source / "scopes" / name).resolve())
        for name in ("_example", "tracked", "loose.txt", "README.md"):
            self.assertFalse(self.target(name).is_symlink(), name)

    def test_rerun_is_idempotent(self):
        self.run_cli()
        before = self.target("alpha").lstat().st_ino
        rc, lines, _ = self.run_cli()
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha: ok", "beta: ok"])
        self.assertEqual(self.target("alpha").lstat().st_ino, before)

    def test_refuses_to_overwrite_non_link(self):
        (self.worktree / "scopes" / "alpha").mkdir(parents=True)
        rc, lines, err = self.run_cli()
        self.assertNotEqual(rc, 0)
        self.assertIn("alpha: conflict", lines[0])
        self.assertIn("alpha", err)
        self.assertTrue(self.target("alpha").is_dir())
        self.assertFalse(self.target("alpha").is_symlink())
        self.assertTrue(self.target("beta").is_symlink())

    def test_refuses_foreign_link(self):
        (self.worktree / "scopes").mkdir()
        self.target("alpha").symlink_to(self.worktree)
        rc, lines, _ = self.run_cli()
        self.assertNotEqual(rc, 0)
        self.assertIn("alpha: conflict", lines[0])
        self.assertEqual(self.target("alpha").resolve(), self.worktree.resolve())

    def test_dry_run_changes_nothing(self):
        rc, lines, _ = self.run_cli("--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha: would link", "beta: would link"])
        self.assertFalse((self.worktree / "scopes").exists())

    def test_unlink_removes_only_own_links(self):
        self.run_cli()
        (self.worktree / "scopes" / "own").mkdir()
        foreign = self.worktree / "scopes" / "foreign"
        foreign.symlink_to(self.worktree)
        self.target("beta").unlink()
        self.target("beta").mkdir()
        rc, lines, _ = self.run_cli("--unlink")
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha: unlinked", "beta: kept (not our link)"])
        self.assertFalse(self.target("alpha").exists())
        self.assertTrue(self.target("beta").is_dir())
        self.assertTrue(foreign.is_symlink())
        self.assertTrue((self.source / "scopes" / "alpha" / "scope.yaml").is_file())

    def test_unlink_dry_run(self):
        self.run_cli()
        rc, lines, _ = self.run_cli("--unlink", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha: would unlink", "beta: would unlink"])
        self.assertTrue(self.target("alpha").is_symlink())

    def test_default_source_is_main_checkout_of_worktree(self):
        _git(self.source, "worktree", "add", "-q", str(self.worktree / "lane"), "-b", "lane")
        lane = self.worktree / "lane"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = link_scopes.main(["--worktree", str(lane)])
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().splitlines(), ["alpha: linked", "beta: linked"])
        self.assertTrue((lane / "scopes" / "alpha").is_symlink())


class NotACheckoutTest(unittest.TestCase):
    """LW-02: a missing or non-git directory gives one stderr line and exit 2, no traceback."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)

    def run_script(self, *argv):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "link_scopes.py"), *argv],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stderr

    def assert_one_line(self, rc, err, expected):
        self.assertEqual(rc, 2)
        self.assertNotIn("Traceback", err)
        self.assertEqual(err.splitlines(), [expected])

    def test_nonexistent_worktree(self):
        missing = self.base / "missing"
        rc, err = self.run_script("--worktree", str(missing))
        self.assert_one_line(rc, err, f"link_scopes: {missing} does not exist")

    def test_plain_directory_worktree(self):
        plain = self.base / "plain"
        plain.mkdir()
        rc, err = self.run_script("--worktree", str(plain))
        self.assert_one_line(rc, err, f"link_scopes: {plain} is not a git checkout")

    def test_non_git_source(self):
        worktree = self.base / "wt"
        worktree.mkdir()
        source = self.base / "src"
        (source / "scopes" / "alpha").mkdir(parents=True)
        rc, err = self.run_script("--worktree", str(worktree), "--source", str(source))
        self.assert_one_line(rc, err, f"link_scopes: {source} is not a git checkout")

    def test_nonexistent_source(self):
        worktree = self.base / "wt"
        worktree.mkdir()
        source = self.base / "nosrc"
        rc, err = self.run_script("--worktree", str(worktree), "--source", str(source))
        self.assert_one_line(rc, err, f"link_scopes: {source} does not exist")


if __name__ == "__main__":
    unittest.main()
