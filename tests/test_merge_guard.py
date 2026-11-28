"""The merge guard: the only merge path for unattended planners (decision D-002)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_guard.py"
spec = importlib.util.spec_from_file_location("merge_guard", SCRIPT)
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

HEAD = "a" * 40
GATE = (
    "uv run pytest -q && uv run ruff check . && uv run ruff format --check . "
    "&& uv run mypy core app"
)


def passing(head=HEAD, **override):
    return {
        "status": "result-ready",
        "assignment": "A1",
        "head_sha": head,
        "checks": [{"command": [GATE], "status": "passed", "evidence": "ok"}],
        **override,
    }


def write_ledger(root, *results, round_id="187"):
    """Write a round ledger the way scripts/round.py lays it out."""
    base = Path(root) / round_id
    (base / "results").mkdir(parents=True)
    paths = []
    for number, result in enumerate(results, 1):
        path = f"results/A1-{number}.json"
        (base / path).write_text(json.dumps(result))
        paths.append(path)
    manifest = {"assignments": {"A1": {"results": paths, "state": "result-ready"}}}
    (base / "round.json").write_text(json.dumps(manifest))


class FakeGh:
    """Answers `gh pr view` with a PR record and records every call."""

    def __init__(self, **pr):
        self.pr = {
            "state": "OPEN",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "headRefOid": HEAD,
            "baseRefName": "main",
            "files": [{"path": "core/pipeline.py"}],
            **pr,
        }
        self.calls = []
        self.merged = False

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if cmd[:3] == ["gh", "pr", "view"]:
            pr = dict(self.pr, state="MERGED" if self.merged else self.pr["state"])
            return 0, json.dumps(pr)
        if cmd[:3] == ["gh", "pr", "merge"]:
            self.merged = True
            return 0, ""
        return 1, "unexpected"

    def merge_calls(self):
        return [c for c in self.calls if c[:3] == ["gh", "pr", "merge"]]


class LedgerCase(unittest.TestCase):
    """Each test gets its own ledger dir; by default it holds one passing result at HEAD."""

    results: tuple = (passing(),)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Path(tmp.name)
        if self.results:
            write_ledger(self.ledger, *self.results)
        patcher = mock.patch.object(mg, "LEDGER", self.ledger)
        patcher.start()
        self.addCleanup(patcher.stop)


class MergeGuardTests(LedgerCase):
    def test_merges_the_reviewed_head_and_confirms_the_merge(self):
        gh = FakeGh()
        self.assertEqual(mg.guard(7, HEAD, run=gh), 0)
        [merge] = gh.merge_calls()
        self.assertEqual(merge[merge.index("--match-head-commit") + 1], HEAD)
        self.assertEqual(gh.calls[-1][:3], ["gh", "pr", "view"])  # merge state re-read

    def test_refuses_a_head_that_moved_after_review(self):
        gh = FakeGh(headRefOid="b" * 40)
        self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0)
        self.assertEqual(gh.merge_calls(), [])

    def test_refuses_an_unmergeable_draft_or_closed_pr(self):
        for pr in ({"mergeable": "CONFLICTING"}, {"isDraft": True}, {"state": "CLOSED"}):
            gh = FakeGh(**pr)
            self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0, pr)
            self.assertEqual(gh.merge_calls(), [], pr)

    def test_refuses_a_pr_that_does_not_target_the_default_branch(self):
        gh = FakeGh(baseRefName="feature")
        self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0)
        self.assertEqual(gh.merge_calls(), [])

    def test_leaves_operator_owned_paths_to_the_operator(self):
        for path in (
            "docs/decisions.md",
            "ARCHITECTURE.md",
            ".agent/config.toml",
            ".claude/settings.json",
            "scripts/merge_guard.py",
        ):
            gh = FakeGh(files=[{"path": "core/pipeline.py"}, {"path": path}])
            self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0, path)
            self.assertEqual(gh.merge_calls(), [], path)

    def test_allows_ordinary_records(self):
        gh = FakeGh(
            files=[{"path": "todo.md"}, {"path": "DATASOURCES.md"}, {"path": "providers.csv"}]
        )
        self.assertEqual(mg.guard(7, HEAD, run=gh), 0)

    def test_requires_a_full_head_sha(self):
        self.assertNotEqual(mg.guard(7, "abc123", run=FakeGh()), 0)


def check_run(conclusion, status="COMPLETED"):
    return {"__typename": "CheckRun", "name": "check", "status": status, "conclusion": conclusion}


def status_context(state):
    return {"__typename": "StatusContext", "context": "ext", "state": state}


class LedgerGateTests(LedgerCase):
    results = ()

    def refused(self, *results, **pr):
        if results:
            write_ledger(self.ledger, *results)
        gh = FakeGh(**pr)
        rc = mg.guard(7, HEAD, run=gh)
        return rc != 0 and gh.merge_calls() == []

    def test_refuses_without_any_ledger_result_at_head(self):
        self.assertTrue(self.refused())

    def test_refuses_a_result_at_a_different_head(self):
        self.assertTrue(self.refused(passing(head="b" * 40)))

    def test_refuses_a_check_that_did_not_pass(self):
        for status in ("failed", "unavailable", "waived"):
            with self.subTest(status=status):
                self.tearDown_ledger()
                result = passing()
                result["checks"][0]["status"] = status
                self.assertTrue(self.refused(result))

    def tearDown_ledger(self):
        for path in sorted(self.ledger.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()

    def test_refuses_and_names_a_missing_gate_component(self):
        checks = [
            {"command": ["uv", "run", "pytest", "-q"], "status": "passed"},
            {"command": ["uv run ruff check ."], "status": "passed"},
            {"command": ["uv run mypy core app"], "status": "passed"},
        ]
        write_ledger(self.ledger, passing(checks=checks))
        with mock.patch("sys.stderr") as err:
            gh = FakeGh()
            self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0)
        written = "".join(call.args[0] for call in err.write.call_args_list)
        self.assertIn("ruff format --check", written)
        self.assertNotIn("missing pytest", written)
        self.assertEqual(gh.merge_calls(), [])

    def test_split_checks_covering_the_gate_pass(self):
        checks = [
            {"command": ["uv", "run", "pytest", "-q"], "status": "passed"},
            {"command": ["uv", "run", "ruff", "check", "."], "status": "passed"},
            {"command": ["uv", "run", "ruff", "format", "--check", "."], "status": "passed"},
            {"command": ["uv", "run", "mypy", "core", "app"], "status": "passed"},
        ]
        self.assertFalse(self.refused(passing(checks=checks)))

    def test_one_passing_complete_result_among_several_at_head_suffices(self):
        failed = passing()
        failed["checks"] = [dict(failed["checks"][0], status="failed")]
        self.assertFalse(self.refused(failed, passing()))

    def test_refuses_a_failed_ci_check(self):
        write_ledger(self.ledger, passing())
        for entry in (
            check_run("FAILURE"),
            check_run("CANCELLED"),
            check_run("TIMED_OUT"),
            check_run("ACTION_REQUIRED"),
            check_run("ERROR"),
            status_context("FAILURE"),
            status_context("ERROR"),
        ):
            with self.subTest(entry=entry):
                gh = FakeGh(statusCheckRollup=[check_run("SUCCESS"), entry])
                self.assertNotEqual(mg.guard(7, HEAD, run=gh), 0)
                self.assertEqual(gh.merge_calls(), [])

    def test_accepts_green_pending_or_absent_ci(self):
        write_ledger(self.ledger, passing())
        for rollup in (
            [check_run("SUCCESS"), status_context("SUCCESS")],
            [check_run(None, status="IN_PROGRESS"), check_run("", status="QUEUED")],
            [status_context("PENDING"), status_context("EXPECTED")],
            [],
            None,
        ):
            with self.subTest(rollup=rollup):
                gh = FakeGh(statusCheckRollup=rollup)
                self.assertEqual(mg.guard(7, HEAD, run=gh), 0)

    def test_view_requests_the_check_rollup(self):
        write_ledger(self.ledger, passing())
        gh = FakeGh()
        mg.guard(7, HEAD, run=gh)
        self.assertIn("statusCheckRollup", gh.calls[0][gh.calls[0].index("--json") + 1])


if __name__ == "__main__":
    unittest.main()
