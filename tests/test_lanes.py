"""Optional live-lane helper: start lanes that outlive the session, probe them, release them by rule."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lanes", ROOT / "scripts" / "lanes.py")
lanes = importlib.util.module_from_spec(_spec)
sys.modules["lanes"] = lanes  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(lanes)


class FakeWorld:
    """Scripted git, gh, Herdr and process table; records every command it is asked to run."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.trees = {str(root): "main"}
        self.prs = {}  # branch -> [states]
        self.dirty = set()
        self.status = {}  # path -> exact `git status --porcelain` output
        self.unpushed = set()
        self.tabs = []  # {"tab_id", "label"}
        self.panes = []  # {"tab_id", "cwd", "agent_status"}
        self.agents = {}  # name -> agent_status
        self.cwds = []
        self.alive = set()
        self.spawned = []
        self.next_id = 10

    def run(self, cmd, timeout=60):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:4] == ["git", "worktree", "list", "--porcelain"]:
            out = []
            for path, branch in self.trees.items():
                out += [
                    f"worktree {path}",
                    f"branch refs/heads/{branch}" if branch else "detached",
                    "",
                ]
            return 0, "\n".join(out)
        if cmd[:2] == ["git", "-C"] and cmd[3:5] == ["status", "--porcelain"]:
            if cmd[2] in self.status:
                return 0, self.status[cmd[2]]
            return 0, "?? x" if cmd[2] in self.dirty else ""
        if cmd[:2] == ["git", "-C"] and cmd[3] == "rev-list":
            return 0, "abc" if cmd[2] in self.unpushed else ""
        if cmd[:2] == ["git", "-C"] and cmd[3] == "clean":
            gone = set(cmd[cmd.index("--") + 1 :])
            left = [ln for ln in self.status.get(cmd[2], "").splitlines() if ln[3:] not in gone]
            self.status[cmd[2]] = "\n".join(left)
            return 0, ""
        if cmd[:2] == ["git", "worktree"] and cmd[2] in ("remove", "prune"):
            if cmd[2] == "remove" and (self.status.get(cmd[3]) or cmd[3] in self.dirty):
                return 128, "fatal: contains modified or untracked files, use --force"
            if cmd[2] == "remove":
                self.trees.pop(cmd[3], None)
            return 0, ""
        if cmd[:3] == ["gh", "pr", "list"]:
            branch = cmd[cmd.index("--head") + 1]
            return 0, json.dumps([{"state": s} for s in self.prs.get(branch, [])])
        if cmd[0] == "herdr":
            return self.herdr(cmd[1:])
        raise AssertionError(f"unexpected command {cmd}")

    def herdr(self, args):
        ok = lambda result: (0, json.dumps({"result": result}))  # noqa: E731
        if args == ["tab", "list"]:
            return ok({"tabs": self.tabs})
        if args == ["pane", "list"]:
            return ok({"panes": self.panes})
        if args[:2] == ["tab", "close"]:
            self.tabs = [t for t in self.tabs if t["tab_id"] != args[2]]
            return ok({})
        if args[:2] == ["agent", "get"]:
            if args[2] not in self.agents:
                return 1, json.dumps({"error": {"code": "agent_not_found"}})
            return ok({"agent": {"name": args[2], "agent_status": self.agents[args[2]]}})
        if args[:2] == ["tab", "create"]:
            label = args[args.index("--label") + 1]
            tab, pane = f"t{self.next_id}", f"p{self.next_id}"
            self.next_id += 1
            self.tabs.append({"tab_id": tab, "label": label})
            return ok({"tab": {"tab_id": tab}, "root_pane": {"pane_id": pane}})
        if args[:2] == ["agent", "start"]:
            self.agents[args[2]] = "working"
            return ok({})
        if args[:2] in (["pane", "run"], ["pane", "get"]):
            return ok({})
        raise AssertionError(f"unexpected herdr {args}")

    def seams(self, in_herdr=True):
        return lanes.Seams(
            run=self.run,
            in_herdr=lambda: in_herdr,
            pid_alive=lambda pid, wrapper: pid in self.alive,
            spawn=lambda argv, cwd, log: self.spawned.append(argv),
            cwds=lambda: self.cwds,
        )


class LaneTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "proj"
        self.root.mkdir()
        self.lane_dir = base / "proj-lanes" / "a1"
        self.lane_dir.mkdir(parents=True)
        self.world = FakeWorld(self.root)
        self.prefix = "live-proj-"
        self.env = {"HERDR_WORKSPACE_ID": "w1"}

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, body="exit 0"):
        path = self.root / "job.sh"
        path.write_text(body + "\n")
        return path

    def start(self, lane="a1", **kw):
        args = {"cwd": self.lane_dir, "cmd": self.script(), "agent": None, "branch": "lane/a1"}
        args.update(kw)
        seams = args.pop("seams", None) or self.world.seams()
        return lanes.start(
            lane, root=self.root, prefix=self.prefix, seams=seams, env=self.env, **args
        )

    def cleanup(self, go=True, handoff=""):
        return lanes.cleanup(
            root=self.root, prefix=self.prefix, go=go, handoff=handoff,
            seams=self.world.seams(), settle=0,
        )  # fmt: skip


class StartTests(LaneTestCase):
    def test_a_command_lane_runs_in_its_own_tab_in_the_supervisors_workspace(self):
        record = self.start()
        create = next(c for c in self.world.calls if c[1:3] == ["tab", "create"])
        self.assertIn("--workspace", create)
        self.assertEqual(create[create.index("--workspace") + 1], "w1")
        self.assertEqual(create[create.index("--label") + 1], "live-proj-a1")
        self.assertTrue(any(c[1:3] == ["pane", "run"] for c in self.world.calls))
        saved = json.loads((self.root / ".agent-runs/lanes/a1/lane.json").read_text())
        self.assertEqual(saved["label"], "live-proj-a1")
        self.assertEqual(saved["tab_id"], record["tab_id"])
        self.assertIn("lanes.py start a1", saved["resume_cmd"])

    def test_inside_herdr_without_a_workspace_id_is_refused(self):
        self.env = {}
        with self.assertRaisesRegex(lanes.LaneError, "HERDR_WORKSPACE_ID"):
            self.start()
        self.assertFalse(any(c[1:3] == ["tab", "create"] for c in self.world.calls))

    def test_a_running_lane_is_never_relaunched(self):
        self.start()
        run = Path(self.root / ".agent-runs/lanes/a1/runner.pid")
        run.write_text("4242\n")
        self.world.alive.add(4242)
        with self.assertRaisesRegex(lanes.LaneError, "already running"):
            self.start()

    def test_an_existing_tab_or_agent_name_is_refused(self):
        self.world.tabs.append({"tab_id": "t9", "label": "live-proj-a1"})
        with self.assertRaisesRegex(lanes.LaneError, "tab live-proj-a1 already exists"):
            self.start()
        self.world.tabs = []
        self.world.agents["live-proj-a1"] = "idle"
        with self.assertRaisesRegex(lanes.LaneError, "agent live-proj-a1 already exists"):
            self.start(cmd=None, agent="claude")

    def test_outside_herdr_a_command_lane_detaches_and_an_agent_lane_is_refused(self):
        seams = self.world.seams(in_herdr=False)
        self.start(seams=seams)
        self.assertEqual(self.world.spawned[0][:3], ["setsid", "-f", "nohup"])
        with self.assertRaisesRegex(lanes.LaneError, "needs Herdr"):
            self.start(lane="a2", cmd=None, agent="claude", seams=seams)

    def test_lane_names_are_plain_slugs(self):
        for bad in ("", "A1", "../x", "a b", "-a"):
            with self.assertRaises(lanes.LaneError):
                self.start(lane=bad)


class ProbeTests(LaneTestCase):
    def record(self, **kw):
        files = self.root / "run"
        files.mkdir(exist_ok=True)
        base = {
            "lane": "a1", "label": "live-proj-a1", "agent": "", "pid_file": str(files / "runner.pid"),
            "rc_file": str(files / "runner.rc"), "resume_cmd": "resume-me",
        }  # fmt: skip
        return {**base, **kw}, files

    def test_states(self):
        seams = self.world.seams()
        rec, files = self.record()
        self.assertEqual(lanes.probe(rec, seams)[0], "dead")
        self.assertIn("resume-me", lanes.probe(rec, seams)[1])
        (files / "runner.pid").write_text("77\n")
        self.world.alive.add(77)
        self.assertEqual(lanes.probe(rec, seams)[0], "running")
        (files / "runner.rc").write_text("0\n2026-01-01T00:00:00Z\n")
        self.assertEqual(lanes.probe(rec, seams)[0], "done")
        (files / "runner.rc").write_text("3\n")
        self.assertEqual(lanes.probe(rec, seams), ("blocked", f"rc 3 ({files / 'runner.rc'})"))
        rec["recorded"] = "round 5"
        self.assertEqual(lanes.probe(rec, seams)[0], "recorded")

    def test_an_idle_agent_is_not_a_finished_lane(self):
        rec, _ = self.record(agent="live-proj-a1")
        self.world.agents["live-proj-a1"] = "idle"
        state, detail = lanes.probe(rec, self.world.seams())
        self.assertEqual(state, "running")
        self.assertIn("idle", detail)
        self.world.agents["live-proj-a1"] = "blocked"
        self.assertEqual(lanes.probe(rec, self.world.seams())[0], "blocked")
        del self.world.agents["live-proj-a1"]
        self.assertEqual(lanes.probe(rec, self.world.seams())[0], "dead")
        self.assertEqual(lanes.probe(rec, self.world.seams(in_herdr=False))[0], "unknown")

    def test_recorded_marks_the_lane_and_status_lists_it(self):
        self.start()
        lanes.mark_recorded(self.root, "a1", "written up in round 7")
        rows = lanes.status_rows(self.root, self.world.seams())
        self.assertEqual([(r["lane"], r["state"]) for r in rows], [("a1", "recorded")])
        with self.assertRaisesRegex(lanes.LaneError, "no lane record"):
            lanes.mark_recorded(self.root, "zz", "x")


class CleanupTests(LaneTestCase):
    def finished_lane(self):
        path = str(self.lane_dir)
        self.world.trees[path] = "lane/a1"
        self.world.prs["lane/a1"] = ["MERGED"]
        self.world.tabs.append({"tab_id": "t1", "label": "live-proj-a1"})
        self.world.panes.append({"tab_id": "t1", "cwd": path, "agent_status": "idle"})
        return path

    def agent_lane_reporting_parent(self, *, state="done", registered=True):
        """A recorded --agent lane whose single pane reports the parent dir, not the worktree."""
        record = self.start(cmd=None, agent="claude")
        if state in ("done", "recorded"):
            Path(record["rc_file"]).write_text("0\n2026-01-01T00:00:00Z\n")
            if state == "recorded":
                lanes.mark_recorded(self.root, "a1", "round 14")
        elif state == "blocked":
            self.world.agents[record["label"]] = "blocked"
        elif state != "running":
            raise AssertionError(state)
        path = str(self.lane_dir)
        if registered:
            self.world.trees[path] = "lane/a1"
            self.world.prs["lane/a1"] = ["MERGED"]
        self.world.tabs = [{"tab_id": record["tab_id"], "label": record["label"]}]
        self.world.panes = [{"tab_id": record["tab_id"], "cwd": "/work", "agent_status": "idle"}]
        self.world.cwds = ["/work"]
        return path, record

    def test_dry_run_changes_nothing(self):
        path = self.finished_lane()
        report = self.cleanup(go=False)
        self.assertIn("would close tab t1 live-proj-a1", report)
        self.assertIn(f"would remove {path} (lane/a1)", report)
        self.assertIn(path, self.world.trees)
        self.assertEqual(len(self.world.tabs), 1)

    def test_a_finished_lane_loses_its_tab_then_its_worktree(self):
        path = self.finished_lane()
        seams = self.world.seams()
        seams.cwds = lambda: [] if not self.world.tabs else [path]  # the tab's shell, until closed
        report = lanes.cleanup(
            root=self.root, prefix=self.prefix, go=True, handoff="", seams=seams, settle=0
        )
        self.assertIn("closed tab t1 live-proj-a1", report)
        self.assertIn(f"removed {path} (lane/a1)", report)
        self.assertNotIn(path, self.world.trees)
        remove = next(c for c in self.world.calls if c[:3] == ["git", "worktree", "remove"])
        self.assertNotIn("--force", remove)

    def test_uncertain_worktrees_are_held_with_a_reason(self):
        path = self.finished_lane()
        self.world.tabs, self.world.panes = [], []
        cases = [
            ("no-pr", lambda: self.world.prs.pop("lane/a1")),
            ("open-pr", lambda: self.world.prs.__setitem__("lane/a1", ["MERGED", "OPEN"])),
            ("dirty", lambda: self.world.dirty.add(path)),
            ("unpushed", lambda: self.world.unpushed.add(path)),
            ("busy", lambda: self.world.cwds.append(path + "/sub")),
        ]
        for reason, arrange in cases:
            with self.subTest(reason):
                self.world.prs["lane/a1"] = ["MERGED"]
                self.world.dirty.clear(), self.world.unpushed.clear(), self.world.cwds.clear()
                arrange()
                self.assertIn(f"hold {path} (lane/a1): {reason}", self.cleanup())
                self.assertIn(path, self.world.trees)

    def recorded_lane(self, branch="lane/a1"):
        """A finished, written-up command lane whose worktree has no tab left."""
        record = self.start(cwd=self.lane_dir, branch=branch)
        Path(record["rc_file"]).write_text("0\n2026-01-01T00:00:00Z\n")
        lanes.mark_recorded(self.root, "a1", "round 9")
        self.world.tabs, self.world.panes = [], []
        path = record["cwd"]
        self.world.trees[path] = branch
        if branch:
            self.world.prs[branch] = ["MERGED"]
        return path

    def test_a_tree_dirty_only_with_disposable_files_is_removed(self):
        path = self.finished_lane()
        self.world.tabs, self.world.panes = [], []
        self.world.status[path] = "\n".join(
            ["?? run.sh", "?? live_runner.sh", "?? .venv/", "?? core/__pycache__/",
             "?? .pytest_cache/", "?? .env", "?? HANDOFF.md", "?? .claude/settings.local.json"]
        )  # fmt: skip
        self.assertIn(f"would remove {path} (lane/a1)", self.cleanup(go=False))
        self.assertIn(f"removed {path} (lane/a1)", self.cleanup())
        clean = next(c for c in self.world.calls if c[3:4] == ["clean"])
        self.assertEqual(clean[-1], ".claude/settings.local.json")
        self.assertNotIn(path, self.world.trees)

    def test_real_uncommitted_files_still_hold_the_tree(self):
        path = self.finished_lane()
        self.world.tabs, self.world.panes = [], []
        for status in ("?? run.sh\n?? core/new.py", " M run.sh", "?? docs/HANDOFF.md"):
            with self.subTest(status):
                self.world.status[path] = status
                report = self.cleanup()
                self.assertTrue(any(r.startswith(f"hold {path} (lane/a1): dirty") for r in report))
                self.assertIn(path, self.world.trees)

    def test_a_recorded_detached_tree_reachable_from_a_remote_is_removed(self):
        path = self.recorded_lane(branch="")
        self.world.status[path] = "?? run.sh"
        report = self.cleanup()
        self.assertIn(f"removed {path} (detached)", report)
        self.assertNotIn(path, self.world.trees)
        self.assertFalse(any(c[:3] == ["gh", "pr", "list"] for c in self.world.calls))

    def test_a_recorded_branch_lane_without_a_pr_is_removed_once_pushed(self):
        path = self.recorded_lane()
        self.world.prs.pop("lane/a1")
        self.assertIn(f"would remove {path} (lane/a1)", self.cleanup(go=False))

    def test_a_detached_tree_with_unpushed_commits_or_no_record_is_held(self):
        path = self.recorded_lane(branch="")
        self.world.unpushed.add(path)
        self.assertIn(f"hold {path} (detached): detached-unpushed", self.cleanup())
        self.assertIn(path, self.world.trees)
        other = str(Path(self.tmp.name) / "proj-other")
        self.world.trees[other] = ""
        self.assertIn(f"hold {other} (detached): detached", self.cleanup())

    def test_a_running_detached_lane_is_untouched(self):
        self.start(cwd=self.lane_dir, branch="")
        Path(self.root / ".agent-runs/lanes/a1/runner.pid").write_text("99\n")
        self.world.alive.add(99)
        path = str(self.lane_dir.resolve())
        self.world.trees[path] = ""
        self.world.tabs, self.world.panes = [], []
        report = self.cleanup()
        self.assertIn(f"hold {path} (detached): lane a1 running", report)
        self.assertFalse(any(c[:3] == ["git", "worktree", "remove"] for c in self.world.calls))

    def test_a_carried_lane_is_released_once_it_is_recorded(self):
        path = self.recorded_lane()
        record = json.loads((self.root / ".agent-runs/lanes/a1/lane.json").read_text())
        self.world.tabs = [{"tab_id": record["tab_id"], "label": record["label"]}]
        self.world.panes = [{"tab_id": record["tab_id"], "cwd": path, "agent_status": "idle"}]
        handoff = "## Carried lanes\n- live-proj-a1: long sweep, do not halt\n"
        report = self.cleanup(go=False, handoff=handoff)
        self.assertIn(f"would close tab {record['tab_id']} live-proj-a1", report)
        self.assertIn(f"would remove {path} (lane/a1)", report)

    def test_carried_lanes_are_held(self):
        path = self.finished_lane()
        handoff = "## Carried lanes\n- live-proj-a1: long sweep, do not halt\n"
        report = self.cleanup(handoff=handoff)
        self.assertIn("hold tab t1 live-proj-a1: carried", report)
        self.assertIn(path, self.world.trees)
        self.assertEqual(len(self.world.tabs), 1)

    def test_working_panes_and_protected_or_foreign_tabs_are_kept(self):
        path = self.finished_lane()
        self.world.panes[0]["agent_status"] = "working"
        self.assertIn("hold tab t1 live-proj-a1: active: working", self.cleanup(go=False))
        self.world.panes[0]["agent_status"] = "idle"
        self.world.tabs.append({"tab_id": "t2", "label": "live-proj-planner"})
        self.world.tabs.append({"tab_id": "t3", "label": "live-proj-x"})
        self.world.panes.append({"tab_id": "t3", "cwd": str(self.root), "agent_status": "idle"})
        self.world.tabs.append({"tab_id": "t4", "label": "live-other-a1"})
        report = self.cleanup()
        self.assertIn("hold tab t3 live-proj-x: foreign", report)
        self.assertEqual({t["tab_id"] for t in self.world.tabs}, {"t2", "t3", "t4"})
        self.assertNotIn(path, self.world.trees)

    def test_a_recorded_lane_that_is_still_running_is_held(self):
        self.start(cwd=self.lane_dir)
        self.world.tabs = []
        path = self.finished_lane()
        Path(self.root / ".agent-runs/lanes/a1/runner.pid").write_text("99\n")
        self.world.alive.add(99)
        report = self.cleanup()
        self.assertIn(f"hold {path} (lane/a1): lane a1 running", report)
        self.assertIn("hold tab t1 live-proj-a1: lane a1 running", report)

    def test_without_a_process_table_busy_is_unknown_and_held(self):
        path = self.finished_lane()
        self.world.tabs, self.world.panes = [], []
        seams = self.world.seams()
        seams.cwds = lambda: None
        report = lanes.cleanup(
            root=self.root, prefix=self.prefix, go=True, handoff="", seams=seams, settle=0
        )
        self.assertIn(f"hold {path} (lane/a1): busy-unknown", report)

    def test_a_finished_records_tab_reporting_a_parent_cwd_is_attributed_to_its_worktree(self):
        for state in ("done", "recorded"):
            with self.subTest(state):
                self.world = FakeWorld(self.root)
                path, record = self.agent_lane_reporting_parent(state=state)
                report = self.cleanup(go=False)
                self.assertIn(f"would close tab {record['tab_id']} live-proj-a1", report)
                self.assertIn(f"would remove {path} (lane/a1)", report)
                self.assertEqual(len(self.world.tabs), 1)

    def test_a_finished_record_whose_worktree_is_already_gone_still_releases_its_tab(self):
        path, record = self.agent_lane_reporting_parent(registered=False)
        report = self.cleanup(go=False)
        self.assertIn(f"would close tab {record['tab_id']} live-proj-a1", report)
        self.assertNotIn(path, "\n".join(report))

    def test_an_attributed_tab_keeps_its_hold_while_its_worktree_is_kept(self):
        path, record = self.agent_lane_reporting_parent()
        self.world.prs["lane/a1"] = ["MERGED", "OPEN"]
        report = self.cleanup(go=False)
        self.assertIn(f"hold tab {record['tab_id']} live-proj-a1: worktree-kept", report)
        self.assertIn(f"hold {path} (lane/a1): open-pr", report)

    def test_an_unfinished_record_holds_its_tab_before_attribution(self):
        for state in ("running", "blocked"):
            with self.subTest(state):
                self.world = FakeWorld(self.root)
                _, record = self.agent_lane_reporting_parent(state=state)
                report = self.cleanup(go=False)
                self.assertIn(f"hold tab {record['tab_id']} live-proj-a1: lane a1 {state}", report)

    def test_a_parent_pane_cwd_without_a_record_is_still_foreign(self):
        self.world.tabs.append({"tab_id": "t1", "label": "live-proj-x"})
        self.world.panes.append({"tab_id": "t1", "cwd": "/work", "agent_status": "idle"})
        self.assertIn("hold tab t1 live-proj-x: foreign", self.cleanup(go=False))

    def test_a_records_cwd_outside_the_lane_paths_is_not_attributed(self):
        self.start(cwd=self.root, cmd=None, agent="claude")
        Path(self.root / ".agent-runs/lanes/a1/runner.rc").write_text("0\n2026-01-01T00:00:00Z\n")
        self.world.tabs = [{"tab_id": "t10", "label": "live-proj-a1"}]
        self.world.panes = [{"tab_id": "t10", "cwd": "/work", "agent_status": "idle"}]
        report = self.cleanup(go=False)
        self.assertIn("hold tab t10 live-proj-a1: foreign", report)
        self.assertNotIn("would close tab t10 live-proj-a1", report)

    def test_a_finished_records_tab_with_no_listed_panes_is_still_foreign(self):
        _, record = self.agent_lane_reporting_parent(registered=False)
        self.world.panes = []
        report = self.cleanup(go=False)
        self.assertIn(f"hold tab {record['tab_id']} live-proj-a1: foreign", report)

    def test_a_pane_reporting_another_lane_path_keeps_the_tab_foreign(self):
        path, record = self.agent_lane_reporting_parent(registered=False)
        self.world.panes.append({"tab_id": record["tab_id"], "cwd": path, "agent_status": "idle"})
        report = self.cleanup(go=False)
        self.assertIn(f"hold tab {record['tab_id']} live-proj-a1: foreign", report)

    def test_a_working_pane_still_holds_its_tab_before_attribution(self):
        _, record = self.agent_lane_reporting_parent()
        self.world.panes[0]["agent_status"] = "working"
        report = self.cleanup(go=False)
        self.assertIn(f"hold tab {record['tab_id']} live-proj-a1: active: working", report)


@unittest.skipUnless(
    shutil.which("setsid") and Path("/proc/self").exists(), "needs setsid and /proc"
)
class WrapperTests(unittest.TestCase):
    """The real wrapper, detached with setsid, outside Herdr."""

    def test_the_wrapper_records_pid_log_and_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "proj", Path(tmp) / "proj-lanes" / "w1"
            root.mkdir(), work.mkdir(parents=True)
            job = root / "job.sh"
            job.write_text("echo hello from $PWD\nexit 3\n")
            seams = lanes.default_seams()
            seams.in_herdr = lambda: False
            rec = lanes.start(
                "w1", root=root, prefix="live-proj-", cwd=work, cmd=job, agent=None,
                branch="", seams=seams, env={},
            )  # fmt: skip
            rc = Path(rec["rc_file"])
            deadline = time.time() + 20
            while not rc.exists() and time.time() < deadline:
                time.sleep(0.1)
            self.assertEqual(rc.read_text().split()[0], "3")
            self.assertIn(f"hello from {work}", Path(rec["log"]).read_text())
            self.assertEqual(lanes.probe(rec, seams)[0], "blocked")


class CliTests(unittest.TestCase):
    def test_status_with_no_lanes_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "ROUND_SUPERVISOR_ROOT": tmp}
            env.pop("HERDR_ENV", None)
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "lanes.py"), "status"],
                capture_output=True, text=True, env=env, check=False,
            )  # fmt: skip
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("no lanes", done.stdout)


if __name__ == "__main__":
    unittest.main()
