# merge-round

Read project AGENTS.md and `.agent/config.toml`; follow [the protocol](../../SPEC.md).

Confirm current head, review, required checks, mergeability and existing merge authorization for
each candidate. Ask only for a missing approval on a concrete reviewed change. Do not infer authority
from a ready verdict. Resolve conflicts deliberately and renew affected checks; never force-push.

Merge serially with `python3 scripts/merge_guard.py PR --head FULL_REVIEWED_HEAD` (raw `gh pr merge`
is denied; decision D-002 in `docs/decisions.md`). It refuses drafts, moved heads, unmergeable or
off-base PRs and operator-owned paths; park those as owed to the operator. Verify each actual merge and refresh the
local default branch between merges. Reassess remaining work if shared assumptions changed. Then:

```sh
python3 scripts/round.py resolve ROUND A1 --outcome merged --head FULL_REVIEWED_HEAD --merge FULL_MERGE_SHA --evidence "PR review and merge references"
```

The CLI checks the result head and supplied check statuses; it does not call the PR host or independently
prove review/merge correctness. That verification belongs to this phase. When a reviewed candidate has a
check that did not pass for a pre-existing reason, record the merged outcome with `--waive "<reason
citing the standing ruling>"`; the assignment stores the reason and the non-passed check
commands/statuses, and `--waive` is refused for a blank reason, with `--outcome cancelled`, or when
every check passed. If work is abandoned, confirm its worker has stopped and resource access ended,
then record cancellation with an explanation:

```sh
python3 scripts/round.py resolve ROUND A1 --outcome cancelled --evidence "Stopped; remaining work recorded in TASK-1"
```

Then release finished lanes: `python3 scripts/lanes.py cleanup --go`. It closes the tabs and
removes the worktrees of lanes whose PRs are merged or closed; it holds anything carried, live,
dirty, unpushed, busy or without a PR. Report held items; never force them.

Do not start another round yet. All merged and cancelled outcomes still require reconciliation.
