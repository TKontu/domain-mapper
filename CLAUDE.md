# Claude entrypoint

Before any action, read and follow [`AGENTS.md`](AGENTS.md). It is the canonical agent contract.
Do not duplicate its rules here.

## Project Commands

Every command in `.claude/commands/` and the `tdd`/`debug`/`verify` skills resolve their tooling
from this table. No product stack exists yet, so most keys are `n/a`; commands skip those steps and
say so. Item PF-01 fills in the uv/pytest/ruff/mypy rows.

<!-- claude:commands -->
| Key       | Command |
| --------- | ------- |
| install   | n/a |
| test      | python3 -m unittest discover -s tests |
| test-one  | python3 -m unittest {arg} |
| lint      | n/a |
| lint-fix  | n/a |
| format    | n/a |
| typecheck | n/a |
| run       | n/a |
| build     | n/a |
<!-- /claude:commands -->

## Commands

- Round workflow: `/bootstrap`, `/takeoff`, `/plan-round`, `/execute-task`, `/review-round`,
  `/merge-round`, `/reconcile`, `/handoff`, plus `/status` and `/pr-verdict`.
- Everyday: `/test`, `/lint-fix`, `/review`, `/secrets-check`, `/commit-push-pr`.
- Skills: `tdd`, `debug`, `verify`.
- Subagents: `review-lens`, `ci-watcher`, `quick-fixer`. Loop watcher: `/loop-manager`.
- Unattended loop: `python3 scripts/round_supervisor.py up` from a Herdr pane
  (`.agent/supervisor.md`); long work runs as lanes via `scripts/lanes.py`.
