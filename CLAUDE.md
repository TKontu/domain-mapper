# Claude entrypoint

Before any action, read and follow [`AGENTS.md`](AGENTS.md). It is the canonical agent contract.
Do not duplicate its rules here.

## Project Commands

Every command in `.claude/commands/` and the `tdd`/`debug`/`verify` skills resolve their tooling
from this table. Keys marked `n/a` have no tooling yet; commands skip those steps and say so.

<!-- claude:commands -->
| Key       | Command |
| --------- | ------- |
| install   | uv sync |
| test      | uv run pytest |
| test-one  | uv run pytest {arg} |
| lint      | uv run ruff check . |
| lint-fix  | uv run ruff check --fix . |
| format    | uv run ruff format . |
| typecheck | uv run mypy core |
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
