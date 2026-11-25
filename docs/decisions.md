# Decisions

Append-only record of operator rulings and architecture decisions.

## D-002 — Planners merge only through the merge guard (2026-11-24, operator)

Adopted from the same standing merge delegation used by the operator's other Reconcile-loop
projects; the number is kept so `scripts/merge_guard.py` and the round prompt cite one decision.

- **Ruling:** the round planner merges any PR whose current head passed independent review
  (`/review-round`), passed the configured gate, and is mergeable, following `/merge-round`.
  Raw `gh pr merge` is denied in `.claude/settings.json`; the only merge path is
  `python3 scripts/merge_guard.py PR --head FULL_REVIEWED_SHA`.
- **The guard checks:** the PR is open, not a draft, mergeable, targets the default branch and
  still has exactly the reviewed head. It merges with `--match-head-commit` and re-reads the PR to
  confirm the merge.
- **Still the operator's:** PRs that change `docs/decisions.md`, `ARCHITECTURE.md`,
  `.agent/config.toml`, `.claude/*` or `scripts/merge_guard.py` (the guard refuses these); PRs
  that decide a (D) or (H) item (the guard cannot see this, so
  the planner parks them as owed). Executors never merge their own work.
- **Why:** in an earlier project, reserving every merge to the operator parked approved PRs, so main
  stopped moving and the supervisor halted ("main unchanged across 3 rounds"). The guard keeps
  routine merges moving without giving planners a raw merge command.
- **Carried in:** `[policy] merge_authority` in `.agent/config.toml` and the round prompt in
  `scripts/round_supervisor.py`.

## D-003 — Public generic engine, private scope packs (2026-11-24, operator)

- **Ruling:** this repository is public and contains only the domain-agnostic engine, its docs,
  the workflow pack and the illustrative `scopes/_example/`. It must not declare or hint at any
  real scope's focus domain. Each real scope pack is its own private repository, cloned into the
  gitignored `scopes/<name>/`.
- **Consequences:** tracked files, commit messages, PR titles/bodies and handoffs use
  domain-neutral words; scope-specific backlog, research, seed lists and source catalogues live in
  the scope repository. Lane worktrees link a private scope from the main checkout
  (`scopes/README.md`).
- **Carried in:** `.gitignore`, `AGENTS.md`, `scopes/README.md`.
