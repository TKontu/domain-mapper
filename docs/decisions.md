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

## D-004 — Search backend: self-hosted SearXNG (2026-11-25, operator)

- **Ruling:** discovery search uses the existing self-hosted SearXNG instance through its JSON API.
  No paid search API for now.
- **Consequences:** the endpoint lives only in the gitignored `.env` as `SEARXNG_URL`;
  `.env.example` carries a placeholder. A SearXNG implementation of the AD-05 `Search` adapter is
  a normal backlog item. Live searches run only when an item assigns them; the offline gate uses
  fakes.
- **Carried in:** `.env.example`, `todo.md` (Phase 0 search-backend row).

## D-005 — Raw storage: content-addressed filesystem blob store (2026-11-25, operator)

- **Ruling:** raw fetched bytes go into a content-addressed filesystem store behind a blob-store
  interface. MinIO stays an optional later backend of the same interface.
- **Consequences:** the store root comes from `.env` `RAW_STORE_DIR`, defaulting to the gitignored
  `data/raw/`. Blobs are keyed by the body's sha256 (as recorded on the `snapshot` row, CR-01), so
  identical bodies are stored once. Implementing the interface and filesystem backend is a normal
  backlog item.
- **Carried in:** `.env.example`, `todo.md` (Phase 0 raw-storage row).

## D-006 — Layering rule and governance ratchets (2026-11-26, operator direction; drafted by planner, owed to operator)

- **Ruling:** `core/` holds domain-agnostic engine logic and must not import concrete adapters;
  `adapters/` implement core interfaces and must not import `core/`; the CLI composition root lives
  in `app/`. Every public symbol has a production caller or an allowlisted "unwired by design" row.
- **Consequences:** enforced shrink-only in the local gate by
  `scripts/check_import_direction_ratchet.py` (GV-01) and `scripts/check_reachability_ratchet.py`
  (GV-02); allowlists only shrink. The configured gate type-checks `core` and `app`.
- **Carried in:** GV-01..GV-04 rows in `todo.md`, `.agent/config.toml` `[verification]`.
