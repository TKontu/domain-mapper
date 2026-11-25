# Domain Mapper agent contract

Canonical guide for every development agent. Vendor files (`CLAUDE.md`) point here and must not
duplicate it.

## Authority and context

- `ARCHITECTURE.md` is the authoritative target for the domain-agnostic engine. `todo.md` is the
  sole engine plan/status registry; an item ID in a row links to its detail in `docs/items/`. A
  handoff, prompt or summary never substitutes for those sources. Decisions are recorded in
  `docs/decisions.md` (append-only).
- **Public repo, private scopes (D-003):** this repository is public and domain-neutral. Real scope
  packs are private checkouts under the gitignored `scopes/<name>/`, each with its own README,
  backlog and research. Read a scope pack only when the assignment names it. Never copy a private
  scope's domain words, provider names, findings or file contents into tracked files, commit
  messages, PR titles/bodies, review comments or handoffs here; describe engine needs in neutral
  terms and use `scopes/_example/` for examples and tests. Before every commit, read the staged diff
  for scope leakage as well as secrets.
- Task markers in `todo.md`: **(D)** needs an operator decision; **(H)** needs the human/domain
  expert. Agents prepare options and evidence for these, never decide them.
- Observed state first: probe the repository, PRs and running lanes before acting. Handoffs and
  memory are dated records, not facts.

## Development loop

The Reconcile-loop protocol runs this project (`SPEC.md`, `.agent/commands/`, `.agent/config.toml`).

The planner owns selection, dependencies, assignment, shared resources and reconciliation. Executors
own only assigned code/records and work in isolated checkouts at the pinned base. Preserve unrelated
work; never force-push. Return target/scope conflicts to planning; make routine authorized choices.

For behavior changes show the intended failing test, implement the smallest fix and run targeted
checks. Use the configured local gate and assigned integration/live checks in their permitted
safe environments. A skipped check is not a pass. Review relevant callers, data boundaries and scale.

- **Local gate first:** run the configured gate locally before every PR; CI is a minimal backstop
  (offline unit tests and a whitespace check only, `.github/workflows/check.yml`). Do not move
  checks into CI that the local gate does not run, and never treat green CI as the verification.
- **Procedures:** use the `tdd`, `debug` and `verify` skills; run `/lint-fix` once the stack has a
  linter. Run the assignment's premise checks before any code.
- **Done means seen:** a behaviour row is done only with `seen: <run id or artifact path>`. Merged
  but never exercised (flag off, stage not run against a real source) stays partial and names the
  run that will show it. An eval threshold is met only with the eval run that shows it.
- **Connections:** a new public symbol names its production caller, or is "unwired by design"
  with the row that wires it.
- **Scale:** loops over entities, snapshots, chunks or embeddings bound memory (stream, never hold
  a whole layer), batch and transaction sizes, crawl concurrency per host and LLM spend, or record
  the untested bound as a residual.
- **Private and live data are absent from fresh checkouts:** snapshots, databases and `.env` are
  gitignored. An assignment names where each comes from; a check skipped for lack of them, or for
  lack of network or model endpoints, is reported as skipped. Green CI does not mean a live
  source or the local-model path was exercised.
- **Who judged what:** say "expert" only for the domain expert's own judgment; any status, score or
  label a model produced is "LLM-built". Quote a figure only with its basis: population, metric,
  reference and who built it. A provider fact is stated only with its cited source and date.
- **Flags have a lifecycle:** each flag names the condition that removes it; a shortcut is recorded
  as debt with its follow-up row.
- **No silent design changes:** an architecture or rubric change needs a recorded decision in
  `docs/decisions.md`; never smuggle one through implementation.

Before publication inspect the diff for defects, secrets and private material. PRs record assignment
identity, head, actual verification and residuals. The designated reviewer checks the current head and
merge authority; executors do not self-merge.

## Merging (D-002)

The planner merges reviewed PRs under the standing delegation D-002 (`docs/decisions.md`), only
through `python3 scripts/merge_guard.py PR --head FULL_REVIEWED_SHA`. Raw `gh pr merge` is denied.
PRs touching `docs/decisions.md`, `ARCHITECTURE.md`, `.agent/config.toml`, `.claude/*` or
`scripts/merge_guard.py`, or deciding a (D)/(H) item, are parked as owed to the operator.

## Data and network safety

- Never commit secrets (`.env`), raw snapshots, crawled content or database files (see `.gitignore`).
- Person-level registers (licensed or sworn professionals) are personal data: collect only what the
  scope pack names, and only after its legitimate-interest note exists.
- Respect each source's terms recorded in the scope pack's source catalogue; never automate a
  source whose terms bar automated access. The verification gate runs offline; live crawls and
  paid APIs run only when an item assigns them.

## Rounds

After integration, verify combined state, reconcile observed outcomes into canonical records and
write a bounded handoff. Start the next round with fresh context only after reconciliation closes the
previous one. Use SPEC.md and `.agent/commands/` for phase contracts. No conversation is project memory.
