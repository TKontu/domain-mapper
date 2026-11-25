# TODO — engine backlog

Phased plan for the domain-agnostic engine. Each phase ends with a milestone usable on its own.
Scope-specific work (vocabulary, rubric content, sources, gold sets, discovery for a real market)
lives in that scope pack's own backlog, not here.

Legend: `[ ]` open · `[x]` done · **(D)** decision needed · **(H)** needs human/domain expert.
An item ID links to its detail in `docs/items/`.

---

## First priority — publish the repository

Git over the NFS-mounted NAS is too slow for interactive work (`git add -A` takes ~4 min and
soft-mount timeouts leave stale `.git/index.lock` files). Finish this before any round:

- [x] Remove a stale `.git/index.lock` only if no git process runs in this checkout (`pgrep -a git`, check each `/proc/<pid>/cwd`)
- [x] `git add -A`; confirm nothing under `scopes/` except `scopes/README.md` and `scopes/_example/` is staged, and no `.env`, `__pycache__` or `settings.local.json`
- [x] Scan the staged diff for secrets (`/secrets-check`) and for private-scope leakage (D-003: no domain words, provider names or findings)
- [x] Run the local gate (`python3 -m unittest discover -s tests`) and `git diff --cached --check`
- [x] Initial commit on `main`; `gh repo create TKontu/domain-mapper --public` (without `--source`: `gh` rejects the NAS checkout's ownership), `git remote add origin …`, `git push -u origin main`
- [x] Check the minimal CI run passes on GitHub ✔ 2026-11-25 (run 45532598718)
- [ ] Consider moving the working checkout to local disk (NAS as backup/remote only) so rounds and lane worktrees are not bound by NFS latency

The private scope repo is already published (initial commit `d5714f3`).

---

## Phase 0 — Foundations & decisions

### Decisions
- [x] **(D)** Public engine repo, private scope packs (D-003) ✔ 2026-11-24
- [ ] **(D)** Search backend for discovery: self-hosted SearXNG vs. a paid search API vs. both — run a 200-query comparison
- [ ] **(D)** Raw storage: filesystem vs. MinIO
- [ ] Record the above in `docs/decisions.md`

### Repo & infra
- [x] **PF-01** Python project skeleton (uv), ruff, mypy, pytest, pre-commit → [docs/items/PF-01.md](docs/items/PF-01.md) ✔ 2026-11-25 (PR #1; gate switch in `.agent/config.toml` owed to operator)
- [x] **PF-02** → [docs/items/PF-02.md](docs/items/PF-02.md) Make `tests/test_round_supervisor.py` hermetic: `RepoNameTests.test_dashboard_lists_only_this_repos_lanes` reads live lane records when `ROUND_SUPERVISOR_ROOT` is set (fails inside supervised sessions) ✔ 2026-11-25 (PR #4)
- [ ] docker compose: Postgres 16 + pgvector, (MinIO), app container
- [x] **PF-03** → [docs/items/PF-03.md](docs/items/PF-03.md) Config: `.env` for endpoints (vLLM gateway, embedding/reranker service, cloud LLM key), budgets; `.env.example` tracked ✔ 2026-11-25 (PR #5; `load_settings` unwired until the Typer CLI row)
- [ ] Typer CLI skeleton: `landscape <stage> --scope --layer --limit --dry-run`
- [ ] Run/work-unit tracking table + resumable stage runner
- [ ] Logging + `llm_call` cost/latency logging
- [ ] Lane worktrees: helper that links private `scopes/<name>/` checkouts into a lane worktree

### Adapters (interfaces + first implementation)
- [ ] `Fetcher` adapter around own web scraping tool (document its API, JS rendering, PDF download)
- [ ] `Parser` adapter: HTML → Markdown; PDF → MinerU → Markdown with page numbers
- [ ] `Search` adapter
- [ ] `LLM` adapter: OpenAI-compatible (vLLM) + Anthropic; JSON-schema/guided output
- [ ] `Embedder` + `Reranker` adapters (local GPU service)
- [ ] Adapter interfaces: `BusinessRegister`, `AccreditationRegister`, `ProfessionalRegister`, `TenderSource`, `PublicationSource` (ARCHITECTURE §10)
- [ ] `source` catalogue table seeded from a scope pack's `sources.yaml` (roles, access class, licence)

### Scope pack schema
- [ ] Pydantic models: scope, segments, layers, vocabulary, sources, rubric, attributes, profiles, views
- [ ] `landscape validate --scope ...` command
- [ ] JSON Schema export for editor autocompletion
- [ ] `scopes/_example/` complete enough to validate (all files, tiny rubric, 3 seed entities)

**Milestone M0:** `landscape validate` passes on `scopes/_example/`; adapters return data for one test URL and one PDF.

---

## Phase 1 — Vertical slice (one layer, ~20 entities)

- [ ] Discover: search + register channels + LLM-recall channel, all tagged by channel
- [ ] Resolve: domain/name normalisation, business-register ID lookup, GLEIF groups, merge suggestions
- [ ] Enrich: register financial data (revenue, size) where public
- [ ] Screen: in-scope + segment classification (local small model), review borderline
- [ ] Crawl: per-entity policy (max pages, priority paths, PDFs), robots.txt, rate limit, snapshots
- [ ] Parse: HTML + MinerU, heading-aware chunks with provenance
- [ ] Index: embeddings + full-text + reranker; `landscape search "<query>" --entity X` for debugging
- [ ] Assess: extractor → verbatim-quote check → verifier → cloud adjudication → review queue
- [ ] Attributes: founded, employees band, sites, scope-defined counts
- [ ] Claim verification framework + first register verifier
- [ ] Aggregate: topic scores + coverage
- [ ] Export JSON for the frontend

### Minimal frontend
- [ ] Mosaic view (generic renderer driven by rubric config)
- [ ] Hover/click cell → status, quote, URL, snapshot date
- [ ] Entity page

**Milestone M1:** mosaic of ~20 providers for one layer of one scope, every cell clickable to evidence.

---

## Phase 2 — Quality & human-in-the-loop

- [ ] Eval harness: per-criterion accuracy, Cohen's κ, silent-rate, false-"strong" rate; discovery recall vs. a reference list
- [ ] `eval/thresholds.yaml` (overridable per scope) + release gate in `publish`
- [ ] Review UI: queue ordered by priority (disagreement × impact), accept/override with note, keyboard-driven
- [ ] Feed review decisions back as few-shot examples / rubric clarifications (versioned)
- [ ] Prompt + rubric versioning; re-assess only affected criteria on version bump
- [ ] Model bake-off harness on a gold set: extractor/verifier pairs, chunk size, top-n, reranker on/off
- [ ] Prompt-injection test cases (malicious text in scraped pages)
- [ ] Cost report per stage; cloud budget per run

**Milestone M2:** one layer meets thresholds; review effort per entity measured.

---

## Phase 3 — Multi-country & multilingual

- [ ] Language packs and multilingual retrieval check per language
- [ ] Register adapters per country (business, accreditation incl. withdrawn/suspended lists → red flags)
- [ ] Tender-source channel (e.g. TED Search API: classification codes + keywords, tenderer country filter)
- [ ] Group resolution across countries; inherited group documents
- [ ] Market presence model: `hq / site / field_service / sales / remote` incl. foreign providers serving a layer
- [ ] Capture–recapture recall estimate per segment × country
- [ ] Map view (MapLibre), layer + segment filters
- [ ] Throughput: batch scheduling on vLLM, crawl concurrency per host, cache hit-rate metrics
- [ ] Core-rubric pass for all; full-rubric pass for shortlist (profile-driven)
- [ ] Compare view (2–4 entities); capability matrix view

**Milestone M3:** nested layers navigable from outer to inner with map + mosaic and published recall estimates.

---

## Phase 4 — Synthesis & publishing

- [ ] Entity profiles (cited, no new facts) — local large model, cloud fallback
- [ ] Segment × layer narratives (cloud LLM, cited) + review
- [ ] Positioning clusters (capability vectors / embeddings)
- [ ] Weight profiles UI
- [ ] Method page generated from scope pack + eval results + versions
- [ ] Exports: Parquet / CSV / JSON
- [ ] Hosting with access control; publishing gated by the scope's declared purpose

**Milestone M4:** full site: Map · Mosaic · Compare · Matrix · Market · Entity · Method.

---

## Phase 5 — Genericity proof

- [ ] Run `scopes/_example/` end to end with **zero changes to `core/`** — log every place it was tempted
- [ ] Refactor any leaked domain logic into scope pack or adapters
- [ ] "New scope" guide in `docs/`

**Milestone M5:** two scopes running from the same engine.

---

## Phase 6 — Operations

- [ ] Scheduled re-crawls (cadence per layer from the scope pack)
- [ ] Snapshot diffing → selective re-assessment
- [ ] Changelog page (status changes, new/closed providers, certification changes)
- [ ] Alerts: crawl failures, budget overrun, review backlog
- [ ] Backups (DB + snapshots)

---

## Open questions

- Should customer-side signals (tender awards, published case studies with named clients) be a standard "track record" topic in the example rubric?
- How to treat providers whose website is thin but who are known to be strong (expert annotation with source tier `expert_note`)?
