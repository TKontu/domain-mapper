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
- [x] **(D)** Search backend for discovery: self-hosted SearXNG (D-004) ✔ 2026-11-25
- [x] **(D)** Raw storage: content-addressed filesystem blob store, MinIO optional later (D-005) ✔ 2026-11-25
- [ ] Record the above in `docs/decisions.md`

### Repo & infra
- [x] **PF-01** Python project skeleton (uv), ruff, mypy, pytest, pre-commit → [docs/items/PF-01.md](docs/items/PF-01.md) ✔ 2026-11-25 (PR #1; gate switch in `.agent/config.toml` owed to operator)
- [x] **PF-02** → [docs/items/PF-02.md](docs/items/PF-02.md) Make `tests/test_round_supervisor.py` hermetic: `RepoNameTests.test_dashboard_lists_only_this_repos_lanes` reads live lane records when `ROUND_SUPERVISOR_ROOT` is set (fails inside supervised sessions) ✔ 2026-11-25 (PR #4)
- [ ] **PF-05** → [docs/items/PF-05.md](docs/items/PF-05.md) docker compose: Postgres 16 + pgvector, (MinIO), app container — partial 2026-11-25: merged (PR #8), not yet seen running; needs a `docker compose up -d db` + pgvector query and an image build on a host with docker
- [x] **PF-03** → [docs/items/PF-03.md](docs/items/PF-03.md) Config: `.env` for endpoints (vLLM gateway, embedding/reranker service, cloud LLM key), budgets; `.env.example` tracked ✔ 2026-11-25 (PR #5; `load_settings` wired by PF-04)
- [x] **PF-04** → [docs/items/PF-04.md](docs/items/PF-04.md) Typer CLI skeleton: `landscape <stage> --scope --layer --limit --dry-run` ✔ 2026-11-25 (PR #7; seen: `uv run landscape discover --scope scopes/_example --limit 2 --dry-run` on main 47e1242; stage bodies unwired until their rows)
- [ ] **RT-01** → [docs/items/RT-01.md](docs/items/RT-01.md) Run/work-unit tracking table + resumable stage runner — partial 2026-11-25: merged (PR #11); seen on SQLite only (in-memory run on main dc3ae93: fresh 2 executed, rerun 1 skipped/1 re-executed); needs a run against the compose Postgres (`DM_TEST_DATABASE_URL`) on a host with docker
- [ ] **RT-02** → [docs/items/RT-02.md](docs/items/RT-02.md) Logging + `llm_call` cost/latency logging — partial 2026-11-25: merged (PR #14); seen on SQLite only (pytest on main 0789345: rows written on success and error, prompt text absent); unwired until the `LLM` adapter row; needs a run against the compose Postgres on a host with docker
- [ ] Configured gate `python3 -m unittest discover -s tests` fails under bare system python (no sqlalchemy) since RT-01; it passes under `uv run`. Resolved by the operator-owned gate switch (PR #3)
- [ ] `ruff format --check .` fails on main at a code block in ARCHITECTURE.md (operator-owned file); gates exclude it until the operator fixes or excludes it
- [x] **PF-06** → [docs/items/PF-06.md](docs/items/PF-06.md) Packet gate `ruff format --check adapters tests` failed on 9 pre-existing test files — done 2026-11-25: PR #29 (layout only, AST-identical); seen: `uv run ruff format --check adapters tests` on main → 22 files already formatted
- [x] **LW-02** → [docs/items/LW-02.md](docs/items/LW-02.md) `link_scopes.py` prints a traceback, not a message, when `--worktree` is not a git checkout (LW-01 residual) — done 2026-11-25: PR #34; seen: `python3 scripts/link_scopes.py --worktree /tmp` on main b564966 → `link_scopes: /tmp is not a git checkout`, exit 2
- [x] **LW-03** → [docs/items/LW-03.md](docs/items/LW-03.md) `lanes.py cleanup` holds `--agent` lane tabs as "foreign" when the agent pane reports cwd `/work` instead of the lane worktree — done 2026-11-25: PR #35 (planner review fix f78bcd5 keeps the worktree-kept hold); seen: `python3 scripts/lanes.py cleanup --go` on main b564966 closed 9 recorded lanes' tabs previously held "foreign" (rounds 5-9, 14); tabs named carried in HANDOFF.md stayed held
- [x] **LW-04** → [docs/items/LW-04.md](docs/items/LW-04.md) `scripts/lanes.py` ruff-formatted (layout only, AST-identical) and `.agent/supervisor.md` states LW-03's attribution exception — done 2026-11-25: PR #37; seen: `uv run ruff format --check adapters tests scripts/lanes.py` on main 0306b62 → 27 files already formatted; no behaviour change
- [x] **SP-05** → [docs/items/SP-05.md](docs/items/SP-05.md) `landscape validate` summary prints counts for optional pack files (vocabulary, sources, attributes, profiles, views) — done 2026-11-25: PR #22; seen: `uv run landscape validate --scope scopes/_example` on head ab7090b (vocabulary 2, sources 3, attributes 3, profiles 2, views 2)
- [x] **LW-01** → [docs/items/LW-01.md](docs/items/LW-01.md) Lane worktrees: helper that links private `scopes/<name>/` checkouts into a lane worktree — done 2026-11-25: PR #25; seen: `python3 scripts/link_scopes.py --worktree <fresh git worktree of main>` → 1 scope linked, worktree status clean, `--unlink` removed it

### Adapters (interfaces + first implementation)
- [ ] **AD-06** → [docs/items/AD-06.md](docs/items/AD-06.md) `Fetcher` adapter around own web scraping tool (document its API, JS rendering, PDF download) — partial 2026-11-25: interface + in-memory fake merged (PR #31); seen offline (pytest on main 361d2dd: 376 passed, 3 Postgres skips); unwired until the Phase 1 Crawl & snapshot row; needs a real implementation wrapping the scraping tool, streaming with a byte cap; `fetched_at` tz policy set by AD-08
- [x] **AD-08** → [docs/items/AD-08.md](docs/items/AD-08.md) `FetchResult.fetched_at` defaults to aware UTC; a naive value raises `FetchError` — done 2026-11-25: PR #38; seen: on main 0306b62 `FetchResult(...).fetched_at` → `2026-11-25 12:44:04+00:00`, pytest 392 passed, 3 Postgres skips; unwired like AD-06
- [x] **AD-09** → [docs/items/AD-09.md](docs/items/AD-09.md) `ParsedDocument.pages` must be a tuple; a list or generator raises `ParseError` — done 2026-11-25: PR #40; seen: pytest on main 2de0f63 (395 passed, 3 Postgres skips), list/generator tests raise `ParseError`; unwired like AD-07
- [ ] **AD-07** → [docs/items/AD-07.md](docs/items/AD-07.md) `Parser` adapter: HTML → Markdown; PDF → MinerU → Markdown with page numbers — partial 2026-11-25: interface + in-memory fake merged (PR #32); seen offline (pytest on main 361d2dd: 376 passed, 3 Postgres skips); unwired until the Phase 1 Parse & chunk row; needs real HTML and MinerU implementations with input-size and page caps; `pages` tuple type enforced by AD-09
- [ ] **AD-10** → [docs/items/AD-10.md](docs/items/AD-10.md) Stdlib `HtmlParser`: HTML → Markdown (ATX headings, lists, links, boilerplate tags dropped, `max_bytes` cap) — partial 2026-11-25: merged (PR #51); seen offline on main 3b0620e (pytest 480 passed, 5 Postgres skips; `parse_and_chunk` heading paths follow HTML headings); unwired until the CLI `parse` stage row; residuals: an unclosed dropped tag drops the rest of the body, `pre` whitespace collapses, a paragraph starting with `#` reads as a heading
- [ ] **AD-05** → [docs/items/AD-05.md](docs/items/AD-05.md) `Search` adapter — partial 2026-11-25: interface + in-memory fake merged (PR #28); seen offline (pytest on main: 349 passed, 3 Postgres skips); unwired until the Phase 1 Discover row; real backend waits on the (D) "Search backend for discovery" decision
- [ ] **AD-01** → [docs/items/AD-01.md](docs/items/AD-01.md) `LLM` adapter: OpenAI-compatible (vLLM) + Anthropic; JSON-schema/guided output — partial 2026-11-25: merged (PR #17); seen only with fake transports (pytest on main bc81dfa); unwired until the Screen/Assess rows; needs one live call per backend against a real gateway/key, and per-model cost rates (build_llm passes 0)
- [ ] **AD-03** → [docs/items/AD-03.md](docs/items/AD-03.md) `Embedder` + `Reranker` adapters (local GPU service) — partial 2026-11-25: merged (PR #23); seen only with fake transports (pytest on main, 292 passed); unwired until the Index and Assess retrieval rows; needs one live call each against the local embedding and rerank services (request/response shapes are assumptions)
- [ ] **AD-04** → [docs/items/AD-04.md](docs/items/AD-04.md) Adapter interfaces: `BusinessRegister`, `AccreditationRegister`, `ProfessionalRegister`, `TenderSource`, `PublicationSource` (ARCHITECTURE §10) — partial 2026-11-25: merged (PR #26); seen offline (pytest: 38 passed, 13 subtests); unwired until the Phase 1 Discover/Resolve/Enrich and Claim verification rows; needs one implementation per country/site, checked against each source's terms and API
- [ ] **AD-02** → [docs/items/AD-02.md](docs/items/AD-02.md) `source` catalogue table seeded from a scope pack's `sources.yaml` (roles, access class, licence) — partial 2026-11-25: merged (PR #20); seen on SQLite only (pytest on main e0aa9dd: 9 passed, 1 Postgres skip); unwired until the Discover row; needs a run against the compose Postgres

### Scope pack schema
- [x] **SP-01** → [docs/items/SP-01.md](docs/items/SP-01.md) Pydantic models: scope, segments, layers, vocabulary, sources, rubric, attributes, profiles, views ✔ 2026-11-25 (PR #10; seen: `load_scope_pack('scopes/_example')` on main dc3ae93 → 2 segments, 3 layers, 2 topics; unwired until `landscape validate`)
- [x] **SP-02** → [docs/items/SP-02.md](docs/items/SP-02.md) `landscape validate --scope ...` command ✔ 2026-11-25 (PR #13; seen: `uv run landscape validate --scope scopes/_example` on main 0789345 → scope example, 2 segments, 3 layers, 2 topics, 2 criteria, exit 0)
- [x] **SP-03** → [docs/items/SP-03.md](docs/items/SP-03.md) JSON Schema export for editor autocompletion ✔ 2026-11-25 (PR #16; seen: `uv run landscape schema --out <tmp>` and `--file rubric.yaml` on main bc81dfa → 7 draft 2020-12 schemas, exit 0)
- [x] **SP-04** → [docs/items/SP-04.md](docs/items/SP-04.md) `scopes/_example/` complete enough to validate (all files, tiny rubric, 3 seed entities) ✔ 2026-11-25 (PR #19; seen: `uv run landscape validate --scope scopes/_example` on main e0aa9dd → 2 segments, 3 layers, 2 topics, 2 criteria, exit 0; seed list unloaded until Discover/Resolve)

**Milestone M0:** `landscape validate` passes on `scopes/_example/`; adapters return data for one test URL and one PDF.

---

## Phase 1 — Vertical slice (one layer, ~20 entities)

- [ ] Discover: search + register channels + LLM-recall channel, all tagged by channel
- [ ] Resolve: domain/name normalisation, business-register ID lookup, GLEIF groups, merge suggestions
- [ ] Enrich: register financial data (revenue, size) where public
- [ ] Screen: in-scope + segment classification (local small model), review borderline
- [ ] Crawl: per-entity policy (max pages, priority paths, PDFs), robots.txt, rate limit, snapshots
  - [ ] **CR-01** → [docs/items/CR-01.md](docs/items/CR-01.md) `snapshot` table + `record_snapshot` immutable row per fetch (sha256 of body, aware-UTC `fetched_at`) — partial 2026-11-25: merged (PR #48); seen offline on main 3f684e8 (pytest 446 passed, 5 Postgres skips; returned id round-trips through `store_chunks`); unwired until the CLI `crawl` stage; needs migrations, entity FK, raw-bytes storage (waits on the (D) raw-storage row)
  - [x] **CH-04** → [docs/items/CH-04.md](docs/items/CH-04.md) `tests/test_chunk_store.py` ruff-formatted (layout only, AST-identical) — done 2026-11-25: PR #50; seen: `uv run ruff format --check core adapters tests` on main 3b0620e → 44 files already formatted
- [ ] Parse: HTML + MinerU, heading-aware chunks with provenance
  - [ ] **CH-01** → [docs/items/CH-01.md](docs/items/CH-01.md) `chunk_document` heading-aware Markdown chunker keeping page, heading path and char offsets — partial 2026-11-25: merged (PR #42); seen offline on main c9dde55 (pytest 412 passed, 3 Postgres skips; a 902-word doc → 3 chunks, paths (A),(A,B),(A,B), 600+360 words with 60-word overlap); unwired until the Phase 1 Parse & chunk row; needs wiring into Parse & chunk and `chunk` row persistence
  - [ ] **CH-02** → [docs/items/CH-02.md](docs/items/CH-02.md) `parse_and_chunk` stage function: parse once with a `Parser`, yield `chunk_document` chunks; eager `ValueError`, `ParseError` propagates — partial 2026-11-25: merged (PR #44); seen offline on main 29f66fa (pytest 423 passed, 3 Postgres skips); unwired until the Parse & chunk persistence row (CLI `parse` stage storing `chunk` rows)
  - [ ] **CH-03** → [docs/items/CH-03.md](docs/items/CH-03.md) `chunk` table + `store_chunks` idempotent chunk row persistence: delete-then-insert per snapshot, batched, one transaction — partial 2026-11-25: merged (PR #46); seen offline on main 2381086 (pytest 433 passed, 4 Postgres skips; `parse_and_chunk` → `store_chunks` round-trips all columns); unwired until the CLI `parse` stage row; needs migrations, snapshot FK, language detection
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
