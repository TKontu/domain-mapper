# Domain Mapper

> A domain-agnostic pipeline for **discovering, profiling, assessing and comparing the providers in a market**. Everything domain-specific lives in a pluggable **scope pack**.

Inspired by "fine print" landscape sites that score every provider in a market on the same rubric and show the result as a per-provider mosaic: every provider gets assessed against the **same criteria**, every cell is backed by **cited evidence**, and the whole landscape can be viewed as a **map, mosaic, side-by-side comparison and methodology page**.

---

## 1. What it produces

For a configured scope, the engine produces:

| Output | Description |
|---|---|
| **Provider registry** | Deduplicated list of in-scope organisations, with legal entities, groups/subsidiaries, sites and the markets they serve |
| **Assessment mosaic** | Every provider × every rubric criterion → a status (e.g. Strong / Standard / Weak / Red flag / Not stated / N/A) with a verbatim quote, source URL and snapshot date |
| **Quantitative attributes** | Headcount, founding year, number of certified services, sites, stated turnaround, distance to customer, etc. |
| **Market statistics** | Counts, capability coverage, gaps and concentration per segment per market layer |
| **Qualitative profiles** | Short cited summaries per provider and per segment/market ("who does what, where, and how well documented") |
| **Views** | Map · Mosaic · Compare · Capability matrix · Market overview · Method |
| **Exports** | Parquet / CSV / JSON for analysis elsewhere |

---

## 2. Scope packs

A scope pack (YAML) declares one landscape:

- **Segments** — categories of provider; a provider can belong to several.
- **Market layers** — nested sets of countries, e.g. *home country ⊂ region ⊂ continent ⊂ global*. A provider belongs to a layer if it **can serve that market** (HQ, site, field team, sales office or documented remote service), not only by HQ location. Discovery and assessment depth usually decrease outward.
- **Vocabulary** — search terms per segment, capability and language.
- **Sources** — discovery channels and authoritative registers used to verify claims.
- **Rubric** — topics (mosaic rows), criteria (cells), a status scale with its own polarity, and evidence rules.
- **Attributes, weight profiles and view configuration.**

The public repository ships only [`scopes/_example/`](scopes/_example/), a small illustrative pack. Real scope packs are private checkouts under `scopes/<name>/` and are gitignored here (see [scopes/README.md](scopes/README.md)).

---

## 3. How it works (one-paragraph version)

A **scope pack** defines segments, geography, search vocabulary per language, authoritative sources and the rubric. The engine **discovers** candidates through multiple independent channels (search, public registers, association lists, tender awards, LLM recall), **resolves** them into canonical entities, **screens** for relevance, **crawls** websites and documents (web scraper + PDF parser), **indexes** chunks (embeddings + BM25 + reranker), then **assesses** each rubric criterion with a local extractor LLM, a second verifier model and deterministic quote checks, escalating disagreements to a cloud LLM and finally to a human. Claims are **cross-checked against authoritative registers** where possible (e.g. a claimed certification vs. the issuing body's public register). Results are aggregated, versioned and published as a static site.

See [ARCHITECTURE.md](ARCHITECTURE.md) for details and [todo.md](todo.md) for the engine plan.

---

## 4. Design principles

1. **Domain-agnostic core, domain-specific scope packs.** No domain logic or domain words in the core. A second scope must run without core code changes.
2. **Evidence or it didn't happen.** Every status links to a verbatim quote + URL + snapshot. No evidence → "Not stated", never a guess.
3. **Verify claims, don't just extract them.** Evidence hierarchy: authoritative register > provider's own formal documents (terms, certificates, scopes) > provider's marketing pages > third-party mentions.
4. **Local first, cloud by exception.** Bulk work on local models (vLLM, embeddings, reranker); a cloud LLM for rubric design, adjudication and synthesis.
5. **Reproducible and versioned.** Every assessment records rubric version, prompt version, model, snapshot. Re-runs reuse cache keyed on content hash.
6. **No false precision.** Show the mosaic and coverage, not one magic score. Weighted scores exist only as user-chosen *profiles*.
7. **Respectful collection.** robots.txt, rate limits, site terms; organisation-level data by default (person-level only where a public register is itself person-level); no scraping of login-walled social networks; no profiling of individuals.
8. **Official free sources before paid ones.** Business registers, certification registers, tenders and GLEIF are often free; paid firmographics only to fill proven gaps.

---

## 5. Toolbox

| Need | Tool |
|---|---|
| Crawling | Own web scraping tool (behind a `Fetcher` adapter) |
| PDF / brochure / terms parsing | MinerU |
| Search for discovery | Self-hosted SearXNG and/or a search API (pluggable) |
| Registers & public data | Business registers, certification/accreditation registers, GLEIF, public tenders, OpenAlex, professional registers — chosen per scope pack |
| Embeddings & reranking | Local multilingual models (e.g. bge-m3 + bge-reranker-v2-m3 class) |
| Bulk extraction & screening | Local vLLM — extractor/verifier pair from different model families; larger local model for harder items |
| Adjudication, rubric drafting, synthesis | Claude (cloud) |
| Storage | PostgreSQL + pgvector, object storage for raw snapshots, DuckDB/Parquet for analytics |
| Frontend | Static site (map, mosaic, compare, method) built from JSON exports |

---

## 6. Quick start (target CLI)

```bash
# 1. pick a scope pack
landscape init --scope scopes/_example

# 2. run stages (each is idempotent and resumable)
landscape discover   --layer home
landscape resolve
landscape screen
landscape crawl      --layer home
landscape parse
landscape index
landscape assess     --layer home
landscape verify-claims
landscape aggregate
landscape review               # open human review queue
landscape publish --out site/

# or everything for a layer
landscape run --layer home
```

**Today (skeleton, PF-04):** `uv sync` installs the `landscape` command. Every stage
(`discover resolve screen crawl parse index assess attributes verify aggregate synthesize publish`)
accepts `--scope PATH --layer TEXT --limit N --dry-run`. With `--dry-run` it prints the resolved
plan and settings (never the API key) and exits 0; without it, a stage exits 1 with "not
implemented yet". A missing scope directory or an invalid `DM_*` setting exits 2.

```bash
uv run landscape discover --scope scopes/_example --layer home --limit 5 --dry-run
```

---

## 7. Repository layout (target)

```
domain-mapper/
├── core/            # domain-agnostic engine (stages, data model, LLM routing)
├── adapters/        # fetcher, search, registers, LLM backends
├── scopes/
│   ├── _example/    # public illustrative scope pack (genericity proof)
│   └── <name>/      # private scope checkouts (gitignored)
├── web/             # static frontend
├── eval/            # metrics and calibration (gold sets live in each scope pack)
├── docs/            # decision records, work items
├── README.md
├── ARCHITECTURE.md
└── todo.md          # engine backlog
```

---

## 8. Development workflow

Development runs in Reconcile-loop rounds (plan → execute → review → merge → reconcile → handoff).
Agents start from [AGENTS.md](AGENTS.md); the protocol is [SPEC.md](SPEC.md) and the round commands
are in `.agent/commands/`. Merges go through `scripts/merge_guard.py` under decision D-002
([docs/decisions.md](docs/decisions.md)). The local test gate is primary; CI is a minimal backstop.

Runtime settings come from `DM_*` environment variables, optionally seeded by a gitignored `.env`
file (copy `.env.example`; placeholders only). `core.config.load_settings()` returns a frozen,
validated settings object and fails fast, naming the variable, on malformed values.

## 9. Status

Planning. Architecture and engine backlog drafted; first item: [PF-01](docs/items/PF-01.md) (Python project skeleton).
