# New scope guide

This guide takes you from an empty directory to a scope pack that `landscape validate --scope <dir>`
accepts. The models that define every field live in `core/scope_pack.py`; the complete public
reference pack is [`scopes/_example/`](../scopes/_example/). See also
[ARCHITECTURE.md §3](../ARCHITECTURE.md#3-scope-pack).

## 1. Keep a real pack private

Only `scopes/_example/` is tracked in this repository. A real pack is its own private repository,
cloned to `scopes/<name>/`, which is gitignored here. Never copy its content, names or findings into
this repository, its commits or PRs (decision D-003). Details: [scopes/README.md](../scopes/README.md).

```sh
mkdir scopes/<name> && cd scopes/<name> && git init
```

## 2. Pack files

A pack is a directory of YAML files. Two are required; the rest are optional and default to empty.
Every model forbids unknown keys, and ids must be unique within their file.

| File | Required | Purpose |
| --- | --- | --- |
| `scope.yaml` | yes | Identity, perspective, segments, market layers, languages, crawl policy. |
| `rubric.yaml` | yes | Status scale and topics (mosaic rows) holding criteria (cells). |
| `vocabulary.yaml` | no | Search terms per segment and language for discovery queries. |
| `sources.yaml` | no | Discovery channels, registers and verification sources. |
| `attributes.yaml` | no | Structured attributes (int, float, enum, bool, date) per entity. |
| `profiles.yaml` | no | Weight profiles keyed by topic or criterion id. |
| `views.yaml` | no | Output view configuration (`id`, `label`, `kind`, `options`). |
| `seed/` | no | Seed entities; illustrative only, not yet loaded by the engine. |

### `scope.yaml` (model `Scope`)

Required: `id`, `version`, `title`, `perspective`, `segments` (at least one), `layers` (at least
one). Optional: `purpose`, `languages`, `crawl`, `presence_terms`.

- Segment (`Segment`): `id`, `label`, `definition`; optional `subtypes`.
- Layer (`Layer`): `id`, `countries` (at least one), `discovery_target`
  (`exhaustive` | `near_exhaustive` | `notable`), `rubric_depth` (`full` | `core`); optional `label`.
- `crawl` (`CrawlPolicy`): `max_pages` (1–50, default 1), `priority_paths` (absolute paths),
  `discover_links` (default false).
- `presence_terms`: uppercase two-letter country code → non-empty list of terms.

Minimal file:

```yaml scope.yaml
id: my_scope
version: 0.0.1
title: My landscape
perspective: Who is looking at this landscape, and why.
segments:
  - id: segment_a
    label: Segment A
    definition: What makes an entity belong to segment A.
layers:
  - id: home
    countries: [XA]
    discovery_target: exhaustive
    rubric_depth: full
```

### `rubric.yaml` (model `Rubric`)

Required: `status_scale` (at least one `Status`) and `topics` (at least one `Topic`).

- Status: `id`, `label`, `value` (number or `null`), `color`.
- Topic: `id`, `label`, `criteria` (at least one); optional `segments` (empty means all).
- Criterion: `id`, `question`, `applies_to` (at least one segment id), `statuses` (status id →
  description); optional `retrieval_hints`, `verify_with`, `evidence_min` (default 1), `core`.

Segment and status references must name ids declared in `scope.yaml` and `status_scale`.

Minimal file:

```yaml rubric.yaml
status_scale:
  - {id: meets, label: Meets, value: 1, color: green}
  - {id: silent, label: Not stated, value: null, color: white}
topics:
  - id: topic_a
    label: Topic A
    criteria:
      - id: criterion_a
        question: Does the entity publish the thing this criterion asks about?
        applies_to: [segment_a]
        statuses:
          meets: Published and current.
```

With these two files the pack validates. Add optional files as you need them.

### Optional files

- `vocabulary.yaml` — `entries`: each `id`, `terms` (language → list of terms), optional `segments`.
- `sources.yaml` — `sources`: each `id`, `role` (`discovery` | `register` | `verification`),
  `adapter`; optional `countries`, `access`, `licence`, `params`.
- `attributes.yaml` — `attributes`: each `id`, `label`, `type`; optional `unit`, `values` (for
  enum), `applies_to`, `query_terms`.
- `profiles.yaml` — `profiles`: each `id`, `label`, `weights` (topic or criterion id → number).
- `views.yaml` — `views`: each `id`, `label`, `kind`; optional `options`.

Copy the shape from the matching file in `scopes/_example/`, then replace its content with yours.

## 3. Check it offline

Both commands run without network or database:

```sh
landscape validate --scope scopes/<name>    # exit 0 and a summary; 1 invalid; 2 missing
landscape schema --out .schemas             # JSON Schemas for editor autocompletion
landscape schema --file rubric.yaml         # print one file's schema
```

`landscape validate` prints the first error as `<file>: <field path>: <message>`.
