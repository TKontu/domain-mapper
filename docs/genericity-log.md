# Genericity log (GP-01, GP-02)

`tests/test_core_genericity.py` loads the domain terms of `scopes/_example/` through
`core/scope_pack.py` (segment, topic, criterion, vocabulary entry, attribute and profile ids, plus
vocabulary terms) and scans `core/**/*.py` and `app/**/*.py` (GP-02) for them, case-insensitive and word-bounded. Every hit
must be listed in `tests/fixtures/core_genericity_allowlist.toml`; unlisted hits and stale entries
fail. This table must agree with the allowlist entry for entry.

Verdicts: `generic-keep` (a generic word that merely coincides with an `_example` term) or
`leaked-follow-up` (domain logic in core; becomes a Phase 5 "Refactor any leaked domain logic" row).

| File | Term | Verdict | Note |
| --- | --- | --- | --- |
| `app/cli.py` | `terms` | generic-keep | local name for serialised `presence_terms` |
| `core/keyword_search.py` | `terms` | generic-keep | BM25 query terms |
| `core/presence_extract.py` | `terms` | generic-keep | `presence_terms` schema field |
| `core/retrieve.py` | `terms` | generic-keep | docstring on tokenised query terms |
| `core/scope_pack.py` | `terms` | generic-keep | vocabulary / presence terms schema fields |

No leaked domain logic found at round 175 (core/) or round 176 (app/).
