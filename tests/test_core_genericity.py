"""GP-01: core/ carries no `_example` domain words unless allowlisted (ratchet)."""

from __future__ import annotations

import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from core.scope_pack import load_scope_pack

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "scopes" / "_example"
ALLOWLIST = ROOT / "tests" / "fixtures" / "core_genericity_allowlist.toml"
LOG = ROOT / "docs" / "genericity-log.md"
VERDICTS = {"generic-keep", "leaked-follow-up"}


def _example_terms(pack_dir: Path = EXAMPLE) -> set[str]:
    pack = load_scope_pack(pack_dir)
    terms: set[str] = set()
    terms.update(s.id for s in pack.scope.segments)
    for topic in pack.rubric.topics:
        terms.add(topic.id)
        terms.update(c.id for c in topic.criteria)
    for entry in pack.vocabulary.entries:
        terms.add(entry.id)
        for words in entry.terms.values():
            terms.update(words)
    terms.update(a.id for a in pack.attributes.attributes)
    terms.update(p.id for p in pack.profiles.profiles)
    return {t.lower() for t in terms if t.strip()}


def _scan(root: Path, terms: set[str]) -> set[tuple[str, str]]:
    """Return (root-relative file, term) for each word-bounded, case-insensitive hit."""
    pats = {t: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", re.I) for t in terms}
    hits: set[tuple[str, str]] = set()
    for path in sorted((root / "core").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        hits.update((rel, t) for t, p in pats.items() if p.search(src))
    return hits


def _load_allowlist(path: Path) -> dict[tuple[str, str], str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for row in data.get("hit", []):
        assert row["verdict"] in VERDICTS, row
        assert str(row.get("reason", "")).strip(), row
        out[(row["file"], row["term"].lower())] = row["verdict"]
    return out


def _problems(hits: set[tuple[str, str]], allowed: set[tuple[str, str]]) -> list[str]:
    unlisted = [f"unlisted hit: {f} -> {t!r}" for f, t in sorted(hits - allowed)]
    stale = [f"stale allowlist entry: {f} -> {t!r}" for f, t in sorted(allowed - hits)]
    return unlisted + stale


def _log_entries(path: Path) -> dict[tuple[str, str], str]:
    row = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([a-z-]+)\s*\|")
    out: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = row.match(line)
        if m:
            out[(m.group(1), m.group(2).lower())] = m.group(3)
    return out


class RealTreeTest(unittest.TestCase):
    def test_terms_loaded(self) -> None:
        self.assertIn("agency", _example_terms())

    def test_core_matches_allowlist(self) -> None:
        problems = _problems(_scan(ROOT, _example_terms()), set(_load_allowlist(ALLOWLIST)))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_log_agrees_with_allowlist(self) -> None:
        self.assertEqual(_log_entries(LOG), _load_allowlist(ALLOWLIST))


class NegativeCaseTest(unittest.TestCase):
    def _tree(self, src: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "core").mkdir()
        (root / "core" / "fake.py").write_text(src)
        return root

    def test_unlisted_segment_id_fails(self) -> None:
        root = self._tree('KIND = "Freelancer_Network"\n')
        problems = _problems(_scan(root, _example_terms()), set())
        self.assertEqual(problems, ["unlisted hit: core/fake.py -> 'freelancer_network'"])

    def test_listed_hit_passes(self) -> None:
        root = self._tree('KIND = "freelancer_network"\n')
        allowed = {("core/fake.py", "freelancer_network")}
        self.assertEqual(_problems(_scan(root, _example_terms()), allowed), [])

    def test_stale_entry_fails(self) -> None:
        root = self._tree("x = 1\n")
        problems = _problems(_scan(root, _example_terms()), {("core/fake.py", "agency")})
        self.assertEqual(problems, ["stale allowlist entry: core/fake.py -> 'agency'"])

    def test_word_bounded(self) -> None:
        root = self._tree("agencyx = my_agency = 1\n")
        self.assertEqual(_scan(root, {"agency"}), set())


if __name__ == "__main__":
    unittest.main()
