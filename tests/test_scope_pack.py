import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from core.scope_pack import ScopePack, ScopePackError, load_scope_pack

EXAMPLE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


class ExamplePackTest(unittest.TestCase):
    def test_example_loads(self) -> None:
        pack = load_scope_pack(EXAMPLE)
        self.assertIsInstance(pack, ScopePack)
        self.assertEqual(len(pack.scope.segments), 2)
        self.assertEqual(len(pack.scope.layers), 3)
        self.assertEqual(len(pack.rubric.topics), 2)
        for section in (
            pack.vocabulary.entries,
            pack.sources.sources,
            pack.attributes.attributes,
            pack.profiles.profiles,
            pack.views.views,
        ):
            self.assertTrue(section)

    def test_example_cross_references_resolve(self) -> None:
        pack = load_scope_pack(EXAMPLE)
        segments = {s.id for s in pack.scope.segments}
        countries = {c for la in pack.scope.layers for c in la.countries}
        weighted = {t.id for t in pack.rubric.topics} | {
            c.id for t in pack.rubric.topics for c in t.criteria
        }
        for e in pack.vocabulary.entries:
            self.assertLessEqual(set(e.segments), segments)
        for a in pack.attributes.attributes:
            self.assertLessEqual(set(a.applies_to), segments)
        for p in pack.profiles.profiles:
            self.assertTrue(p.weights)
            self.assertLessEqual(set(p.weights), weighted)
        for s in pack.sources.sources:
            self.assertLessEqual(set(s.countries), countries)

    def test_example_seed_entities(self) -> None:
        import yaml

        data = yaml.safe_load((EXAMPLE / "seed" / "entities.yaml").read_text(encoding="utf-8"))
        entities = data["entities"]
        self.assertEqual(len(entities), 3)
        segments = {s.id for s in load_scope_pack(EXAMPLE).scope.segments}
        self.assertEqual(len({e["id"] for e in entities}), 3)
        for e in entities:
            self.assertIn(e["segment"], segments)
            self.assertIn(e["country"], {"XA", "XB", "XC"})
            self.assertTrue(e["website"].split("/")[2].endswith(".example"))


class MalformedPackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        for name in ("scope.yaml", "rubric.yaml"):
            shutil.copy(EXAMPLE / name, self.dir / name)

    def edit(self, name: str, old: str, new: str) -> None:
        p = self.dir / name
        text = p.read_text()
        self.assertIn(old, text)
        p.write_text(text.replace(old, new, 1))

    def assert_fails(self, file: str, fragment: str) -> None:
        with self.assertRaises(ScopePackError) as ctx:
            load_scope_pack(self.dir)
        self.assertEqual(ctx.exception.file, file)
        self.assertIn(file, str(ctx.exception))
        self.assertIn(fragment, str(ctx.exception))

    def test_unknown_segment_in_applies_to(self) -> None:
        self.edit("rubric.yaml", "applies_to: [agency]", "applies_to: [nobody]")
        self.assert_fails("rubric.yaml", "nobody")

    def test_unknown_segment_in_topic_segments(self) -> None:
        self.edit("rubric.yaml", "label: Terms & commercial", "label: T\n    segments: [ghost]")
        self.assert_fails("rubric.yaml", "ghost")

    def test_unknown_status_id(self) -> None:
        self.edit("rubric.yaml", "weak:     Terms only", "meh:     Terms only")
        self.assert_fails("rubric.yaml", "meh")

    def test_duplicate_id(self) -> None:
        self.edit("scope.yaml", "id: freelancer_network", "id: agency")
        self.assert_fails("scope.yaml", "duplicate segment")

    def test_extra_key(self) -> None:
        self.edit("scope.yaml", "purpose: internal_evaluation", "purpose: x\nsurprise: 1")
        self.assert_fails("scope.yaml", "surprise")

    def test_missing_required_file(self) -> None:
        (self.dir / "rubric.yaml").unlink()
        self.assert_fails("rubric.yaml", "missing")

    def test_invalid_optional_file(self) -> None:
        (self.dir / "views.yaml").write_text("views:\n  - {id: map}\n")
        self.assert_fails("views.yaml", "label")

    def test_valid_optional_file(self) -> None:
        (self.dir / "profiles.yaml").write_text(
            "profiles:\n  - {id: p, label: P, weights: {terms: 1.0}}\n"
        )
        self.assertEqual(len(load_scope_pack(self.dir).profiles.profiles), 1)


if __name__ == "__main__":
    unittest.main()


class CrawlPolicyTest(unittest.TestCase):
    """``scope.yaml``'s optional ``crawl`` section (CR-05)."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        for name in ("scope.yaml", "rubric.yaml"):
            shutil.copy(EXAMPLE / name, self.dir / name)
        path = self.dir / "scope.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data.pop("crawl", None)  # start from a pack without the section
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def with_crawl(self, body: str) -> None:
        path = self.dir / "scope.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\n" + body, encoding="utf-8")

    def assert_policy_fails(self, body: str, fragment: str) -> None:
        self.with_crawl(body)
        with self.assertRaises(ScopePackError) as ctx:
            load_scope_pack(self.dir)
        self.assertEqual(ctx.exception.file, "scope.yaml")
        self.assertIn(fragment, str(ctx.exception))

    def test_default_policy_without_section(self) -> None:
        policy = load_scope_pack(self.dir).scope.crawl
        self.assertEqual(policy.max_pages, 1)
        self.assertEqual(policy.priority_paths, [])

    def test_valid_policy(self) -> None:
        self.with_crawl("crawl:\n  max_pages: 3\n  priority_paths: [/a, /b]\n")
        policy = load_scope_pack(self.dir).scope.crawl
        self.assertEqual(policy.max_pages, 3)
        self.assertEqual(policy.priority_paths, ["/a", "/b"])

    def test_root_priority_path_rejected(self) -> None:
        self.assert_policy_fails('crawl: {priority_paths: ["/"]}', "crawl.priority_paths")

    def test_discover_links_defaults_false_and_parses(self) -> None:
        self.assertFalse(load_scope_pack(self.dir).scope.crawl.discover_links)
        self.with_crawl("crawl: {discover_links: true}")
        self.assertTrue(load_scope_pack(self.dir).scope.crawl.discover_links)

    def test_discover_links_rejects_non_bool(self) -> None:
        self.assert_policy_fails("crawl: {discover_links: maybe}", "crawl.discover_links")

    def test_max_pages_zero_rejected(self) -> None:
        self.assert_policy_fails("crawl: {max_pages: 0}", "crawl.max_pages")

    def test_max_pages_above_fifty_rejected(self) -> None:
        self.assert_policy_fails("crawl: {max_pages: 51}", "crawl.max_pages")

    def test_path_without_leading_slash_rejected(self) -> None:
        self.assert_policy_fails("crawl: {priority_paths: [about]}", "crawl.priority_paths")

    def test_path_with_query_rejected(self) -> None:
        self.assert_policy_fails('crawl: {priority_paths: ["/about?x=1"]}', "crawl.priority_paths")

    def test_path_with_fragment_rejected(self) -> None:
        self.assert_policy_fails('crawl: {priority_paths: ["/about#top"]}', "crawl.priority_paths")

    def test_path_with_host_rejected(self) -> None:
        self.assert_policy_fails(
            'crawl: {priority_paths: ["//example.invalid/x"]}', "crawl.priority_paths"
        )

    def test_duplicate_path_rejected(self) -> None:
        self.assert_policy_fails("crawl: {priority_paths: [/a, /a]}", "crawl.priority_paths")


class JsonSchemaExportTest(unittest.TestCase):
    """``scope_pack_json_schemas`` (SP-03)."""

    FILES = {
        "scope.yaml",
        "rubric.yaml",
        "vocabulary.yaml",
        "sources.yaml",
        "attributes.yaml",
        "profiles.yaml",
        "views.yaml",
    }

    def setUp(self) -> None:
        from core.scope_pack import scope_pack_json_schemas

        self.schemas = scope_pack_json_schemas()

    def test_one_schema_per_pack_file(self) -> None:
        self.assertEqual(set(self.schemas), self.FILES)
        for name, schema in self.schemas.items():
            with self.subTest(name):
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(schema["type"], "object")
                self.assertIsInstance(schema["properties"], dict)

    def test_schemas_forbid_extra_keys_everywhere(self) -> None:
        for name, schema in self.schemas.items():
            objects = [schema, *schema.get("$defs", {}).values()]
            for obj in objects:
                if obj.get("type") == "object" and "properties" in obj:
                    with self.subTest(name, title=obj.get("title")):
                        self.assertIs(obj["additionalProperties"], False)

    def test_example_files_match_exported_schema(self) -> None:
        import yaml

        from core.scope_pack import _FILE_MODELS

        for path in sorted(EXAMPLE.glob("*.yaml")):
            with self.subTest(path.name):
                schema = self.schemas[path.name]
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                self.assertLessEqual(set(data), set(schema["properties"]))
                self.assertLessEqual(set(schema.get("required", [])), set(data))
                model = _FILE_MODELS[path.name]
                required = {n for n, f in model.model_fields.items() if f.is_required()}
                self.assertEqual(set(schema.get("required", [])), required)
