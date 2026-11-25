import shutil
import tempfile
import unittest
from pathlib import Path

from core.scope_pack import ScopePack, ScopePackError, load_scope_pack

EXAMPLE = Path(__file__).resolve().parent.parent / "scopes" / "_example"


class ExamplePackTest(unittest.TestCase):
    def test_example_loads(self) -> None:
        pack = load_scope_pack(EXAMPLE)
        self.assertIsInstance(pack, ScopePack)
        self.assertEqual(len(pack.scope.segments), 2)
        self.assertEqual(len(pack.scope.layers), 3)
        self.assertEqual(len(pack.rubric.topics), 2)
        self.assertEqual(pack.sources.sources, [])
        self.assertEqual(pack.views.views, [])


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
