"""NS-01: docs/new-scope.md stays in step with the scope-pack models and the CLI."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from core.scope_pack import load_scope_pack, scope_pack_json_schemas

_ROOT = Path(__file__).resolve().parents[1]
_GUIDE = _ROOT / "docs" / "new-scope.md"
_BLOCK = re.compile(r"^```yaml ([\w.]+\.yaml)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _blocks() -> dict[str, str]:
    return {name: body for name, body in _BLOCK.findall(_GUIDE.read_text(encoding="utf-8"))}


def test_guide_minimal_pack_validates(tmp_path: Path) -> None:
    blocks = _blocks()
    assert {"scope.yaml", "rubric.yaml"} <= set(blocks)
    for name, body in blocks.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    load_scope_pack(tmp_path)
    result = CliRunner().invoke(app, ["validate", "--scope", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_guide_names_only_real_files_and_commands() -> None:
    text = _GUIDE.read_text(encoding="utf-8")
    example = _ROOT / "scopes" / "_example"
    for name in set(re.findall(r"`(\w+\.yaml)`", text)):
        assert (example / name).is_file() or name in scope_pack_json_schemas(), name
    assert (example / "seed").is_dir()
    commands = {c.name for c in app.registered_commands}
    for cmd in set(re.findall(r"landscape ([a-z][\w-]*)", text)):
        assert cmd in commands, cmd


def test_guide_is_domain_neutral_and_linked() -> None:
    text = _GUIDE.read_text(encoding="utf-8").lower()
    for word in ("translat", "iso 17100", "agency"):
        assert word not in text, word
    assert "docs/new-scope.md" in (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "../docs/new-scope.md" in (_ROOT / "scopes" / "README.md").read_text(encoding="utf-8")
