"""RT-01: CLI engines are built through ``core.runs.create_db_engine``; temp SQLite, offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

import app.cli as cli
from core.runs import create_db_engine

EXAMPLE_SCOPE = str(Path(__file__).resolve().parent.parent / "scopes" / "_example")


class CliEngineFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DM_")}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.db_url = f"sqlite:///{os.getcwd()}/dm.sqlite"
        os.environ["DM_DATABASE_URL"] = self.db_url

    def test_resolve_builds_engine_via_create_db_engine(self) -> None:
        spy = mock.Mock(wraps=create_db_engine)
        with mock.patch.object(cli, "create_db_engine", spy, create=True):
            result = CliRunner().invoke(cli.app, ["resolve", "--scope", EXAMPLE_SCOPE])
        self.assertEqual(result.exit_code, 0, result.output)
        spy.assert_called_once_with(self.db_url)

    def test_cli_does_not_import_raw_create_engine(self) -> None:
        self.assertFalse(hasattr(cli, "create_engine"))


if __name__ == "__main__":
    unittest.main()
