"""`python -m app.cli` must run the CLI, not exit silently."""

import subprocess
import sys


def test_python_m_app_cli_help_prints_usage() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "app.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Usage" in proc.stdout
