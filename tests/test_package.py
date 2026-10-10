"""Smoke tests: the package imports and the command-line entry point works."""

import re
import subprocess
import sys

import gestureflow
from gestureflow.__main__ import main


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", gestureflow.__version__)


def test_check_returns_success(capsys):
    assert main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "installed correctly" in out
    assert "Hand model:" in out


def test_python_dash_m_runs():
    # Runs exactly what a user types, in a separate process.
    result = subprocess.run(
        [sys.executable, "-m", "gestureflow", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert gestureflow.__version__ in result.stdout
