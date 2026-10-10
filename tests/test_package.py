"""Smoke tests: the package imports and the entry point runs."""

import re
import subprocess
import sys

import gestureflow
from gestureflow.__main__ import main


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", gestureflow.__version__)


def test_main_returns_success(capsys):
    assert main([]) == 0
    assert "installed correctly" in capsys.readouterr().out


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
