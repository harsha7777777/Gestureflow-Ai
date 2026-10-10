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


def test_download_model_skips_verified_file(monkeypatch, tmp_path, capsys):
    from gestureflow.detection import model_file

    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"model")
    monkeypatch.setenv("GESTUREFLOW_MODEL_PATH", str(model))
    monkeypatch.setattr(model_file, "MODEL_SHA256", model_file.sha256_of(model))
    assert main(["--download-model"]) == 0
    assert "already downloaded" in capsys.readouterr().out
