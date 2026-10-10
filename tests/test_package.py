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


def test_train_from_terminal(monkeypatch, tmp_path, capsys):
    from gestureflow.storage.samples import save_sample
    from tests.conftest import make_pose_sample

    monkeypatch.setenv("GESTUREFLOW_USER_DATA_DIR", str(tmp_path))
    for i in range(3):
        save_sample(make_pose_sample("palm", "open", i))
        save_sample(make_pose_sample("neutral", "relaxed", 10 + i))
    assert main(["--train"]) == 0
    out = capsys.readouterr().out
    assert "Poses: 6 of 6 held-out examples correct" in out
    assert "Pose confusion matrix" in out
    assert "only 3 pose examples" in out
    assert (tmp_path / "trained" / "gesture_model.joblib").exists()


def test_train_from_terminal_without_examples(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GESTUREFLOW_USER_DATA_DIR", str(tmp_path))
    assert main(["--train"]) == 1
    assert "Not enough to train" in capsys.readouterr().err


def test_delete_examples_from_terminal(monkeypatch, tmp_path, capsys):
    from gestureflow.storage.samples import count_by_kind, save_sample
    from tests.conftest import make_pose_sample, make_swipe_sample

    monkeypatch.setenv("GESTUREFLOW_USER_DATA_DIR", str(tmp_path))
    save_sample(make_pose_sample("pause", "open", 1))
    save_sample(make_swipe_sample("pause", 0.3, 2))
    assert main(["--delete-examples", "pause", "--kind", "pose"]) == 0
    assert "Deleted 1 pose example of 'pause'." in capsys.readouterr().out
    assert count_by_kind() == {"pause": {"dynamic": 1}}
    assert main(["--delete-examples", "pause"]) == 2  # --kind is required
