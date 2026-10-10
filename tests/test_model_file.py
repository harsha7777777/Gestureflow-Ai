import hashlib

import pytest

from gestureflow.detection import model_file
from gestureflow.detection.model_file import ModelDownloadError, download_model


@pytest.fixture
def fake_remote(tmp_path):
    """A local file served through a file:// URL, standing in for the real download."""
    source = tmp_path / "remote.task"
    source.write_bytes(b"pretend this is a model")
    return source.as_uri(), hashlib.sha256(source.read_bytes()).hexdigest()


def test_download_verifies_and_saves(tmp_path, fake_remote):
    url, digest = fake_remote
    dest = tmp_path / "models" / "hand_landmarker.task"
    assert download_model(dest, url=url, expected_sha256=digest) == dest
    assert dest.read_bytes() == b"pretend this is a model"


def test_download_rejects_wrong_hash(tmp_path, fake_remote):
    url, _ = fake_remote
    dest = tmp_path / "models" / "hand_landmarker.task"
    with pytest.raises(ModelDownloadError, match="verification"):
        download_model(dest, url=url, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part")), "temporary file left behind"


def test_download_reports_unreachable_url(tmp_path):
    dest = tmp_path / "hand_landmarker.task"
    with pytest.raises(ModelDownloadError, match="Could not download"):
        download_model(dest, url=(tmp_path / "missing").as_uri(), expected_sha256="0" * 64)


def test_model_path_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREFLOW_MODEL_PATH", str(tmp_path / "custom.task"))
    assert model_file.model_path() == tmp_path / "custom.task"
    assert not model_file.is_model_present()
