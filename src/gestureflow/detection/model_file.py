"""Locating and downloading the MediaPipe hand landmark model.

The model (about 7.8 MB, Apache 2.0 license) is published by Google. It is not
stored in this repository; it is downloaded once into the models/ folder:

    python -m gestureflow --download-model

The download is checked against a known SHA-256 hash, so a corrupted or
tampered file is rejected. This is the only network access the app makes.
"""

import hashlib
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path

from gestureflow.storage.paths import models_dir

MODEL_FILENAME = "hand_landmarker.task"
# Version "1" of the float16 model, pinned (not "latest") so everyone gets the same file.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


class ModelDownloadError(RuntimeError):
    """The model could not be downloaded or failed verification."""


def model_path() -> Path:
    """Where the model is expected. Override with GESTUREFLOW_MODEL_PATH."""
    override = os.environ.get("GESTUREFLOW_MODEL_PATH")
    return Path(override) if override else models_dir() / MODEL_FILENAME


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_model_present() -> bool:
    return model_path().is_file()


def _ssl_context() -> ssl.SSLContext:
    # Python from python.org on macOS ships without root certificates, which makes
    # HTTPS downloads fail. certifi (already installed as a MediaPipe dependency)
    # provides them.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_model(
    destination: Path | None = None,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
) -> Path:
    """Download the model to `destination` and verify its hash. Returns the path."""
    destination = destination or model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file first, so an interrupted download never leaves
    # a half-written model where the app would try to load it.
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response,
        ):
            while chunk := response.read(1 << 16):
                out.write(chunk)
        actual = sha256_of(tmp_path)
        if actual != expected_sha256:
            raise ModelDownloadError(
                f"Downloaded model failed verification (SHA-256 {actual}, "
                f"expected {expected_sha256}). Please try again."
            )
        tmp_path.replace(destination)
        return destination
    except OSError as exc:  # network errors (URLError) are OSErrors too
        raise ModelDownloadError(
            f"Could not download the hand model from {url}: {exc}. "
            "Check your internet connection and try again."
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
