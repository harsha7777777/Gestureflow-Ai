"""Checks that the pinned dependencies are installed and do not conflict.

These tests need `pip install -e ".[dev]"` (or requirements.txt) to have run.
"""

from importlib import metadata

import pytest


@pytest.mark.parametrize("module", ["cv2", "numpy", "sklearn", "mediapipe"])
def test_core_dependency_imports(module):
    pytest.importorskip(module)


def test_mediapipe_tasks_hand_landmarker_available():
    # We use ONE MediaPipe API: the Tasks API. Fail loudly if it ever disappears.
    pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import vision

    assert hasattr(vision, "HandLandmarker")
    assert hasattr(vision.RunningMode, "VIDEO")


def test_only_one_opencv_distribution_installed():
    # MediaPipe depends on opencv-contrib-python. Installing opencv-python as
    # well makes both packages fight over the same `cv2` folder.
    installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
    opencv = sorted(name for name in installed if name.startswith("opencv-"))
    assert len(opencv) <= 1, f"Conflicting OpenCV packages installed: {opencv}"
