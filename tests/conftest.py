"""Shared test fixtures. No test needs a real webcam."""

import os

# Run Qt without a screen (CI machines have none). Must be set before Qt starts.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gestureflow.detection.landmarks import HandLandmarks  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def make_open_hand(center=(0.5, 0.5), size=0.3) -> HandLandmarks:
    """A synthetic, roughly hand-shaped set of 21 landmarks (open palm, fingers up)."""
    cx, cy = center
    points = [(0.0, 0.5)]  # wrist at the bottom
    # (x offset, base y) for thumb, index, middle, ring, pinky; 4 joints each
    for finger, x in enumerate((-0.35, -0.15, 0.0, 0.15, 0.3)):
        for joint in range(4):
            spread = x * (1 + 0.3 * joint) if finger == 0 else x
            points.append((spread, 0.2 - 0.17 * joint))
    arr = np.array([(cx + px * size, cy + py * size, 0.0) for px, py in points], dtype=np.float32)
    return HandLandmarks(points=arr, handedness="Right", score=0.98)


@pytest.fixture
def open_hand() -> HandLandmarks:
    return make_open_hand()
