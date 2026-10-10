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


def make_fist(center=(0.5, 0.5), size=0.3) -> HandLandmarks:
    """A synthetic fist: fingertips curled back toward the palm."""
    hand = make_open_hand(center, size)
    points = hand.points.copy()
    palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
    for tip_chain in ((2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)):
        for joint in tip_chain:
            points[joint] = palm + (points[joint] - palm) * 0.25
    return HandLandmarks(points=points, handedness="Right", score=0.97)


def make_swipe(dx=0.3, dy=0.0, frames=30, fps=30, start=(0.35, 0.5), size=0.3, jitter=0.0):
    """A synthetic swipe: an open hand moving (dx, dy) in normalized image units.

    Returns (frames array of shape (T, 21, 3), timestamps in ms).
    """
    rng = np.random.default_rng(0)
    seq, times = [], []
    for i in range(frames):
        f = i / (frames - 1)
        center = (start[0] + dx * f, start[1] + dy * f)
        points = make_open_hand(center, size).points
        if jitter:
            points = points + rng.normal(0, jitter, size=(1, 3)).astype(np.float32)
        seq.append(points)
        times.append(round(i * 1000 / fps))
    return np.stack(seq), np.array(times)


@pytest.fixture
def open_hand() -> HandLandmarks:
    return make_open_hand()
