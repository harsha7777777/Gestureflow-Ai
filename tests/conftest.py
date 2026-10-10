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


def make_half_curled(center=(0.5, 0.5), size=0.3) -> HandLandmarks:
    """Something between open and fist: a relaxed, ordinary hand (good neutral)."""
    open_pts = make_open_hand(center, size).points
    fist_pts = make_fist(center, size).points
    return HandLandmarks(points=(open_pts + fist_pts) / 2, handedness="Right", score=0.95)


SHAPES = {"open": make_open_hand, "fist": make_fist, "relaxed": make_half_curled}


def make_pose_sample(gesture, shape, seed, frames=30):
    """A recorded pose example with a little natural variation, like a real recording."""
    from gestureflow.storage.samples import Sample

    rng = np.random.default_rng(seed)
    center = (0.35 + 0.3 * rng.random(), 0.35 + 0.3 * rng.random())
    size = 0.22 + 0.12 * rng.random()
    base = SHAPES[shape](center, size).points
    seq = [base + rng.normal(0, 0.002, base.shape).astype(np.float32) for _ in range(frames)]
    return Sample(gesture, "static", np.stack(seq), np.arange(frames) * 33, "Right", 1.0)


def make_swipe_sample(gesture, dx, seed, dy=0.0, frames=60):
    """A recorded movement example: still, swipe in the middle, still (2 s at 30 fps)."""
    from gestureflow.storage.samples import Sample

    rng = np.random.default_rng(seed)
    size = 0.22 + 0.1 * rng.random()
    start = (0.5 - dx / 2 + 0.05 * rng.normal(), 0.5 - dy / 2 + 0.05 * rng.normal())
    seq = []
    for i in range(frames):
        f = min(max((i - 15) / 25, 0.0), 1.0)  # moves during frames 15-40
        center = (start[0] + dx * f, start[1] + dy * f)
        points = make_open_hand(center, size).points
        seq.append(points + rng.normal(0, 0.002, points.shape).astype(np.float32))
    return Sample(gesture, "dynamic", np.stack(seq), np.arange(frames) * 33, "Right", 1.0)


def make_neutral_movement_sample(seed, frames=60):
    """Ordinary motion: a relaxed hand drifting slowly and wobbling (not a swipe)."""
    from gestureflow.storage.samples import Sample

    rng = np.random.default_rng(seed)
    size = 0.22 + 0.1 * rng.random()
    angle = rng.random() * 2 * np.pi
    seq = []
    for i in range(frames):
        drift = 0.04 * i / frames
        wobble = 0.02 * np.sin(i / 4 + angle)
        center = (0.5 + drift * np.cos(angle) + wobble, 0.5 + drift * np.sin(angle))
        points = make_half_curled(center, size).points
        seq.append(points + rng.normal(0, 0.002, points.shape).astype(np.float32))
    return Sample("neutral", "dynamic", np.stack(seq), np.arange(frames) * 33, "Right", 1.0)


def make_training_set(per_gesture=5):
    """Poses: palm, fist (+ neutral relaxed hand). Movements: swipe_left, swipe_right
    (+ neutral motion)."""
    samples = []
    for i in range(per_gesture):
        samples.append(make_pose_sample("palm", "open", seed=100 + i))
        samples.append(make_pose_sample("fist", "fist", seed=200 + i))
        samples.append(make_pose_sample("neutral", "relaxed", seed=300 + i))
        samples.append(make_swipe_sample("swipe_right", 0.3, seed=400 + i))
        samples.append(make_swipe_sample("swipe_left", -0.3, seed=500 + i))
        samples.append(make_neutral_movement_sample(seed=600 + i))
    return samples
