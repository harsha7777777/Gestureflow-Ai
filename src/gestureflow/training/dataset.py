"""Turning recorded examples into rows of numbers a classifier can learn from.

There are two separate models, because poses and movements are different problems:

- The **pose model** learns from single frames. Every frame of a pose example is
  one training row (a one-second recording gives about 30 rows). Frames from
  "neutral" examples of either kind teach it what an ordinary hand looks like.
- The **movement model** learns from windows of frames. Live, the app looks at
  the last 1.5 seconds, a window that slides forward with every frame. Training
  cuts exactly that kind of window out of each recorded example, ending every
  0.1 s, so the model learns what the movement looks like through the same
  sliding window it will see live (early, middle and late in the movement).

Rows remember which example they came from (`groups`), so evaluation can hold
out whole examples. Testing on frames from an example the model also trained on
would make the accuracy look far better than it really is.
"""

from dataclasses import dataclass

import numpy as np

from gestureflow.features.dynamic import (
    MIN_WINDOW_MS,
    NUM_SEQUENCE_FEATURES,
    WINDOW_MS,
    has_enough_motion,
    sequence_features,
)
from gestureflow.features.static import NUM_STATIC_FEATURES, static_features
from gestureflow.storage.samples import NEUTRAL, Sample

WINDOW_STEP_MS = 100  # a training window ends every 0.1 s through the recording
MIN_WINDOW_FRAMES = 6


@dataclass
class Dataset:
    X: np.ndarray  # (rows, features)
    y: np.ndarray  # (rows,) gesture names
    groups: np.ndarray  # (rows,) index of the example each row came from
    original: np.ndarray  # (rows,) True for rows of the unmodified example

    def __len__(self) -> int:
        return len(self.y)

    def examples_per_class(self) -> dict[str, int]:
        counts: dict[str, set] = {}
        for label, group in zip(self.y, self.groups, strict=True):
            counts.setdefault(str(label), set()).add(int(group))
        return {label: len(groups) for label, groups in sorted(counts.items())}


def _empty(num_features: int) -> Dataset:
    return Dataset(
        np.zeros((0, num_features), np.float32),
        np.array([], dtype=object),
        np.array([], dtype=np.int64),
        np.array([], dtype=bool),
    )


def _build(rows: list, num_features: int) -> Dataset:
    if not rows:
        return _empty(num_features)
    features, y, groups, original = zip(*rows, strict=True)
    return Dataset(
        np.stack(features).astype(np.float32),
        np.array(y, dtype=object),
        np.array(groups, dtype=np.int64),
        np.array(original, dtype=bool),
    )


def pose_dataset(samples: list[Sample]) -> Dataset:
    """One row per frame of every pose example, plus every frame of neutral examples."""
    rows = []
    for index, sample in enumerate(samples):
        if sample.kind != "static" and sample.gesture != NEUTRAL:
            continue
        for points in sample.frames:
            try:
                features = static_features(points, sample.aspect)
            except ValueError:  # a degenerate detection; skip that frame
                continue
            rows.append((features, sample.gesture, index, True))
    return _build(rows, NUM_STATIC_FEATURES)


def sliding_windows(sample: Sample) -> list[tuple[np.ndarray, np.ndarray]]:
    """Windows like the live ones: up to WINDOW_MS long, ending every WINDOW_STEP_MS."""
    t = sample.timestamps_ms.astype(np.int64)
    windows = []
    for end in range(int(t[0]) + MIN_WINDOW_MS, int(t[-1]) + 1, WINDOW_STEP_MS):
        keep = (t <= end) & (t > end - WINDOW_MS)
        if keep.sum() >= MIN_WINDOW_FRAMES:
            windows.append((sample.frames[keep], t[keep]))
    return windows


def movement_windows(sample: Sample) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """(frames, timestamps, is_original) for the full example and its usable windows."""
    windows = [(sample.frames, sample.timestamps_ms, True)]
    for frames, times in sliding_windows(sample):
        # Live, only windows that pass the motion check are ever classified, so
        # those are the ones to learn from (neutral keeps every window).
        if sample.gesture != NEUTRAL and not has_enough_motion(frames, sample.aspect):
            continue
        windows.append((frames, times, False))
    return windows


def movement_dataset(samples: list[Sample]) -> Dataset:
    """Rows for every movement example (full recording plus sliding windows)."""
    rows = []
    for index, sample in enumerate(samples):
        if sample.kind != "dynamic" or len(sample.frames) < MIN_WINDOW_FRAMES:
            continue
        try:
            windows = movement_windows(sample)
        except ValueError:  # degenerate hand in this example
            continue
        for frames, times, is_original in windows:
            try:
                features = sequence_features(frames, times, sample.aspect)
            except ValueError:
                continue
            rows.append((features, sample.gesture, index, is_original))
    return _build(rows, NUM_SEQUENCE_FEATURES)
