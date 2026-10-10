"""Turning recorded examples into rows of numbers a classifier can learn from.

There are two separate models, because poses and movements are different problems:

- The **pose model** learns from single frames. Every frame of a pose example is
  one training row (a one-second recording gives about 30 rows). Frames from
  "neutral" examples of either kind teach it what an ordinary hand looks like.
- The **movement model** learns from whole windows of frames. Every movement
  example gives one row for the full recording plus a few shorter "crops" of
  it. Live, the app looks at a sliding window that rarely lines up exactly with
  the movement, and crops teach the model that a slightly early or late window
  is still the same movement.

Rows remember which example they came from (`groups`), so evaluation can hold
out whole examples. Testing on frames from an example the model also trained on
would make the accuracy look far better than it really is.
"""

from dataclasses import dataclass

import numpy as np

from gestureflow.features.dynamic import (
    NUM_SEQUENCE_FEATURES,
    has_enough_motion,
    sequence_features,
)
from gestureflow.features.static import NUM_STATIC_FEATURES, static_features
from gestureflow.storage.samples import NEUTRAL, Sample

# Shorter windows cut from each movement example: (length, position), both as a
# fraction of the recording. Position 0 = start, 0.5 = middle, 1 = end.
CROPS = ((0.8, 0.0), (0.8, 0.5), (0.8, 1.0), (0.6, 0.0), (0.6, 0.5), (0.6, 1.0))
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


def crop(sample: Sample, length: float, position: float) -> tuple[np.ndarray, np.ndarray]:
    """Frames and timestamps of a shorter window inside a movement example."""
    t = sample.timestamps_ms.astype(np.float64)
    duration = t[-1] - t[0]
    start = t[0] + position * duration * (1 - length)
    keep = (t >= start - 1e-6) & (t <= start + duration * length + 1e-6)
    return sample.frames[keep], sample.timestamps_ms[keep]


def movement_windows(sample: Sample) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """(frames, timestamps, is_original) for the full example and its usable crops."""
    windows = [(sample.frames, sample.timestamps_ms, True)]
    for length, position in CROPS:
        frames, times = crop(sample, length, position)
        if len(frames) < MIN_WINDOW_FRAMES:
            continue
        # A crop that cut the movement out isn't an example of the movement anymore.
        if sample.gesture != NEUTRAL and not has_enough_motion(frames, sample.aspect):
            continue
        windows.append((frames, times, False))
    return windows


def movement_dataset(samples: list[Sample]) -> Dataset:
    """Rows for every movement example (full window plus crops)."""
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
