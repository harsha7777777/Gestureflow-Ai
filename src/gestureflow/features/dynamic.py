"""Features for dynamic gestures (movements, like a swipe to the left).

A swipe is a short sequence of frames. Two problems to solve:

1. People swipe at different speeds and cameras run at different frame rates,
   so sequences have different lengths. We resample every sequence to the same
   number of evenly spaced steps (by time, not by frame count).
2. A hand resting at the side of the frame must never count as a swipe. Before a
   window is classified at all, `has_enough_motion` checks that the hand really
   travelled far enough. That check is a plain rule, not something the
   classifier has to learn, which makes it reliable.

All distances are measured in "hand sizes" (see static.hand_size), so the same
movement gives the same numbers close to or far from the camera.
"""

import numpy as np

from gestureflow.detection.landmarks import MIDDLE_MCP, WRIST
from gestureflow.features.static import (
    MIN_HAND_SIZE,
    STATIC_FEATURE_NAMES,
    static_features,
    unstretch,
)

# The palm center is steadier than any single landmark while fingers move.
PALM_LANDMARKS = (0, 5, 9, 13, 17)

RESAMPLED_STEPS = 16

# Minimum net travel of the palm, in hand sizes, for a window to count as a
# movement. One hand size is roughly 8-10 cm, a deliberate but short swipe.
MIN_SWIPE_DISTANCE = 1.0


def _feature_names() -> tuple[str, ...]:
    names = [f"traj{i}_{axis}" for i in range(RESAMPLED_STEPS) for axis in "xy"]
    names += ["net_dx", "net_dy", "path_length"]
    names += [f"mean_{name}" for name in STATIC_FEATURE_NAMES]
    return tuple(names)


SEQUENCE_FEATURE_NAMES = _feature_names()
NUM_SEQUENCE_FEATURES = len(SEQUENCE_FEATURE_NAMES)  # 32 + 3 + 78 = 113


def resample(values: np.ndarray, timestamps_ms: np.ndarray, steps: int = RESAMPLED_STEPS):
    """Linearly resample a (T, ...) array to `steps` evenly spaced points in time."""
    values = np.asarray(values, dtype=np.float32)
    t = np.asarray(timestamps_ms, dtype=np.float64)
    if len(values) != len(t):
        raise ValueError("values and timestamps must have the same length")
    if len(values) < 2:
        raise ValueError("need at least 2 frames to resample")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must strictly increase")

    target = np.linspace(t[0], t[-1], steps)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(target, t, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((steps, *values.shape[1:])).astype(np.float32)


def palm_trajectory(frames: np.ndarray, aspect: float = 1.0) -> tuple[np.ndarray, float]:
    """Palm-center path relative to the first frame, in hand sizes.

    frames: array of shape (T, 21, 3). Returns (trajectory of shape (T, 2), mean hand size).
    """
    frames = np.asarray(frames, dtype=np.float32)
    unstretched = np.stack([unstretch(f, aspect) for f in frames])
    sizes = np.linalg.norm(unstretched[:, MIDDLE_MCP, :2] - unstretched[:, WRIST, :2], axis=1)
    mean_size = float(np.mean(sizes))
    if mean_size < MIN_HAND_SIZE:
        raise ValueError("hand is too small or degenerate")
    centers = unstretched[:, list(PALM_LANDMARKS), :2].mean(axis=1)
    return (centers - centers[0]) / mean_size, mean_size


def motion_stats(frames: np.ndarray, aspect: float = 1.0) -> tuple[float, float, float]:
    """(net dx, net dy, path length) of the palm over the window, in hand sizes."""
    traj, _ = palm_trajectory(frames, aspect)
    net = traj[-1] - traj[0]
    path = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    return float(net[0]), float(net[1]), path


def has_enough_motion(
    frames: np.ndarray, aspect: float = 1.0, min_distance: float = MIN_SWIPE_DISTANCE
) -> bool:
    """True if the palm moved at least `min_distance` hand sizes from start to end.

    Net distance (not path length) is used on purpose: a hand that jitters or
    wobbles in place has a long path but almost no net movement.
    """
    dx, dy, _ = motion_stats(frames, aspect)
    return float(np.hypot(dx, dy)) >= min_distance


def sequence_features(
    frames: np.ndarray, timestamps_ms: np.ndarray, aspect: float = 1.0
) -> np.ndarray:
    """Feature vector for one movement window, shape (NUM_SEQUENCE_FEATURES,).

    frames: (T, 21, 3) landmarks; timestamps_ms: (T,) frame times.
    """
    frames = np.asarray(frames, dtype=np.float32)
    resampled = resample(frames, timestamps_ms)
    traj, _ = palm_trajectory(resampled, aspect)
    net = traj[-1] - traj[0]
    path = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    # The average hand pose lets "swipe with open palm" differ from "swipe with a fist".
    mean_pose = np.mean([static_features(f, aspect) for f in resampled], axis=0)
    return np.concatenate([traj.ravel(), net, [path], mean_pose]).astype(np.float32)
