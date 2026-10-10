"""Features for dynamic gestures (movements, like a swipe to the left).

A swipe is a short sequence of frames. Two problems to solve:

1. People swipe at different speeds and cameras run at different frame rates,
   so sequences have different lengths. We resample every sequence to the same
   number of evenly spaced steps (by time, not by frame count).
2. A hand resting at the side of the frame must never count as a movement.
   Before a window is classified at all, `has_enough_motion` checks that the
   hand really moved: either the palm travelled far enough (a swipe), or the
   fingers clearly changed shape (a grab, a pinch, opening the hand). That check
   is a plain rule, not something the classifier has to learn, which makes it
   reliable.

All distances are measured in "hand sizes" (see static.hand_size), so the same
movement gives the same numbers close to or far from the camera.
"""

import numpy as np

from gestureflow.detection.landmarks import MIDDLE_MCP, WRIST
from gestureflow.features.static import (
    FINGER_NAMES,
    FINGERTIPS,
    MIN_HAND_SIZE,
    STATIC_FEATURE_NAMES,
    normalize_landmarks,
    static_features,
    unstretch,
)

# The palm center is steadier than any single landmark while fingers move.
PALM_LANDMARKS = (0, 5, 9, 13, 17)

RESAMPLED_STEPS = 16

# Moments (indices into the resampled steps) at which the finger shape is kept,
# so a grab (open -> fist) differs from a release (fist -> open).
KEY_STEPS = (0, 5, 10, 15)

# A window counts as a movement if the palm got at least this far from where it
# started, in hand sizes. One hand size (wrist to middle knuckle) is about 8-10 cm,
# so this is a short but deliberate swipe. The farthest point is used, so a
# there-and-back movement counts too.
MIN_SWIPE_DISTANCE = 0.8

# ...or if a fingertip moved at least this far relative to the wrist (in hand
# sizes), like closing the hand into a fist or pinching. Landmark noise on a
# still hand is well below 0.1.
MIN_SHAPE_CHANGE = 0.6


def _feature_names() -> tuple[str, ...]:
    names = [f"traj{i}_{axis}" for i in range(RESAMPLED_STEPS) for axis in "xy"]
    names += ["net_dx", "net_dy", "path_length"]
    names += [f"mean_{name}" for name in STATIC_FEATURE_NAMES]
    names += [f"tip{k}_{name}_{axis}" for k in KEY_STEPS for name in FINGER_NAMES for axis in "xy"]
    return tuple(names)


SEQUENCE_FEATURE_NAMES = _feature_names()
NUM_SEQUENCE_FEATURES = len(SEQUENCE_FEATURE_NAMES)  # 32 + 3 + 78 + 40 = 153


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


def _fingertips(frames: np.ndarray, aspect: float) -> np.ndarray:
    """Fingertip (x, y) relative to the wrist, in hand sizes. Shape (T, 5, 2)."""
    return np.stack([normalize_landmarks(f, aspect)[list(FINGERTIPS), :2] for f in frames])


def motion_amount(frames: np.ndarray, aspect: float = 1.0) -> tuple[float, float]:
    """(palm travel, shape change) over the window, both in hand sizes.

    Palm travel is the farthest the palm got from where it started. Shape change
    is the farthest any fingertip got from its starting spot relative to the wrist.
    Distances from the start (not path length) are used on purpose: a hand that
    jitters in place has a long path but never gets far from where it began.
    """
    frames = np.asarray(frames, dtype=np.float32)
    traj, _ = palm_trajectory(frames, aspect)
    travel = float(np.max(np.linalg.norm(traj, axis=1)))
    tips = _fingertips(frames, aspect)
    shape = float(np.max(np.linalg.norm(tips - tips[0], axis=2)))
    return travel, shape


def has_enough_motion(
    frames: np.ndarray,
    aspect: float = 1.0,
    min_distance: float = MIN_SWIPE_DISTANCE,
    min_shape_change: float = MIN_SHAPE_CHANGE,
) -> bool:
    """True if the palm travelled far enough or the fingers clearly changed shape."""
    travel, shape = motion_amount(frames, aspect)
    return travel >= min_distance or shape >= min_shape_change


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
    key_tips = _fingertips(resampled[list(KEY_STEPS)], aspect)
    return np.concatenate([traj.ravel(), net, [path], mean_pose, key_tips.ravel()]).astype(
        np.float32
    )
