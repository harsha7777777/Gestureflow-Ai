"""Features for static gestures (a hand pose held still, like a thumbs-up).

Raw landmarks depend on where your hand is in the frame and how close it is to
the camera. A classifier trained on raw numbers would learn "hand in the top-left
corner" instead of "thumbs-up". So we:

1. undo the image stretch (x is multiplied by width/height),
2. translate so the wrist is at the origin (position no longer matters),
3. divide by hand size, the wrist-to-middle-knuckle distance (distance to the
   camera no longer matters).

Then we add a few relationships between landmarks that describe the pose
directly: distances between fingertips and how far each finger is extended.
"""

import itertools

import numpy as np

from gestureflow.detection.landmarks import MIDDLE_MCP, NUM_LANDMARKS, WRIST

FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
FINGER_BASES = (2, 5, 9, 13, 17)  # the knuckle each finger starts from
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TIP_PAIRS = tuple(itertools.combinations(range(5), 2))  # 10 pairs

# Hands smaller than this (in normalized image units) are treated as invalid,
# so we never divide by (almost) zero.
MIN_HAND_SIZE = 1e-4


def _feature_names() -> tuple[str, ...]:
    names = [f"p{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
    names += [f"dist_{FINGER_NAMES[a]}_{FINGER_NAMES[b]}" for a, b in TIP_PAIRS]
    names += [f"extension_{name}" for name in FINGER_NAMES]
    return tuple(names)


STATIC_FEATURE_NAMES = _feature_names()
NUM_STATIC_FEATURES = len(STATIC_FEATURE_NAMES)  # 63 + 10 + 5 = 78


def unstretch(points: np.ndarray, aspect: float) -> np.ndarray:
    """Return a copy with x scaled by the image aspect ratio, so x and y use the same unit."""
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] *= aspect
    return out


def hand_size(points: np.ndarray, aspect: float = 1.0) -> float:
    """Wrist to middle-finger knuckle distance. It barely changes when fingers move."""
    p = unstretch(points, aspect)
    return float(np.linalg.norm(p[MIDDLE_MCP, :2] - p[WRIST, :2]))


def normalize_landmarks(points: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """Translate to the wrist and scale by hand size. Returns shape (21, 3).

    Raises ValueError if the hand is degenerate (all points on top of each other).
    """
    p = unstretch(points, aspect)
    size = float(np.linalg.norm(p[MIDDLE_MCP, :2] - p[WRIST, :2]))
    if size < MIN_HAND_SIZE:
        raise ValueError("hand is too small or degenerate to normalize")
    return (p - p[WRIST]) / size


def static_features(points: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """Feature vector for one hand pose, shape (NUM_STATIC_FEATURES,)."""
    norm = normalize_landmarks(points, aspect)
    tips = norm[list(FINGERTIPS)]
    bases = norm[list(FINGER_BASES)]

    tip_distances = [np.linalg.norm(tips[a] - tips[b]) for a, b in TIP_PAIRS]
    # Extension: how far the fingertip is from the wrist compared with its knuckle.
    # Roughly 2 for a straight finger and close to 1 (or below) for a curled one.
    extensions = np.linalg.norm(tips, axis=1) / np.maximum(np.linalg.norm(bases, axis=1), 1e-6)

    return np.concatenate([norm.ravel(), tip_distances, extensions]).astype(np.float32)
