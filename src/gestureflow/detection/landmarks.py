"""Plain data types for hand landmarks, independent of MediaPipe.

The rest of the app works with these types instead of MediaPipe's own objects,
so swapping or upgrading the detector only touches detection/hand_detector.py.
"""

from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 21

# Landmark indices, as defined by MediaPipe's hand model.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

# Pairs of landmark indices that form the "bones" of the hand skeleton.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # palm
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # thumb
    (1, 2), (2, 3), (3, 4),
    # index finger
    (5, 6), (6, 7), (7, 8),
    # middle finger
    (9, 10), (10, 11), (11, 12),
    # ring finger
    (13, 14), (14, 15), (15, 16),
    # pinky
    (17, 18), (18, 19), (19, 20),
)  # fmt: skip


@dataclass(frozen=True)
class HandLandmarks:
    """One detected hand in one frame.

    points: array of shape (21, 3). x and y are normalized to the image
        (0..1, origin top-left); z is depth relative to the wrist, roughly in
        the same scale as x (smaller = closer to the camera).
    handedness: "Left" or "Right", as seen in the (mirrored) image.
    score: the detector's confidence in the handedness label, 0..1.
    """

    points: np.ndarray
    handedness: str
    score: float

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"expected points of shape (21, 3), got {self.points.shape}")
