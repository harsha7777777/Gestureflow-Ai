"""Drawing hand landmarks onto a frame (for display only)."""

import cv2
import numpy as np

from gestureflow.detection.landmarks import HAND_CONNECTIONS, HandLandmarks

BONE_COLOR = (255, 255, 255)  # white lines
JOINT_COLOR = (0, 200, 255)  # in RGB: a light blue
TIP_COLOR = (255, 80, 80)  # fingertips in red, to make finger poses easy to read
FINGERTIPS = {4, 8, 12, 16, 20}


def to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized (x, y) landmark coordinates to integer pixel positions."""
    xy = points[:, :2] * np.array([width, height], dtype=np.float32)
    xy = np.clip(xy, 0, [width - 1, height - 1])
    return xy.astype(np.int32)


def draw_hands(image_rgb: np.ndarray, hands: list[HandLandmarks]) -> np.ndarray:
    """Draw each hand's skeleton onto `image_rgb` in place and return it."""
    height, width = image_rgb.shape[:2]
    # Scale line thickness with image size so it looks the same at any resolution.
    thickness = max(1, round(min(width, height) / 240))
    for hand in hands:
        px = to_pixels(hand.points, width, height)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                image_rgb, tuple(px[start]), tuple(px[end]), BONE_COLOR, thickness, cv2.LINE_AA
            )
        for i, (x, y) in enumerate(px):
            color = TIP_COLOR if i in FINGERTIPS else JOINT_COLOR
            cv2.circle(image_rgb, (int(x), int(y)), thickness + 2, color, -1, cv2.LINE_AA)
    return image_rgb
