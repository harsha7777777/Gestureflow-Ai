"""Hand landmark detection using the MediaPipe Tasks API (HandLandmarker).

This is the only module that imports MediaPipe. It turns a camera frame into
our own HandLandmarks type (see landmarks.py).
"""

from pathlib import Path

import numpy as np

from gestureflow.detection.landmarks import NUM_LANDMARKS, HandLandmarks
from gestureflow.detection.model_file import model_path


class DetectorError(RuntimeError):
    """The hand detector could not be created (for example, the model is missing)."""


class HandDetector:
    """Detects hand landmarks in a stream of video frames.

    Uses MediaPipe's VIDEO running mode, which tracks the hand between frames
    instead of searching the whole image every time. That is faster and steadier,
    but it requires frame timestamps that always increase.

    Use as a context manager so MediaPipe's resources are released:

        with HandDetector() as detector:
            hands = detector.detect(frame_rgb, timestamp_ms)
    """

    def __init__(
        self,
        model: Path | None = None,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model = model or model_path()
        if not model.is_file():
            raise DetectorError(
                f"Hand model not found at {model}. "
                "Download it with: python -m gestureflow --download-model"
            )

        # Imported here so that importing this module stays cheap (and works in
        # tests) even where MediaPipe's native libraries can't load.
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # MediaPipe raises plain RuntimeError/OSError
            raise DetectorError(f"Could not start the hand detector: {exc}") from exc
        self._last_timestamp_ms = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]:
        """Return the hands found in an RGB frame (an empty list if none)."""
        import mediapipe as mp

        # MediaPipe rejects timestamps that don't strictly increase.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        height, width = frame_rgb.shape[:2]
        return convert_result(result, aspect=width / height)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def convert_result(result, aspect: float = 1.0) -> list[HandLandmarks]:
    """Convert a MediaPipe HandLandmarkerResult into a list of HandLandmarks."""
    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=False):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if points.shape != (NUM_LANDMARKS, 3):
            continue
        best = handedness[0] if handedness else None
        hands.append(
            HandLandmarks(
                points=points,
                handedness=(best.category_name if best else "Unknown") or "Unknown",
                score=float(best.score) if best and best.score is not None else 0.0,
                aspect=aspect,
            )
        )
    return hands
