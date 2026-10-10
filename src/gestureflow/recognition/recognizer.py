"""Live recognition: hand landmarks in, gesture events out.

Every camera frame goes through two paths:

- **Movement path:** the last MOVEMENT_WINDOW_MS of frames form a sliding
  window. Only when the hand really moved in that window (the fixed motion
  check from features/dynamic.py) is the window shown to the movement model.
- **Pose path:** only when the hand is *still* is the frame shown to the pose
  model. A swipe with an open palm therefore can't trigger an "open palm" pose
  halfway through. The reverse also holds: while the hand is still, no
  movement is read, so holding a pose can't fire a movement.

Both paths go through their own RecognitionFilter (threshold, smoothing,
neutral, release, cooldown). After a movement fires, the window is cleared and
nothing can fire for LOCKOUT_MS, so bringing your hand back after a swipe right
is not taken as a swipe left.

The output is gesture names only. What a gesture *does* is decided elsewhere
(mapping, M4); this module doesn't know actions exist.
"""

import contextlib
from collections import deque
from dataclasses import dataclass

import numpy as np

from gestureflow.detection.landmarks import HandLandmarks
from gestureflow.features.dynamic import has_enough_motion, palm_trajectory, sequence_features
from gestureflow.features.static import static_features
from gestureflow.recognition.filter import FilterSettings, GestureEvent, RecognitionFilter
from gestureflow.training.model import GestureModel

MOVEMENT_WINDOW_MS = 1500  # how much recent motion the movement model sees
MIN_MOVEMENT_WINDOW_MS = 400
MIN_MOVEMENT_FRAMES = 8
STILL_WINDOW_MS = 300  # a pose is only read when the palm moved less than...
MAX_STILL_TRAVEL = 0.35  # ...this many hand sizes in the last STILL_WINDOW_MS
LOCKOUT_MS = 800  # after a movement, nothing fires for this long


@dataclass
class LiveResult:
    """What to show the user for one frame."""

    label: str | None  # current confident guess (may be "neutral"), None = not sure
    confidence: float  # confidence of the raw guess for this frame
    source: str  # "pose", "movement" or "" (nothing)
    event: GestureEvent | None = None  # set on the one frame a gesture is recognized
    moving: bool = False


class LiveRecognizer:
    def __init__(
        self, model: GestureModel, threshold: float = 0.7, cooldown_ms: int = 1500
    ) -> None:
        self.model = model
        # A pose must be held for about half a second (12 of 15 frames at 30 FPS),
        # so passing through a hand shape on the way to something else doesn't count.
        self.pose_filter = RecognitionFilter(
            FilterSettings(threshold, window=15, min_agree=12, cooldown_ms=cooldown_ms), "pose"
        )
        # Consecutive windows overlap heavily, so fewer agreeing frames are needed.
        self.movement_filter = RecognitionFilter(
            FilterSettings(threshold, window=3, min_agree=2, cooldown_ms=cooldown_ms), "movement"
        )
        self._frames: deque = deque()  # (t_ms, points, aspect)
        self._locked_until = -1

    @property
    def threshold(self) -> float:
        return self.pose_filter.settings.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.pose_filter.settings.threshold = value
        self.movement_filter.settings.threshold = value

    def reset(self) -> None:
        self._frames.clear()
        self.pose_filter.reset()
        self.movement_filter.reset()

    def update(self, hands: list[HandLandmarks], t_ms: int) -> LiveResult:
        if not hands:
            self.reset()  # gaps break both windows; start fresh when the hand returns
            return LiveResult(None, 0.0, "")
        hand = hands[0]
        self._frames.append((t_ms, hand.points, hand.aspect))
        while self._frames and t_ms - self._frames[0][0] > MOVEMENT_WINDOW_MS:
            self._frames.popleft()

        locked = t_ms < self._locked_until
        still = self._is_still(hand.aspect)
        moving, mv_label, mv_conf = (
            (False, None, 0.0) if still else self._movement_guess(hand.aspect)
        )
        event = self.movement_filter.update(None if locked else mv_label, mv_conf, t_ms)
        if event is not None:
            self._frames.clear()
            self._locked_until = t_ms + LOCKOUT_MS
            self.pose_filter.reset()
            return LiveResult(event.name, mv_conf, "movement", event, moving=True)

        pose_label, pose_conf = None, 0.0
        if still and self.model.pose is not None:
            with contextlib.suppress(ValueError):  # a degenerate detection: no guess
                pose_label, pose_conf = self.model.pose.predict(
                    static_features(hand.points, hand.aspect)
                )
        event = self.pose_filter.update(None if locked else pose_label, pose_conf, t_ms)

        if moving:
            shown, conf, source = mv_label, mv_conf, "movement"
        else:
            shown, conf, source = pose_label, pose_conf, "pose" if pose_label else ""
        if shown is not None and conf < self.threshold:
            shown = None  # below the threshold the app is "not sure"
        return LiveResult(shown, conf, source, event, moving)

    # ------------------------------------------------------------------ helpers

    def _window(self) -> tuple[np.ndarray, np.ndarray]:
        times = np.array([t for t, _, _ in self._frames], dtype=np.int64)
        frames = np.stack([p for _, p, _ in self._frames])
        return frames, times

    def _movement_guess(self, aspect: float) -> tuple[bool, str | None, float]:
        """(hand moved enough, movement guess, confidence) for the current window."""
        if len(self._frames) < MIN_MOVEMENT_FRAMES:
            return False, None, 0.0
        frames, times = self._window()
        if times[-1] - times[0] < MIN_MOVEMENT_WINDOW_MS:
            return False, None, 0.0
        try:
            if not has_enough_motion(frames, aspect):
                return False, None, 0.0
            if self.model.movement is None:
                return True, None, 0.0
            label, conf = self.model.movement.predict(sequence_features(frames, times, aspect))
        except ValueError:  # degenerate hand or duplicate timestamps
            return False, None, 0.0
        return True, label, conf

    def _is_still(self, aspect: float) -> bool:
        recent = [p for t, p, _ in self._frames if self._frames[-1][0] - t <= STILL_WINDOW_MS]
        if len(recent) < 2:
            return True
        try:
            trajectory, _ = palm_trajectory(np.stack(recent), aspect)
        except ValueError:
            return False
        return float(np.max(np.linalg.norm(trajectory, axis=1))) < MAX_STILL_TRAVEL
