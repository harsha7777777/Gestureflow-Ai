"""Recording one gesture example from the live hand stream.

This is a small state machine fed with every frame's detection result. It has
no timer of its own: it uses the frames' timestamps, which makes it easy to
test and means the countdown can't drift from what the camera actually saw.

    IDLE --start()--> COUNTDOWN --(3 s)--> RECORDING --(duration)--> DONE
                                                    \\--(problem)--> FAILED

Recording only collects landmarks. Training mode never triggers any action;
this module doesn't even know actions exist (see docs/architecture.md).
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from gestureflow.detection.landmarks import HandLandmarks
from gestureflow.features.dynamic import MIN_SWIPE_DISTANCE, has_enough_motion
from gestureflow.storage.samples import NEUTRAL, Sample, clean_gesture_name

COUNTDOWN_MS = 3000
STATIC_DURATION_MS = 1000  # hold the pose for one second
DYNAMIC_DURATION_MS = 1500  # long enough for one swipe
MIN_STATIC_FRAMES = 5
MIN_DYNAMIC_FRAMES = 8
MAX_HAND_GAP_MS = 300  # a movement with the hand missing longer than this is unusable


class RecorderState(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    DONE = "done"
    FAILED = "failed"


@dataclass
class RecorderStatus:
    state: RecorderState
    message: str
    progress: float = 0.0  # 0..1 within the current phase


@dataclass
class SampleRecorder:
    gesture: str
    kind: str  # "static" or "dynamic"
    countdown_ms: int = COUNTDOWN_MS
    state: RecorderState = RecorderState.IDLE
    result: Sample | None = None
    failure: str = ""
    _phase_start: int | None = None
    _last_hand_t: int | None = None
    _frames: list = field(default_factory=list)
    _times: list = field(default_factory=list)
    _handedness: str = "Unknown"
    _aspect: float = 1.0

    def __post_init__(self) -> None:
        self.gesture = clean_gesture_name(self.gesture)
        if self.kind not in ("static", "dynamic"):
            raise ValueError(f"kind must be 'static' or 'dynamic', not {self.kind!r}")

    @property
    def duration_ms(self) -> int:
        return STATIC_DURATION_MS if self.kind == "static" else DYNAMIC_DURATION_MS

    @property
    def finished(self) -> bool:
        return self.state in (RecorderState.DONE, RecorderState.FAILED)

    def start(self) -> None:
        self.state = RecorderState.COUNTDOWN
        self._phase_start = None  # set by the first frame, so time comes from the camera
        self._frames, self._times = [], []
        self.result, self.failure = None, ""

    def cancel(self) -> None:
        self.state = RecorderState.IDLE

    def update(self, hands: list[HandLandmarks], t_ms: int) -> RecorderStatus:
        """Feed one frame's detections. Returns what to show the user."""
        if self.state == RecorderState.COUNTDOWN:
            if self._phase_start is None:
                self._phase_start = t_ms
            elapsed = t_ms - self._phase_start
            if elapsed < self.countdown_ms:
                remaining = (self.countdown_ms - elapsed) / 1000
                return RecorderStatus(
                    self.state, f"Get ready... {remaining:.0f}", elapsed / self.countdown_ms
                )
            self.state = RecorderState.RECORDING
            self._phase_start = t_ms
            self._last_hand_t = t_ms

        if self.state == RecorderState.RECORDING:
            return self._record(hands, t_ms)
        if self.state == RecorderState.DONE:
            return RecorderStatus(self.state, "Got it!", 1.0)
        if self.state == RecorderState.FAILED:
            return RecorderStatus(self.state, self.failure, 1.0)
        return RecorderStatus(self.state, "")

    # ------------------------------------------------------------------ internals

    def _record(self, hands: list[HandLandmarks], t_ms: int) -> RecorderStatus:
        elapsed = t_ms - self._phase_start
        if hands:
            hand = hands[0]
            self._frames.append(hand.points)
            self._times.append(t_ms)
            self._handedness, self._aspect = hand.handedness, hand.aspect
            self._last_hand_t = t_ms
        elif self.kind == "dynamic" and t_ms - self._last_hand_t > MAX_HAND_GAP_MS:
            return self._fail("Your hand left the camera view. Keep it visible and try again.")

        if elapsed < self.duration_ms:
            action = "Hold the pose" if self.kind == "static" else "Do the movement now"
            return RecorderStatus(self.state, f"Recording... {action}", elapsed / self.duration_ms)
        return self._finish()

    def _finish(self) -> RecorderStatus:
        minimum = MIN_STATIC_FRAMES if self.kind == "static" else MIN_DYNAMIC_FRAMES
        if len(self._frames) < minimum:
            return self._fail("No hand was seen long enough. Keep your hand in view and try again.")
        frames = np.stack(self._frames)
        if (
            self.kind == "dynamic"
            and self.gesture != NEUTRAL
            and not has_enough_motion(frames, self._aspect)
        ):
            return self._fail(
                f"The hand barely moved. Move it about {MIN_SWIPE_DISTANCE:g} hand length or more "
                "across the frame and try again."
            )
        self.result = Sample(
            gesture=self.gesture,
            kind=self.kind,
            frames=frames,
            timestamps_ms=np.array(self._times, dtype=np.int64),
            handedness=self._handedness,
            aspect=self._aspect,
        )
        self.state = RecorderState.DONE
        return RecorderStatus(self.state, "Got it!", 1.0)

    def _fail(self, message: str) -> RecorderStatus:
        self.state = RecorderState.FAILED
        self.failure = message
        return RecorderStatus(self.state, message, 1.0)
