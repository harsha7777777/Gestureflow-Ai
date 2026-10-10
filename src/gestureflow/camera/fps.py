"""Frames-per-second measurement over a short sliding window."""

import time
from collections import deque
from collections.abc import Callable


class FpsCounter:
    """Counts frames and reports the average rate over the last `window` seconds.

    A sliding window reacts to slowdowns within a second or two, while still
    being steady enough to read on screen.
    """

    def __init__(self, window: float = 1.0, clock: Callable[[], float] = time.perf_counter):
        self._window = window
        self._clock = clock  # injectable so tests can control time
        self._times: deque[float] = deque()

    def tick(self) -> float:
        """Record one frame and return the current FPS."""
        now = self._clock()
        self._times.append(now)
        while self._times and now - self._times[0] > self._window:
            self._times.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
