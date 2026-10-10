"""The safety filter between "the model thinks..." and "a gesture happened".

A classifier gives a guess for every frame, and single frames are noisy. The
filter only reports a gesture event when all of these hold:

1. **Threshold:** the guess is at least `threshold` confident. Below that the
   frame counts as "not sure", and not sure means do nothing.
2. **Smoothing:** the same gesture won `min_agree` of the last `window` frames.
3. **Neutral:** the winner is not the neutral class.
4. **Release:** a gesture that just fired must disappear before it can fire
   again, so holding a pose triggers once, not 30 times a second.
5. **Cooldown:** the same gesture can't fire again within `cooldown_ms`.

It works on names and numbers only, so any future input (eye tracking, voice)
can reuse it.
"""

from collections import Counter, deque
from dataclasses import dataclass

from gestureflow.storage.samples import NEUTRAL


@dataclass(frozen=True)
class GestureEvent:
    name: str
    confidence: float
    t_ms: int
    source: str  # which model produced it, e.g. "pose" or "movement"


@dataclass
class FilterSettings:
    threshold: float = 0.7
    window: int = 6
    min_agree: int = 5
    cooldown_ms: int = 1500

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 1 <= self.min_agree <= self.window:
            raise ValueError("min_agree must be between 1 and window")


class RecognitionFilter:
    def __init__(self, settings: FilterSettings | None = None, source: str = "") -> None:
        self.settings = settings or FilterSettings()
        self.source = source
        self._recent: deque = deque(maxlen=self.settings.window)
        self._latched: str | None = None
        self._last_fired: dict[str, int] = {}

    def reset(self) -> None:
        """Forget recent frames (e.g. when the hand leaves). Cooldowns are kept."""
        self._recent.clear()
        self._latched = None

    def stable(self) -> tuple[str | None, float]:
        """The gesture that currently passes threshold + smoothing (neutral included)."""
        votes = Counter(label for label, _ in self._recent if label is not None)
        if not votes:
            return None, 0.0
        label, count = votes.most_common(1)[0]
        if count < self.settings.min_agree:
            return None, 0.0
        confidences = [c for lbl, c in self._recent if lbl == label]
        return label, sum(confidences) / len(confidences)

    def update(self, label: str | None, confidence: float, t_ms: int) -> GestureEvent | None:
        """Feed one frame's guess (None = no guess). Returns an event or None."""
        if self._recent.maxlen != self.settings.window:
            self._recent = deque(self._recent, maxlen=self.settings.window)
        sure = label is not None and confidence >= self.settings.threshold
        self._recent.append((label if sure else None, confidence))

        stable, mean_confidence = self.stable()
        if stable != self._latched:
            self._latched = None  # the previous gesture was released
        if stable is None or stable == NEUTRAL or stable == self._latched:
            return None
        # Latch even when the cooldown blocks it, so a pose held through the
        # cooldown doesn't fire the moment the cooldown ends.
        self._latched = stable
        last = self._last_fired.get(stable)
        if last is not None and t_ms - last < self.settings.cooldown_ms:
            return None
        self._last_fired[stable] = t_ms
        return GestureEvent(stable, mean_confidence, t_ms, self.source)
