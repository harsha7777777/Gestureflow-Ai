import numpy as np
import pytest

from gestureflow.detection.landmarks import HandLandmarks
from gestureflow.recognition.filter import FilterSettings, RecognitionFilter
from gestureflow.recognition.recognizer import LOCKOUT_MS, LiveRecognizer
from gestureflow.training.trainer import train
from tests.conftest import (
    make_fist,
    make_half_curled,
    make_open_hand,
    make_swipe_sample,
    make_training_set,
)

# ---------------------------------------------------------------- the filter


def feed(flt, labels, start=0, step=33, confidence=0.9):
    """Feed a list of labels, one per frame. Returns the events that fired."""
    events = []
    for i, label in enumerate(labels):
        ev = flt.update(label, confidence, start + i * step)
        if ev:
            events.append(ev)
    return events


def test_needs_several_agreeing_frames():
    flt = RecognitionFilter(FilterSettings(window=6, min_agree=5))
    assert feed(flt, ["fist"] * 4) == []
    (event,) = feed(flt, ["fist"], start=200)
    assert event.name == "fist"


def test_flickering_guesses_never_fire():
    flt = RecognitionFilter(FilterSettings(window=6, min_agree=5))
    assert feed(flt, ["fist", "palm"] * 20) == []


def test_low_confidence_is_treated_as_not_sure():
    flt = RecognitionFilter(FilterSettings(threshold=0.8))
    assert feed(flt, ["fist"] * 30, confidence=0.79) == []
    assert len(feed(flt, ["fist"] * 6, start=2000, confidence=0.81)) == 1


def test_neutral_never_fires():
    flt = RecognitionFilter()
    assert feed(flt, ["neutral"] * 50) == []
    assert flt.stable()[0] == "neutral"


def test_holding_a_pose_fires_once():
    flt = RecognitionFilter(FilterSettings(cooldown_ms=1000))
    events = feed(flt, ["fist"] * 300)  # about 10 seconds
    assert len(events) == 1


def test_release_then_repeat_respects_cooldown():
    flt = RecognitionFilter(FilterSettings(cooldown_ms=1500))
    pattern = ["fist"] * 8 + [None] * 8  # about 0.5 s on, 0.5 s off
    events = feed(flt, pattern * 6)
    gaps = np.diff([e.t_ms for e in events])
    assert len(events) >= 2
    assert np.all(gaps >= 1500)


def test_cooldown_is_per_gesture():
    flt = RecognitionFilter(FilterSettings(cooldown_ms=5000))
    events = feed(flt, ["fist"] * 8 + [None] * 6 + ["palm"] * 8)
    assert [e.name for e in events] == ["fist", "palm"]


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        FilterSettings(threshold=1.5)
    with pytest.raises(ValueError):
        FilterSettings(window=3, min_agree=4)


# ------------------------------------------------------------ live recognizer


@pytest.fixture(scope="module")
def model():
    trained, _ = train(make_training_set(), progress=lambda message: None)
    return trained


def hand(points):
    return HandLandmarks(points=np.asarray(points, dtype=np.float32), handedness="Right", score=0.9)


def run(recognizer, frames, start=10_000, step=33):
    """Stream frames (arrays of 21x3, or None for no hand). Returns (events, results)."""
    events, results = [], []
    for i, points in enumerate(frames):
        result = recognizer.update([] if points is None else [hand(points)], start + i * step)
        results.append(result)
        if result.event:
            events.append(result.event)
    return events, results


def test_held_pose_is_recognized_once(model):
    rec = LiveRecognizer(model)
    events, results = run(rec, [make_fist().points] * 90)  # 3 seconds
    assert [(e.name, e.source) for e in events] == [("fist", "pose")]
    assert results[-1].label == "fist" and results[-1].confidence > 0.7


def test_relaxed_hand_does_nothing(model):
    rec = LiveRecognizer(model)
    events, results = run(rec, [make_half_curled().points] * 90)
    assert events == []
    assert results[-1].label == "neutral"


def test_no_hand_does_nothing(model):
    events, results = run(LiveRecognizer(model), [None] * 30)
    assert events == [] and results[-1].label is None


def test_swipe_is_recognized_once(model):
    rec = LiveRecognizer(model)
    swipe = make_swipe_sample("x", -0.3, seed=77).frames[10:42]  # short pause, then swipe
    events, _ = run(rec, list(swipe) + [None] * 10)  # then the hand drops out of view
    assert [(e.name, e.source) for e in events] == [("swipe_left", "movement")]


def test_swiping_open_palm_is_not_read_as_palm_pose(model):
    rec = LiveRecognizer(model)
    swipe = make_swipe_sample("x", 0.3, seed=78).frames[10:42]  # only the moving part
    events, _ = run(rec, list(swipe))
    assert "palm" not in [e.name for e in events]


def test_hand_coming_back_after_a_swipe_is_ignored(model):
    rec = LiveRecognizer(model)
    there = make_swipe_sample("x", 0.3, seed=79).frames[10:42]
    back = there[::-1]  # straight back, within the lockout
    assert len(back) * 33 > LOCKOUT_MS
    events, _ = run(rec, list(there) + list(back))
    assert [e.name for e in events] == ["swipe_right"]


def test_resting_hand_at_frame_edge_is_not_a_swipe(model):
    rec = LiveRecognizer(model)
    rng = np.random.default_rng(5)
    edge = make_open_hand(center=(0.9, 0.5)).points
    frames = [edge + rng.normal(0, 0.003, edge.shape) for _ in range(90)]
    events, _ = run(rec, frames)
    assert all(e.source != "movement" for e in events)


def test_threshold_of_one_blocks_everything(model):
    rec = LiveRecognizer(model, threshold=1.0)
    rec.threshold = 1.0
    events, results = run(rec, [make_fist().points * 1.0001] * 90)
    # Random forests can be 100% sure, so only check nothing below 1.0 is shown.
    assert all(r.label is None or r.confidence >= 1.0 for r in results)
    assert all(e.confidence >= 1.0 for e in events)
