import pytest

from gestureflow.training.recorder import (
    COUNTDOWN_MS,
    DYNAMIC_DURATION_MS,
    RecorderState,
    SampleRecorder,
)
from tests.conftest import make_fist, make_open_hand, make_swipe

FRAME_MS = 33


def feed(recorder, hands_at, total_ms):
    """Feed frames every 33 ms; hands_at(t) returns the hands seen at time t."""
    status = None
    for t in range(0, total_ms, FRAME_MS):
        status = recorder.update(hands_at(t), 10_000 + t)
        if recorder.finished:
            break
    return status


def test_static_recording_succeeds():
    rec = SampleRecorder("Thumbs Up", "static")
    rec.start()
    first = rec.update([make_open_hand()], 10_000)
    assert first.state == RecorderState.COUNTDOWN and "3" in first.message
    status = feed(rec, lambda t: [make_open_hand()], COUNTDOWN_MS + 2000)
    assert status.state == RecorderState.DONE
    assert rec.result.gesture == "thumbs_up"
    assert 25 <= len(rec.result.frames) <= 32  # about one second of frames


def test_countdown_frames_are_not_recorded():
    rec = SampleRecorder("x", "static")
    rec.start()
    feed(rec, lambda t: [make_open_hand()], COUNTDOWN_MS + 2000)
    assert rec.result.timestamps_ms[0] >= 10_000 + COUNTDOWN_MS


def test_static_without_hand_fails():
    rec = SampleRecorder("x", "static", countdown_ms=0)
    rec.start()
    status = feed(rec, lambda t: [], 2000)
    assert status.state == RecorderState.FAILED
    assert "hand" in status.message.lower()


def swipe_hands(dx):
    frames, _ = make_swipe(dx=dx, frames=61)  # 2 s at 30 fps

    def at(t):
        i = min(t // FRAME_MS, len(frames) - 1)
        return [make_open_hand().__class__(frames[i], "Right", 0.9)]

    return at


def test_dynamic_swipe_succeeds():
    rec = SampleRecorder("swipe_right", "dynamic", countdown_ms=0)
    rec.start()
    status = feed(rec, swipe_hands(0.3), DYNAMIC_DURATION_MS + 500)
    assert status.state == RecorderState.DONE
    assert rec.result.kind == "dynamic"


def test_dynamic_without_motion_fails():
    rec = SampleRecorder("swipe_right", "dynamic", countdown_ms=0)
    rec.start()
    status = feed(rec, swipe_hands(0.0), DYNAMIC_DURATION_MS + 500)
    assert status.state == RecorderState.FAILED
    assert "barely moved" in status.message


def test_failure_message_says_how_far_the_hand_moved():
    rec = SampleRecorder("screenshot", "dynamic", countdown_ms=0)
    rec.start()
    status = feed(rec, swipe_hands(0.0), DYNAMIC_DURATION_MS + 500)
    assert "palm 0%" in status.message and "fingers 0%" in status.message


def test_dynamic_grab_in_place_succeeds():
    rec = SampleRecorder("screenshot", "dynamic", countdown_ms=0)
    rec.start()
    open_pts, fist_pts = make_open_hand().points, make_fist().points

    def closing(t):
        f = min(t / 1000, 1.0)
        return [make_open_hand().__class__(open_pts + (fist_pts - open_pts) * f, "Right", 0.9)]

    assert feed(rec, closing, DYNAMIC_DURATION_MS + 500).state == RecorderState.DONE


def test_dynamic_neutral_needs_no_motion():
    rec = SampleRecorder("neutral", "dynamic", countdown_ms=0)
    rec.start()
    assert feed(rec, swipe_hands(0.0), DYNAMIC_DURATION_MS + 500).state == RecorderState.DONE


def test_dynamic_fails_if_hand_disappears():
    rec = SampleRecorder("swipe_right", "dynamic", countdown_ms=0)
    rec.start()
    moving = swipe_hands(0.3)
    status = feed(rec, lambda t: moving(t) if t < 400 or t > 900 else [], 2000)
    assert status.state == RecorderState.FAILED
    assert "left the camera view" in status.message


def test_cancel_and_restart():
    rec = SampleRecorder("x", "static", countdown_ms=0)
    rec.start()
    rec.update([make_open_hand()], 0)
    rec.cancel()
    assert rec.state == RecorderState.IDLE
    rec.start()
    assert feed(rec, lambda t: [make_open_hand()], 2000).state == RecorderState.DONE


def test_rejects_bad_kind():
    with pytest.raises(ValueError):
        SampleRecorder("x", "wiggle")
