import pytest

from gestureflow.camera.fps import FpsCounter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_zero_before_two_frames():
    counter = FpsCounter(clock=FakeClock())
    assert counter.fps == 0.0
    assert counter.tick() == 0.0


def test_steady_30_fps():
    clock = FakeClock()
    counter = FpsCounter(window=1.0, clock=clock)
    for _ in range(90):
        counter.tick()
        clock.now += 1 / 30
    assert counter.fps == pytest.approx(30, rel=0.02)


def test_reacts_to_slowdown():
    clock = FakeClock()
    counter = FpsCounter(window=1.0, clock=clock)
    for _ in range(60):
        counter.tick()
        clock.now += 1 / 30
    for _ in range(20):
        counter.tick()
        clock.now += 1 / 10
    assert counter.fps == pytest.approx(10, rel=0.1)
