"""Camera tests with a fake cv2.VideoCapture, so no webcam is needed."""

import numpy as np
import pytest

from gestureflow.camera import capture
from gestureflow.camera.capture import Camera, CameraError, list_cameras


class FakeVideoCapture:
    available = {0, 1}  # camera indices that "exist"
    frames_ok = True

    def __init__(self, index, backend=None):
        self.index = index
        self.released = False

    def isOpened(self):  # noqa: N802 (OpenCV's name)
        return self.index in self.available

    def set(self, prop, value):
        return True

    def read(self):
        if self.frames_ok:
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeVideoCapture.available = {0, 1}
    FakeVideoCapture.frames_ok = True
    monkeypatch.setattr(capture.cv2, "VideoCapture", FakeVideoCapture)
    return FakeVideoCapture


def test_list_cameras(fake_cv2):
    assert [c.index for c in list_cameras()] == [0, 1]
    assert list_cameras()[1].name == "Camera 1"


def test_list_cameras_none(fake_cv2):
    fake_cv2.available = set()
    assert list_cameras() == []


def test_open_missing_camera_raises_helpful_error(fake_cv2):
    with pytest.raises(CameraError, match="Could not open camera 5"):
        Camera(5)


def test_read_returns_frames(fake_cv2):
    with Camera(0) as cam:
        assert cam.read().shape == (480, 640, 3)


def test_occasional_failures_are_tolerated_but_persistent_ones_raise(fake_cv2):
    fake_cv2.frames_ok = False
    with Camera(0) as cam:
        for _ in range(Camera.MAX_CONSECUTIVE_FAILURES - 1):
            assert cam.read() is None
        with pytest.raises(CameraError, match="stopped sending images"):
            cam.read()
