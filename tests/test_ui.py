"""UI tests: run Qt offscreen with fake cameras and a fake detector."""

import time

import numpy as np
import pytest

from gestureflow.camera.capture import CameraError, CameraInfo
from gestureflow.ui.capture_worker import CaptureWorker, describe
from gestureflow.ui.main_window import MainWindow
from tests.conftest import make_open_hand


class FakeCamera:
    def __init__(self, index):
        self.index = index

    def read(self):
        time.sleep(0.005)
        return np.full((120, 160, 3), 60, dtype=np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class BrokenCamera(FakeCamera):
    def __init__(self, index):
        raise CameraError("Could not open camera 0. (test)")


class FakeDetector:
    def detect(self, frame, timestamp_ms):
        return [make_open_hand()]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def fake_worker_factory(camera_cls):
    def make(index, mirror=True):
        return CaptureWorker(index, mirror, open_camera=camera_cls, make_detector=FakeDetector)

    return make


def wait_until(qapp, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_describe():
    assert describe([]) == "No hand"
    assert describe([make_open_hand()]) == "Right hand"


def test_worker_emits_frames_and_stops(qapp):
    worker = CaptureWorker(0, open_camera=FakeCamera, make_detector=FakeDetector)
    frames = []

    def on_frame(image, fps, hands):
        frames.append((image, fps, hands))
        if len(frames) == 5:
            worker.stop()

    worker.frame_ready.connect(on_frame)
    worker.run()  # runs in this thread; returns once stop() is called
    assert len(frames) == 5
    image, _, hands = frames[-1]
    assert (image.width(), image.height()) == (160, 120)
    assert hands == "Right hand"


def test_worker_reports_camera_errors(qapp):
    worker = CaptureWorker(0, open_camera=BrokenCamera, make_detector=FakeDetector)
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors == ["Could not open camera 0. (test)"]


def test_window_without_camera_shows_help(qapp):
    window = MainWindow(find_cameras=lambda: [])
    window.refresh_cameras()
    assert "No camera found" in window.video_label.text()
    assert not window.retry_button.isHidden()
    assert not window.is_capturing
    window.close()


@pytest.mark.parametrize("camera_cls", [FakeCamera])
def test_window_shows_live_frames(qapp, camera_cls):
    cameras = [CameraInfo(0, "Camera 0"), CameraInfo(1, "Camera 1")]
    window = MainWindow(find_cameras=lambda: cameras, make_worker=fake_worker_factory(camera_cls))
    window.show()
    window.refresh_cameras()
    assert window.camera_box.count() == 2
    assert wait_until(
        qapp,
        lambda: (
            window.video_label.pixmap() is not None and not window.video_label.pixmap().isNull()
        ),
    )
    assert wait_until(qapp, lambda: window.fps_label.text() not in ("FPS: --", "FPS: 0"))
    assert window.hand_label.text() == "Right hand"

    # Switching cameras restarts capture cleanly.
    window.camera_box.setCurrentIndex(1)
    assert window.is_capturing
    window.close()
    assert not window.is_capturing


def test_window_shows_camera_error(qapp):
    window = MainWindow(
        find_cameras=lambda: [CameraInfo(0, "Camera 0")],
        make_worker=fake_worker_factory(BrokenCamera),
    )
    window.refresh_cameras()
    assert wait_until(qapp, lambda: "Could not open camera" in window.video_label.text())
    assert wait_until(qapp, lambda: not window.retry_button.isHidden())
    window.close()


def test_stop_before_run_is_not_lost(qapp):
    # Switching cameras quickly can call stop() before the thread has started run().
    worker = CaptureWorker(0, open_camera=FakeCamera, make_detector=FakeDetector)
    worker.stop()
    worker.run()  # must return immediately instead of looping forever
