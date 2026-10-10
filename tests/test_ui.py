"""UI tests: run Qt offscreen with fake cameras and a fake detector."""

import time
from pathlib import Path

import numpy as np
import pytest

from gestureflow.camera.capture import CameraError, CameraInfo
from gestureflow.storage.samples import count_samples, load_samples
from gestureflow.ui.capture_worker import CaptureWorker, describe
from gestureflow.ui.main_window import MainWindow
from gestureflow.ui.record_panel import RecordPanel
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
    window = MainWindow(
        find_cameras=lambda: [], record_panel=RecordPanel(root=Path("/nonexistent"))
    )
    window.refresh_cameras()
    assert "No camera found" in window.video_label.text()
    assert not window.retry_button.isHidden()
    assert not window.is_capturing
    window.close()


@pytest.mark.parametrize("camera_cls", [FakeCamera])
def test_window_shows_live_frames(qapp, camera_cls):
    cameras = [CameraInfo(0, "Camera 0"), CameraInfo(1, "Camera 1")]
    window = MainWindow(
        find_cameras=lambda: cameras,
        make_worker=fake_worker_factory(camera_cls),
        record_panel=RecordPanel(root=Path("/nonexistent")),
    )
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
        record_panel=RecordPanel(root=Path("/nonexistent")),
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


def test_record_panel_saves_static_example(qapp, tmp_path):
    panel = RecordPanel(root=tmp_path, countdown_ms=0)
    panel.name_box.setEditText("Thumbs Up")
    panel.start_recording()
    assert panel.is_recording and not panel.record_button.isEnabled()
    for i in range(40):  # about 1.3 s of frames at 30 fps
        panel.on_hands([make_open_hand()], 1000 + i * 33)
    assert not panel.is_recording
    assert count_samples(tmp_path) == {"thumbs_up": 1}
    assert "Saved example 1 of 'thumbs_up'" in panel.status_label.text()
    (sample,), _ = load_samples(root=tmp_path)
    assert sample.kind == "static"


def test_record_panel_reports_failure_and_saves_nothing(qapp, tmp_path):
    panel = RecordPanel(root=tmp_path, countdown_ms=0)
    panel.name_box.setEditText("swipe_left")
    panel.dynamic_radio.setChecked(True)
    panel.start_recording()
    for i in range(75):  # 2.5 s of a hand that doesn't move
        panel.on_hands([make_open_hand()], 1000 + i * 33)
    assert "barely moved" in panel.status_label.text()
    assert count_samples(tmp_path) == {}
    assert panel.record_button.isEnabled()


def test_record_panel_rejects_bad_name(qapp, tmp_path):
    panel = RecordPanel(root=tmp_path)
    panel.name_box.setEditText("!!!")
    panel.start_recording()
    assert not panel.is_recording
    assert "not a valid gesture name" in panel.status_label.text()


def test_record_panel_undo(qapp, tmp_path):
    panel = RecordPanel(root=tmp_path, countdown_ms=0)
    panel.name_box.setEditText("ok")
    for _ in range(2):
        panel.start_recording()
        for i in range(40):
            panel.on_hands([make_open_hand()], 1000 + i * 33)
    assert count_samples(tmp_path) == {"ok": 2}
    panel.undo_last()
    assert count_samples(tmp_path) == {"ok": 1}


def test_record_panel_ignores_frames_when_idle(qapp, tmp_path):
    panel = RecordPanel(root=tmp_path, countdown_ms=0)
    panel.on_hands([make_open_hand()], 0)
    assert count_samples(tmp_path) == {}
