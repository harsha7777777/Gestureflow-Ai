"""Background worker: camera -> hand detection -> drawing, off the UI thread.

Reading the camera and running MediaPipe take a few milliseconds per frame. Doing
that on the UI thread would freeze the window, so it runs on a QThread and sends
finished images to the window through Qt signals.
"""

import threading
import time
from collections.abc import Callable

import cv2
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from gestureflow.camera.capture import Camera, CameraError
from gestureflow.camera.fps import FpsCounter
from gestureflow.detection.hand_detector import DetectorError, HandDetector
from gestureflow.ui.drawing import draw_hands


class CaptureWorker(QObject):
    # image to show, current FPS, short description of what was detected
    frame_ready = Signal(QImage, float, str)
    # a user-facing error message; the worker stops after sending it
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        camera_index: int,
        mirror: bool = True,
        open_camera: Callable[[int], Camera] = Camera,
        make_detector: Callable[[], HandDetector] = HandDetector,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.mirror = mirror
        # Factories are parameters so tests can pass fakes instead of real hardware.
        self._open_camera = open_camera
        self._make_detector = make_detector
        # An Event (not a bool reset inside run) so a stop() that arrives before
        # run() has even started is never lost.
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        """Ask the loop to finish after the current frame (safe from any thread)."""
        self._stop_requested.set()

    def run(self) -> None:
        fps = FpsCounter()
        try:
            with self._open_camera(self.camera_index) as camera, self._make_detector() as detector:
                while not self._stop_requested.is_set():
                    frame = camera.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    if self.mirror:
                        # Selfie view: moving your hand right moves it right on screen.
                        frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hands = detector.detect(rgb, time.monotonic_ns() // 1_000_000)
                    draw_hands(rgb, hands)
                    self.frame_ready.emit(to_qimage(rgb), fps.tick(), describe(hands))
                    # The frame is discarded here: nothing is ever written to disk.
        except (CameraError, DetectorError) as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # keep the app alive and tell the user what happened
            self.error.emit(f"Unexpected error while reading the camera: {exc}")
        finally:
            self.finished.emit()


def to_qimage(rgb) -> QImage:
    height, width = rgb.shape[:2]
    # .copy() makes Qt own the pixels; the numpy array is reused for the next frame.
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def describe(hands) -> str:
    if not hands:
        return "No hand"
    return ", ".join(f"{h.handedness} hand" for h in hands)
