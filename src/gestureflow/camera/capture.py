"""Webcam access with OpenCV.

Knows nothing about hands: it opens a camera and hands out frames.
"""

import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Requested capture size. 640x480 is supported by almost every webcam and keeps
# hand detection fast; MediaPipe doesn't need more detail than this.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CameraError(RuntimeError):
    """A camera could not be opened or stopped delivering frames."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str


def _backend() -> int:
    """The OpenCV capture backend that works best on each OS."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        # DirectShow opens much faster than the default (MSMF) on most Windows webcams.
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def camera_help() -> str:
    """OS-specific advice shown when no camera works."""
    if sys.platform == "darwin":
        return (
            "Check that your camera is not used by another app (Zoom, FaceTime, ...). "
            "If this is the first run, macOS may have asked for camera access: allow it in "
            "System Settings > Privacy & Security > Camera for the app running Python "
            "(Terminal, iTerm or VS Code), then quit and reopen that app."
        )
    if sys.platform == "win32":
        return (
            "Check that your camera is connected and not used by another app (Teams, Zoom, ...). "
            "Make sure Settings > Privacy & security > Camera allows desktop apps to use it."
        )
    return "Check that your camera is connected and not used by another app."


def list_cameras(max_cameras: int = 5) -> list[CameraInfo]:
    """Find cameras by trying indices 0, 1, 2, ... until one fails to open.

    OpenCV cannot read camera names, so they are labeled "Camera 0", "Camera 1", ...
    Opening a camera can take a moment, so call this once, not every frame.
    """
    found = []
    # Probing past the last camera always fails, and OpenCV prints scary-looking
    # errors when it does. Silence OpenCV's log while probing, then restore it.
    previous_level = cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        for index in range(max_cameras):
            cap = cv2.VideoCapture(index, _backend())
            try:
                if not cap.isOpened():
                    break
                found.append(CameraInfo(index=index, name=f"Camera {index}"))
            finally:
                cap.release()
    finally:
        cv2.utils.logging.setLogLevel(previous_level)
    return found


class Camera:
    """An open webcam. Use as a context manager so the camera is always released."""

    # How many failed reads in a row we tolerate before reporting an error.
    # Webcams occasionally drop a frame, especially right after opening.
    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, index: int = 0, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.index = index
        self._cap = cv2.VideoCapture(index, _backend())
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"Could not open camera {index}. {camera_help()}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._failures = 0

    def read(self) -> np.ndarray | None:
        """Return the next frame as a BGR array, or None if this read failed.

        Raises CameraError after too many failures in a row (camera unplugged,
        taken over by another app, or access denied).
        """
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._failures = 0
            return frame
        self._failures += 1
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
            raise CameraError(f"Camera {self.index} stopped sending images. {camera_help()}")
        return None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
