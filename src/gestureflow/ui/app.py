"""Starts the Qt application."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from gestureflow.detection.model_file import (
    MODEL_URL,
    ModelDownloadError,
    download_model,
    is_model_present,
    model_path,
)
from gestureflow.ui.main_window import MainWindow


def ensure_model(parent=None) -> bool:
    """Make sure the hand model exists, offering to download it. Returns True if ready."""
    if is_model_present():
        return True
    answer = QMessageBox.question(
        parent,
        "Download hand model?",
        "GestureFlow AI needs MediaPipe's hand landmark model (about 8 MB) to find your hand.\n\n"
        f"Download it now from Google?\n{MODEL_URL}\n\n"
        f"It will be saved to {model_path()}. This is a one-time download; no camera "
        "data is ever sent anywhere.",
    )
    if answer != QMessageBox.StandardButton.Yes:
        return False
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        download_model()
    except ModelDownloadError as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(parent, "Download failed", str(exc))
        return False
    QApplication.restoreOverrideCursor()
    return True


def run_app(camera_index: int | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GestureFlow AI")

    if not ensure_model():
        print(
            "The hand model is required. Download it with: python -m gestureflow --download-model",
            file=sys.stderr,
        )
        return 1

    window = MainWindow(initial_camera=camera_index)
    window.show()
    window.refresh_cameras()
    return app.exec()
