"""Main window: live camera view with hand landmarks, FPS and camera selection."""

from collections.abc import Callable

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gestureflow import __version__
from gestureflow.camera.capture import CameraInfo, camera_help, list_cameras
from gestureflow.ui.capture_worker import CaptureWorker

PRIVACY_NOTE = "Video is processed on this computer only and is never saved."


class MainWindow(QMainWindow):
    def __init__(
        self,
        find_cameras: Callable[[], list[CameraInfo]] = list_cameras,
        make_worker: Callable[..., CaptureWorker] = CaptureWorker,
        initial_camera: int | None = None,
    ) -> None:
        super().__init__()
        self._find_cameras = find_cameras
        self._make_worker = make_worker
        self._initial_camera = initial_camera
        self._thread: QThread | None = None
        self._worker: CaptureWorker | None = None

        self.setWindowTitle(f"GestureFlow AI {__version__}")
        self.resize(820, 700)
        self._build_ui()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        self.camera_box = QComboBox()
        self.camera_box.setMinimumWidth(140)
        self.camera_box.currentIndexChanged.connect(self._on_camera_changed)

        self.refresh_button = QPushButton("Find cameras")
        self.refresh_button.clicked.connect(self.refresh_cameras)

        self.mirror_box = QCheckBox("Mirror view")
        self.mirror_box.setChecked(True)
        self.mirror_box.toggled.connect(self._restart_capture)

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        top.addWidget(self.camera_box)
        top.addWidget(self.refresh_button)
        top.addWidget(self.mirror_box)
        top.addStretch()

        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setWordWrap(True)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet("background: #111; color: #eee; padding: 24px;")

        self.retry_button = QPushButton("Try again")
        self.retry_button.clicked.connect(self.refresh_cameras)
        self.retry_button.hide()

        self.fps_label = QLabel("FPS: --")
        self.hand_label = QLabel("No hand")
        privacy = QLabel(PRIVACY_NOTE)
        privacy.setStyleSheet("color: gray;")

        bottom = QHBoxLayout()
        bottom.addWidget(self.fps_label)
        bottom.addSpacing(24)
        bottom.addWidget(self.hand_label)
        bottom.addStretch()
        bottom.addWidget(privacy)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.retry_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(bottom)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------- cameras

    def refresh_cameras(self) -> None:
        """Look for cameras and start the first one (or the one asked for)."""
        self._stop_capture()
        self.video_label.setText("Looking for cameras...")
        self.retry_button.hide()

        cameras = self._find_cameras()

        # Signals are blocked while the list is rebuilt, otherwise every added item
        # (and the final selection) would start a camera of its own.
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        for cam in cameras:
            self.camera_box.addItem(cam.name, cam.index)
        if cameras:
            wanted = self._initial_camera if self._initial_camera is not None else cameras[0].index
            self.camera_box.setCurrentIndex(max(self.camera_box.findData(wanted), 0))
        self.camera_box.blockSignals(False)
        self._initial_camera = None

        if not cameras:
            self.show_error(f"No camera found. {camera_help()}")
            return
        self._start_capture()

    @Slot(int)
    def _on_camera_changed(self, _position: int) -> None:
        self._restart_capture()

    # ------------------------------------------------------------- capture

    def _restart_capture(self) -> None:
        self._start_capture()

    def _start_capture(self) -> None:
        self._stop_capture()  # never run two cameras at once
        index = self.camera_box.currentData()
        if index is None:
            return
        self.retry_button.hide()
        self.video_label.setText(f"Opening camera {index}...")

        self._thread = QThread(self)
        self._worker = self._make_worker(index, mirror=self.mirror_box.isChecked())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self.show_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _stop_capture(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None

    @property
    def is_capturing(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ------------------------------------------------------------- display

    @Slot(QImage, float, str)
    def _on_frame(self, image: QImage, fps: float, hands: str) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)
        self.fps_label.setText(f"FPS: {fps:.0f}")
        self.hand_label.setText(hands)

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.video_label.clear()
        self.video_label.setText(message)
        self.fps_label.setText("FPS: --")
        self.hand_label.setText("")
        self.retry_button.show()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's method name)
        self._stop_capture()
        super().closeEvent(event)
