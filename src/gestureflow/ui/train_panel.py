"""Panel for training the model and testing recognition live.

Training runs on a background thread (it takes a few seconds) so the camera
view keeps running. Live test only *shows* what is recognized: no action is
ever triggered from this panel.
"""

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gestureflow.recognition.recognizer import LiveRecognizer
from gestureflow.storage.samples import load_samples
from gestureflow.training.model import GestureModel, ModelError, load_model, save_model
from gestureflow.training.trainer import Evaluation, TrainingReport, train

DEFAULT_THRESHOLD = 0.70
MAX_EVENTS_SHOWN = 8


class TrainingWorker(QObject):
    progress = Signal(str)
    done = Signal(object, object, object)  # model, report, load errors
    failed = Signal(str)
    finished = Signal()

    def __init__(self, samples_root: Path | None, model_root: Path | None) -> None:
        super().__init__()
        self._samples_root = samples_root
        self._model_root = model_root

    def run(self) -> None:
        try:
            samples, errors = load_samples(root=self._samples_root)
            model, report = train(samples, progress=self.progress.emit)
            save_model(model, self._model_root)
            self.done.emit(model, report, errors)
        except ValueError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # keep the app alive, tell the user
            self.failed.emit(f"Training failed unexpectedly: {exc}")
        finally:
            self.finished.emit()


class TrainPanel(QWidget):
    def __init__(self, samples_root: Path | None = None, model_root: Path | None = None) -> None:
        super().__init__()
        self._samples_root = samples_root
        self._model_root = model_root
        self._model: GestureModel | None = None
        self._recognizer: LiveRecognizer | None = None
        self._thread: QThread | None = None
        self._worker: TrainingWorker | None = None
        self._build_ui()
        self._load_existing_model()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.start_training)
        self.train_status = QLabel()
        self.train_status.setWordWrap(True)
        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #c77700;")
        self.pose_table = QTableWidget()
        self.movement_table = QTableWidget()
        self.pose_title = QLabel()
        self.movement_title = QLabel()
        for table in (self.pose_table, self.movement_table):
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.hide()

        train_box = QGroupBox("1. Train on your examples")
        train_layout = QVBoxLayout()
        train_layout.addWidget(self.train_button)
        train_layout.addWidget(self.train_status)
        train_layout.addWidget(self.results_label)
        train_layout.addWidget(self.warnings_label)
        train_box.setLayout(train_layout)

        self.details_box = QGroupBox("3. Mix-ups on held-out examples")
        details_layout = QVBoxLayout()
        details_layout.addWidget(self.pose_title)
        details_layout.addWidget(self.pose_table)
        details_layout.addWidget(self.movement_title)
        details_layout.addWidget(self.movement_table)
        self.details_box.setLayout(details_layout)
        self.details_box.hide()

        self.live_button = QPushButton("Start live test")
        self.live_button.setCheckable(True)
        self.live_button.toggled.connect(self.set_live)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 99)
        self.threshold_slider.setValue(round(DEFAULT_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold)
        self.threshold_label = QLabel()
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Confidence needed:"))
        threshold_row.addWidget(self.threshold_slider, stretch=1)
        threshold_row.addWidget(self.threshold_label)

        self.seeing_label = QLabel("Seeing: -")
        self.seeing_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.recognized_label = QLabel("Recognized: -")
        self.recognized_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2e8b57;")
        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(120)
        note = QLabel("Live test only shows what is recognized. It never runs any action.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")

        live_box = QGroupBox("2. Test live")
        live_layout = QVBoxLayout()
        live_layout.addWidget(self.live_button)
        live_layout.addLayout(threshold_row)
        live_layout.addWidget(self.seeing_label)
        live_layout.addWidget(self.confidence_bar)
        live_layout.addWidget(self.recognized_label)
        live_layout.addWidget(self.events_list)
        live_layout.addWidget(note)
        live_box.setLayout(live_layout)

        layout = QVBoxLayout()
        layout.addWidget(train_box)
        layout.addWidget(live_box)
        layout.addWidget(self.details_box)
        layout.addStretch()
        self.setLayout(layout)
        self._on_threshold(self.threshold_slider.value())

    # ------------------------------------------------------------------ model

    @property
    def threshold(self) -> float:
        return self.threshold_slider.value() / 100

    def _load_existing_model(self) -> None:
        try:
            self._set_model(load_model(self._model_root))
            self.train_status.setText(
                f"Using the model trained {self._model.trained_at.replace('T', ' at ')[:19]}. "
                "Train again after recording new examples."
            )
        except ModelError as exc:
            self._set_model(None)
            self.train_status.setText(str(exc))

    def _set_model(self, model: GestureModel | None) -> None:
        self._model = model
        self._recognizer = LiveRecognizer(model, self.threshold) if model else None
        self.live_button.setEnabled(model is not None)
        if model is None and self.live_button.isChecked():
            self.live_button.setChecked(False)

    @property
    def is_training(self) -> bool:
        return self._thread is not None

    def start_training(self) -> None:
        if self.is_training:
            return
        self.live_button.setChecked(False)
        self.train_button.setEnabled(False)
        self.train_status.setText("Loading examples...")
        self._thread = QThread(self)
        self._worker = TrainingWorker(self._samples_root, self._model_root)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.train_status.setText)
        self._worker.done.connect(self._on_trained)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._thread.start()

    @Slot(object, object, object)
    def _on_trained(self, model: GestureModel, report: TrainingReport, errors: list) -> None:
        self._set_model(model)
        self.train_status.setText("Training finished and saved. Try it with Start live test.")
        self.show_report(report, errors)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.train_status.setText(message)

    @Slot()
    def _on_worker_finished(self) -> None:
        self._thread.quit()
        self._thread.wait(5000)
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = self._worker = None
        self.train_button.setEnabled(True)

    def show_report(self, report: TrainingReport, errors: list | None = None) -> None:
        self.results_label.setText("\n".join(report.summary_lines()))
        notes = list(report.warnings) + [f"Skipped a damaged file: {e}" for e in errors or []]
        self.warnings_label.setText("\n".join(f"• {n}" for n in notes))
        _fill_confusion(self.pose_table, self.pose_title, "Poses", report.pose)
        _fill_confusion(self.movement_table, self.movement_title, "Movements", report.movement)
        self.details_box.setVisible(
            not (self.pose_table.isHidden() and self.movement_table.isHidden())
        )

    # ------------------------------------------------------------------ live test

    @property
    def is_live(self) -> bool:
        return self.live_button.isChecked() and self._recognizer is not None

    @Slot(bool)
    def set_live(self, on: bool) -> None:
        self.live_button.setText("Stop live test" if on else "Start live test")
        if self._recognizer is not None:
            self._recognizer.reset()
        if not on:
            self.seeing_label.setText("Seeing: -")
            self.confidence_bar.setValue(0)

    @Slot(int)
    def _on_threshold(self, value: int) -> None:
        self.threshold_label.setText(f"{value}%")
        if self._recognizer is not None:
            self._recognizer.threshold = value / 100

    @Slot(object, int)
    def on_hands(self, hands, t_ms: int) -> None:
        """Called for every camera frame with the detected hands."""
        if not self.is_live:
            return
        result = self._recognizer.update(hands, t_ms)
        if not hands:
            text = "no hand"
        elif result.label is None:
            text = "not sure"
        else:
            text = f"{result.label} ({result.source})"
        self.seeing_label.setText(f"Seeing: {text}")
        self.confidence_bar.setValue(round(result.confidence * 100) if hands else 0)
        if result.event is not None:
            ev = result.event
            stamp = datetime.now().strftime("%H:%M:%S")
            self.recognized_label.setText(f"Recognized: {ev.name} ({ev.confidence:.0%})")
            self.events_list.insertItem(0, f"{stamp}  {ev.name}  {ev.confidence:.0%}  {ev.source}")
            while self.events_list.count() > MAX_EVENTS_SHOWN:
                self.events_list.takeItem(self.events_list.count() - 1)

    def shutdown(self) -> None:
        """Wait for a running training to finish (called when the window closes)."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30000)


def _fill_confusion(table: QTableWidget, title: QLabel, name: str, ev: Evaluation | None) -> None:
    if ev is None or ev.confusion is None:
        table.hide()
        title.setText("")
        return
    classes = ev.classes
    table.setRowCount(len(classes))
    table.setColumnCount(len(classes))
    table.setVerticalHeaderLabels(classes)
    table.setHorizontalHeaderLabels(classes)
    for r in range(len(classes)):
        total = max(int(ev.confusion[r].sum()), 1)
        for c in range(len(classes)):
            n = int(ev.confusion[r, c])
            item = QTableWidgetItem(str(n))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if n:
                strength = 60 + int(150 * n / total)
                color = QColor(46, 139, 87, strength) if r == c else QColor(200, 60, 60, strength)
                item.setBackground(color)
            table.setItem(r, c, item)
    title.setText(
        f"{name}: rows = what you recorded, columns = what the model guessed. "
        "Numbers off the green diagonal are mix-ups."
    )
    title.setWordWrap(True)
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    # Show the whole table without empty space or an inner scroll bar.
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    height += sum(table.rowHeight(r) for r in range(table.rowCount()))
    table.setFixedHeight(height)
    table.show()
