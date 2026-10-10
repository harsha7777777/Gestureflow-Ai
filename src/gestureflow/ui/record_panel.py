"""Panel for recording gesture examples (landmarks only, never video)."""

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from gestureflow.storage.samples import (
    NEUTRAL,
    SampleError,
    clean_gesture_name,
    count_samples,
    delete_last_sample,
    save_sample,
)
from gestureflow.training.recorder import RecorderState, SampleRecorder

TIPS = (
    "Record 5–15 examples of each gesture, moving your hand a little between examples "
    '(position, distance, angle). Also record examples named "neutral": your hand '
    "doing ordinary things (resting, typing, waving casually), so the app learns when "
    "to do nothing."
)


class RecordPanel(QGroupBox):
    def __init__(self, root: Path | None = None, countdown_ms: int | None = None) -> None:
        super().__init__("Record gesture examples")
        self._root = root  # None = the normal user_data/gestures folder
        self._countdown_ms = countdown_ms
        self._recorder: SampleRecorder | None = None
        self._build_ui()
        self.refresh_counts()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        self.name_box = QComboBox()
        self.name_box.setEditable(True)
        self.name_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.name_box.lineEdit().setPlaceholderText("e.g. thumbs_up or swipe_left")

        self.static_radio = QRadioButton("Pose (hold still)")
        self.dynamic_radio = QRadioButton("Movement (swipe)")
        self.static_radio.setChecked(True)
        kind_group = QButtonGroup(self)
        kind_group.addButton(self.static_radio)
        kind_group.addButton(self.dynamic_radio)
        kinds = QHBoxLayout()
        kinds.addWidget(self.static_radio)
        kinds.addWidget(self.dynamic_radio)

        self.record_button = QPushButton("Record example")
        self.record_button.clicked.connect(self.start_recording)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_recording)
        self.cancel_button.setEnabled(False)
        self.undo_button = QPushButton("Undo last")
        self.undo_button.clicked.connect(self.undo_last)
        buttons = QHBoxLayout()
        buttons.addWidget(self.record_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.undo_button)

        self.status_label = QLabel("Choose a name, then click Record example.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)

        self.counts_list = QListWidget()
        tips = QLabel(TIPS)
        tips.setWordWrap(True)
        tips.setStyleSheet("color: gray;")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Gesture name:"))
        layout.addWidget(self.name_box)
        layout.addLayout(kinds)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Saved examples:"))
        layout.addWidget(self.counts_list, stretch=1)
        layout.addWidget(tips)
        self.setLayout(layout)
        self.setMinimumWidth(300)

    # ------------------------------------------------------------------ actions

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and not self._recorder.finished

    def refresh_counts(self) -> None:
        counts = count_samples(self._root)
        current = self.name_box.currentText()
        self.name_box.blockSignals(True)
        self.name_box.clear()
        self.name_box.addItems(sorted(set(counts) | {NEUTRAL}))
        self.name_box.setEditText(current)
        self.name_box.blockSignals(False)

        self.counts_list.clear()
        if not counts:
            self.counts_list.addItem("(none yet)")
        for name, n in counts.items():
            self.counts_list.addItem(f"{name}: {n} example{'s' if n != 1 else ''}")
        if NEUTRAL not in counts:
            self.counts_list.addItem(f"{NEUTRAL}: 0 (recommended: record some!)")

    def start_recording(self) -> None:
        kind = "static" if self.static_radio.isChecked() else "dynamic"
        try:
            name = clean_gesture_name(self.name_box.currentText())
        except SampleError as exc:
            self._show_status(str(exc))
            return
        self.name_box.setEditText(name)
        options = {} if self._countdown_ms is None else {"countdown_ms": self._countdown_ms}
        self._recorder = SampleRecorder(name, kind, **options)
        self._recorder.start()
        self._set_recording_ui(True)
        self._show_status("Get ready...", 0.0)

    def cancel_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.cancel()
        self._recorder = None
        self._set_recording_ui(False)
        self._show_status("Recording cancelled.")

    def undo_last(self) -> None:
        try:
            name = clean_gesture_name(self.name_box.currentText())
        except SampleError as exc:
            self._show_status(str(exc))
            return
        deleted = delete_last_sample(name, self._root)
        self._show_status(
            f"Deleted the last example of '{name}'." if deleted else f"No examples of '{name}'."
        )
        self.refresh_counts()

    @Slot(object, int)
    def on_hands(self, hands, t_ms: int) -> None:
        """Called for every camera frame with the detected hands."""
        if not self.is_recording:
            return
        status = self._recorder.update(hands, t_ms)
        self._show_status(status.message, status.progress)
        if status.state == RecorderState.DONE:
            self._save(self._recorder)
        if self._recorder.finished:
            self._set_recording_ui(False)

    # ------------------------------------------------------------------ helpers

    def _save(self, recorder: SampleRecorder) -> None:
        try:
            save_sample(recorder.result, self._root)
        except (SampleError, OSError) as exc:
            QMessageBox.warning(self, "Could not save example", str(exc))
            return
        count = count_samples(self._root).get(recorder.gesture, 0)
        self._show_status(f"Saved example {count} of '{recorder.gesture}'.", 1.0)
        self.refresh_counts()

    def _set_recording_ui(self, recording: bool) -> None:
        self.record_button.setEnabled(not recording)
        self.undo_button.setEnabled(not recording)
        self.cancel_button.setEnabled(recording)
        self.name_box.setEnabled(not recording)
        self.static_radio.setEnabled(not recording)
        self.dynamic_radio.setEnabled(not recording)

    def _show_status(self, message: str, progress: float | None = None) -> None:
        self.status_label.setText(message)
        if progress is not None:
            self.progress.setValue(round(progress * 100))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
