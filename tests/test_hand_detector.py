import os
from types import SimpleNamespace

import numpy as np
import pytest

from gestureflow.detection.hand_detector import DetectorError, HandDetector, convert_result
from gestureflow.detection.model_file import is_model_present


def fake_result(n_points=21, label="Left", score=0.9):
    landmarks = [SimpleNamespace(x=i / 21, y=0.5, z=-0.01 * i) for i in range(n_points)]
    category = SimpleNamespace(category_name=label, score=score)
    return SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]])


def test_convert_result():
    (hand,) = convert_result(fake_result())
    assert hand.points.shape == (21, 3)
    assert hand.points[20, 0] == pytest.approx(20 / 21)
    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.9)


def test_convert_empty_result():
    assert convert_result(SimpleNamespace(hand_landmarks=[], handedness=[])) == []


def test_convert_skips_malformed_hand():
    assert convert_result(fake_result(n_points=5)) == []


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(DetectorError, match="--download-model"):
        HandDetector(model=tmp_path / "missing.task")


@pytest.mark.skipif(not is_model_present(), reason="hand model not downloaded")
@pytest.mark.skipif(
    bool(os.environ.get("GESTUREFLOW_NO_METAL")),
    reason="MediaPipe on macOS needs a Metal GPU, which CI virtual machines lack",
)
def test_real_detector_finds_no_hand_in_blank_frame():
    with HandDetector() as detector:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame, 0) == []
        # Repeated or decreasing timestamps must not crash MediaPipe.
        assert detector.detect(frame, 0) == []
