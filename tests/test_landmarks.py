import numpy as np
import pytest

from gestureflow.detection.landmarks import HAND_CONNECTIONS, NUM_LANDMARKS, HandLandmarks


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        HandLandmarks(points=np.zeros((20, 3)), handedness="Left", score=1.0)


def test_connections_form_one_skeleton():
    assert len(HAND_CONNECTIONS) == 21
    used = {i for pair in HAND_CONNECTIONS for i in pair}
    assert used == set(range(NUM_LANDMARKS))


def test_synthetic_hand_fixture_is_valid(open_hand):
    assert open_hand.points.shape == (21, 3)
    assert np.all((open_hand.points[:, :2] >= 0) & (open_hand.points[:, :2] <= 1))
