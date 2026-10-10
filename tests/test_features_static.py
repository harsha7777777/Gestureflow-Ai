import numpy as np
import pytest

from gestureflow.features.static import (
    NUM_STATIC_FEATURES,
    STATIC_FEATURE_NAMES,
    hand_size,
    normalize_landmarks,
    static_features,
)
from tests.conftest import make_fist, make_open_hand

EXTENSION = slice(NUM_STATIC_FEATURES - 5, NUM_STATIC_FEATURES)


def test_shape_and_names(open_hand):
    features = static_features(open_hand.points)
    assert features.shape == (NUM_STATIC_FEATURES,) == (len(STATIC_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_wrist_is_origin_and_hand_size_is_one(open_hand):
    norm = normalize_landmarks(open_hand.points)
    assert np.allclose(norm[0], 0)
    assert np.linalg.norm(norm[9, :2]) == pytest.approx(1.0)


def test_position_in_frame_does_not_matter():
    left = static_features(make_open_hand(center=(0.25, 0.4)).points)
    right = static_features(make_open_hand(center=(0.75, 0.6)).points)
    assert np.allclose(left, right, atol=1e-5)


def test_distance_to_camera_does_not_matter():
    near = static_features(make_open_hand(size=0.4).points)
    far = static_features(make_open_hand(size=0.15).points)
    assert np.allclose(near, far, atol=1e-5)


def test_aspect_ratio_is_undone(open_hand):
    # The same hand seen in a 4:3 frame has its x coordinates squeezed by 3/4.
    squeezed = open_hand.points.copy()
    squeezed[:, 0] = 0.5 + (squeezed[:, 0] - 0.5) * 0.75
    assert np.allclose(
        static_features(squeezed, aspect=4 / 3), static_features(open_hand.points), atol=1e-5
    )
    assert hand_size(squeezed, aspect=4 / 3) == pytest.approx(hand_size(open_hand.points))


def test_extension_separates_open_hand_from_fist():
    open_ext = static_features(make_open_hand().points)[EXTENSION]
    fist_ext = static_features(make_fist().points)[EXTENSION]
    assert np.all(open_ext[1:] > fist_ext[1:] + 0.3)  # index..pinky clearly differ


def test_degenerate_hand_raises():
    with pytest.raises(ValueError):
        static_features(np.full((21, 3), 0.5, dtype=np.float32))
