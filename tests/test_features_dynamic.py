import numpy as np
import pytest

from gestureflow.features.dynamic import (
    NUM_SEQUENCE_FEATURES,
    RESAMPLED_STEPS,
    SEQUENCE_FEATURE_NAMES,
    has_enough_motion,
    motion_stats,
    resample,
    sequence_features,
)
from tests.conftest import make_swipe

NET_DX = SEQUENCE_FEATURE_NAMES.index("net_dx")
NET_DY = SEQUENCE_FEATURE_NAMES.index("net_dy")


def test_resample_linear_ramp_with_uneven_timestamps():
    t = np.array([0, 10, 50, 100])
    values = t.astype(np.float32) * 2  # value = 2 * time
    out = resample(values, t, steps=5)
    assert np.allclose(out, [0, 50, 100, 150, 200])


def test_resample_keeps_trailing_shape():
    frames, t = make_swipe(frames=23)
    assert resample(frames, t).shape == (RESAMPLED_STEPS, 21, 3)


@pytest.mark.parametrize(
    "values,t",
    [([1.0], [0]), ([1.0, 2.0], [0, 0]), ([1.0, 2.0], [0, 10, 20])],
)
def test_resample_rejects_bad_input(values, t):
    with pytest.raises(ValueError):
        resample(np.array(values), np.array(t))


def test_feature_shape():
    frames, t = make_swipe()
    features = sequence_features(frames, t)
    assert features.shape == (NUM_SEQUENCE_FEATURES,)
    assert np.all(np.isfinite(features))


def test_swipe_direction_is_captured():
    right = sequence_features(*make_swipe(dx=0.3))
    left = sequence_features(*make_swipe(dx=-0.3, start=(0.65, 0.5)))
    up = sequence_features(*make_swipe(dx=0.0, dy=-0.3, start=(0.5, 0.65)))
    assert right[NET_DX] > 2 and left[NET_DX] < -2
    assert up[NET_DY] < -2 and abs(up[NET_DX]) < 1e-3


def test_swipe_counts_as_motion():
    frames, _ = make_swipe(dx=0.3)
    assert has_enough_motion(frames)


def test_resting_hand_at_frame_edge_is_not_motion():
    # The key safety case: a hand resting at the side of the frame, with sensor noise.
    frames, _ = make_swipe(dx=0.0, start=(0.92, 0.5), jitter=0.003)
    assert not has_enough_motion(frames)


def test_wobbling_hand_is_not_motion():
    frames, _ = make_swipe(dx=0.0, start=(0.5, 0.5))
    wobble = 0.05 * np.sin(np.linspace(0, 6 * np.pi, len(frames)))
    frames[:, :, 0] += wobble[:, None].astype(np.float32)
    dx, dy, path = motion_stats(frames)
    assert path > 3  # it travelled a lot...
    assert not has_enough_motion(frames)  # ...but went nowhere


def test_same_swipe_at_different_distance_gives_same_features():
    near = sequence_features(*make_swipe(dx=0.3, size=0.3))
    far = sequence_features(*make_swipe(dx=0.15, size=0.15, start=(0.4, 0.5)))
    assert np.allclose(near, far, atol=1e-4)


def test_same_swipe_at_different_frame_rates_gives_same_features():
    fast_camera = sequence_features(*make_swipe(frames=31, fps=30))
    slow_camera = sequence_features(*make_swipe(frames=16, fps=15))
    assert np.allclose(fast_camera, slow_camera, atol=1e-3)
