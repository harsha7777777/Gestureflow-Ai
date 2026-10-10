import json

import numpy as np
import pytest

from gestureflow.storage.samples import (
    Sample,
    SampleError,
    clean_gesture_name,
    count_samples,
    delete_gesture,
    delete_last_sample,
    load_samples,
    save_sample,
)
from tests.conftest import make_swipe


def make_sample(gesture="thumbs_up", kind="static", frames=5):
    seq, t = make_swipe(dx=0.0, frames=frames)
    return Sample(gesture, kind, seq, t + 1_000_000, handedness="Right", aspect=4 / 3)


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("Thumbs Up", "thumbs_up"),
        ("swipe-left", "swipe-left"),
        ("  OK!  ", "ok"),
        ("v2", "v2"),
        ("../etc", "etc"),  # path tricks are stripped, so names can't escape the folder
    ],
)
def test_clean_gesture_name(raw, clean):
    assert clean_gesture_name(raw) == clean


@pytest.mark.parametrize("bad", ["", "   ", "!!!", "a" * 41, "_hidden"])
def test_rejects_bad_names(bad):
    with pytest.raises(SampleError):
        clean_gesture_name(bad)


def test_save_and_load_round_trip(tmp_path):
    original = make_sample()
    path = save_sample(original, tmp_path)
    assert path.parent == tmp_path / "thumbs_up"
    (loaded,), errors = load_samples(root=tmp_path)
    assert errors == []
    assert loaded.gesture == "thumbs_up" and loaded.kind == "static"
    assert loaded.handedness == "Right"
    assert loaded.aspect == pytest.approx(4 / 3, abs=1e-4)
    assert np.allclose(loaded.frames, original.frames, atol=1e-5)
    assert list(loaded.timestamps_ms) == [0, 33, 67, 100, 133]  # relative to the first frame


def test_file_contains_only_landmarks(tmp_path):
    path = save_sample(make_sample(), tmp_path)
    data = json.loads(path.read_text())
    assert set(data) == {"format", "gesture", "kind", "created", "handedness", "aspect", "frames"}
    assert set(data["frames"][0]) == {"t_ms", "points"}


def test_counts_and_deletion(tmp_path):
    for _ in range(3):
        save_sample(make_sample("thumbs_up"), tmp_path)
    save_sample(make_sample("swipe_left", kind="dynamic"), tmp_path)
    assert count_samples(tmp_path) == {"swipe_left": 1, "thumbs_up": 3}
    assert delete_gesture("swipe_left", tmp_path) == 1
    assert count_samples(tmp_path) == {"thumbs_up": 3}


def test_undo_deletes_most_recent(tmp_path):
    first = save_sample(make_sample(), tmp_path)
    second = save_sample(make_sample(), tmp_path)
    assert delete_last_sample("thumbs_up", tmp_path) == second
    assert first.exists() and not second.exists()
    delete_last_sample("thumbs_up", tmp_path)
    assert count_samples(tmp_path) == {}  # empty folder removed too
    assert delete_last_sample("thumbs_up", tmp_path) is None


def test_damaged_file_is_reported_not_fatal(tmp_path):
    save_sample(make_sample(), tmp_path)
    (tmp_path / "thumbs_up" / "zzz_broken.json").write_text("{not json")
    samples, errors = load_samples(root=tmp_path)
    assert len(samples) == 1
    assert len(errors) == 1 and "zzz_broken.json" in errors[0]


def test_rejects_wrong_shapes(tmp_path):
    bad = Sample("x", "static", np.zeros((3, 20, 3)), np.arange(3))
    with pytest.raises(SampleError):
        save_sample(bad, tmp_path)
    with pytest.raises(SampleError):
        save_sample(Sample("x", "wobble", np.zeros((3, 21, 3)), np.arange(3)), tmp_path)


def test_default_location_is_user_data(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREFLOW_USER_DATA_DIR", str(tmp_path))
    path = save_sample(make_sample())
    assert path.is_relative_to(tmp_path / "gestures")


def test_rapid_saves_keep_recording_order(tmp_path):
    # Many saves within the same clock tick must still sort in the order they were made.
    paths = [save_sample(make_sample(), tmp_path) for _ in range(12)]
    assert sorted(paths) == paths
    assert len(set(paths)) == 12


def test_count_and_delete_by_kind(tmp_path):
    from gestureflow.storage.samples import count_by_kind, delete_samples_of_kind
    from tests.conftest import make_pose_sample, make_swipe_sample

    save_sample(make_pose_sample("pause", "open", 1), tmp_path)
    for i in range(3):
        save_sample(make_swipe_sample("pause", 0.3, i), tmp_path)
    assert count_by_kind(tmp_path) == {"pause": {"static": 1, "dynamic": 3}}
    assert delete_samples_of_kind("pause", "static", tmp_path) == 1
    assert count_by_kind(tmp_path) == {"pause": {"dynamic": 3}}
    assert delete_samples_of_kind("pause", "dynamic", tmp_path) == 3
    assert count_by_kind(tmp_path) == {}  # empty folder removed too
    assert delete_samples_of_kind("missing", "static", tmp_path) == 0
