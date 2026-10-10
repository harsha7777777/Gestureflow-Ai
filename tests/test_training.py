import numpy as np
import pytest

from gestureflow.features.dynamic import NUM_SEQUENCE_FEATURES
from gestureflow.features.static import NUM_STATIC_FEATURES, static_features
from gestureflow.training.dataset import CROPS, movement_dataset, pose_dataset
from gestureflow.training.model import ModelError, load_model, model_file, save_model
from gestureflow.training.trainer import check_examples, split_folds, train
from tests.conftest import make_pose_sample, make_swipe_sample, make_training_set


@pytest.fixture(scope="module")
def trained():
    return train(make_training_set(), progress=lambda message: None)


def test_pose_dataset_has_one_row_per_frame_and_skips_movements():
    samples = [
        make_pose_sample("palm", "open", 1),
        make_swipe_sample("swipe_right", 0.3, 2),
        make_pose_sample("neutral", "relaxed", 3),
    ]
    data = pose_dataset(samples)
    assert data.X.shape == (60, NUM_STATIC_FEATURES)
    assert set(data.y) == {"palm", "neutral"}
    assert data.examples_per_class() == {"neutral": 1, "palm": 1}


def test_movement_dataset_adds_crops_that_keep_the_motion():
    data = movement_dataset([make_swipe_sample("swipe_right", 0.3, 1)])
    assert data.X.shape[1] == NUM_SEQUENCE_FEATURES
    assert data.original.sum() == 1  # the full recording
    assert 1 < len(data) <= 1 + len(CROPS)
    assert set(data.groups) == {0}  # all rows belong to the same example


def test_folds_hold_every_gesture_and_every_example_once():
    folds = split_folds({"a": [0, 1, 2, 3, 4], "b": [5, 6, 7]}, k=3)
    assert sorted(g for fold in folds for g in fold) == list(range(8))
    for fold in folds:
        assert fold & {0, 1, 2, 3, 4} and fold & {5, 6, 7}


def test_training_separates_synthetic_gestures(trained):
    model, report = trained
    assert model.pose.classes == ("fist", "neutral", "palm")
    assert model.movement.classes == ("neutral", "swipe_left", "swipe_right")
    # These synthetic gestures are easy; this checks the pipeline, not real-world accuracy.
    assert report.pose.accuracy == 1.0
    assert report.movement.accuracy == 1.0
    assert report.pose.folds == 5
    # Every example tested exactly once; neutral movements also teach the pose model.
    assert report.pose.examples == {"fist": 5, "neutral": 10, "palm": 5}
    assert report.pose.confusion.sum() == 20
    assert set(report.pose.candidates) == {"Random forest", "SVM"}


def test_trained_model_predicts_new_examples(trained):
    model, _ = trained
    fist = make_pose_sample("x", "fist", seed=999)
    label, confidence = model.pose.predict(static_features(fist.frames[0]))
    assert label == "fist" and 0.5 < confidence <= 1.0


def test_summary_mentions_held_out_counts(trained):
    _, report = trained
    lines = report.summary_lines()
    assert lines[0].startswith("Poses: 20 of 20 held-out examples correct (100%")
    assert lines[1].startswith("Movements: 15 of 15")


def test_confusion_matrix_shows_a_real_mix_up():
    # "palm" examples that are really fists can't be told apart from "fist".
    samples = [make_pose_sample("fist", "fist", i) for i in range(4)]
    samples += [make_pose_sample("palm", "fist", 50 + i) for i in range(4)]
    _, report = train(samples, progress=lambda m: None)
    assert report.pose.accuracy < 0.9
    assert report.pose.confusion.sum() == 8


def test_one_gesture_alone_cannot_be_trained():
    with pytest.raises(ValueError, match="Not enough to train"):
        train([make_pose_sample("palm", "open", i) for i in range(5)], progress=lambda m: None)


def test_two_examples_train_but_one_example_is_not_evaluated():
    samples = [make_pose_sample("palm", "open", 1), make_pose_sample("fist", "fist", 2)]
    model, report = train(samples, progress=lambda m: None)
    assert model.pose is not None and not report.pose.evaluated
    assert "not enough examples to test" in report.summary_lines()[0]


def test_warnings_are_plain_language():
    samples = [make_pose_sample("palm", "open", i) for i in range(3)]
    samples += [make_swipe_sample("palm", 0.3, 9)]
    warnings = " ".join(check_examples(samples))
    assert "only 3 pose examples" in warnings
    assert "both as a pose and as a movement" in warnings
    assert "No 'neutral' examples" in warnings


def test_model_round_trip(tmp_path, trained):
    model, _ = trained
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    x = np.zeros(NUM_STATIC_FEATURES, dtype=np.float32)
    assert loaded.pose.predict(x) == model.pose.predict(x)
    assert loaded.gestures == ["fist", "neutral", "palm", "swipe_left", "swipe_right"]


def test_missing_or_outdated_model_explains_what_to_do(tmp_path, trained):
    with pytest.raises(ModelError, match="No trained model yet"):
        load_model(tmp_path)
    model, _ = trained
    model.num_static_features = 5  # as if the features changed in a newer version
    try:
        save_model(model, tmp_path)
        with pytest.raises(ModelError, match="train again"):
            load_model(tmp_path)
    finally:
        model.num_static_features = NUM_STATIC_FEATURES
    model_file(tmp_path).write_bytes(b"garbage")
    with pytest.raises(ModelError, match="could not be read"):
        load_model(tmp_path)
