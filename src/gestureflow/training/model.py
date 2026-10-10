"""The trained gesture model: what training produces and live recognition uses.

It contains up to two scikit-learn classifiers (poses and movements) and a few
facts about how they were trained. It outputs gesture *names* with a confidence
and knows nothing about what a gesture is mapped to.

Saved as `user_data/trained/gesture_model.joblib`. joblib (installed with
scikit-learn) uses Python's pickle format, which can run code when loaded, so
only ever load model files you trained yourself. This file is never shared or
imported from elsewhere; profiles (M5) will exchange JSON only.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from gestureflow.features.dynamic import NUM_SEQUENCE_FEATURES
from gestureflow.features.static import NUM_STATIC_FEATURES
from gestureflow.storage.paths import user_data_dir

MODEL_FORMAT = 1
MODEL_FILENAME = "gesture_model.joblib"


class ModelError(Exception):
    """The trained model is missing, damaged or from an incompatible version."""


@dataclass
class TrainedClassifier:
    kind: str  # "pose" or "movement"
    algorithm: str  # e.g. "Random forest"
    estimator: object  # a fitted scikit-learn classifier with predict_proba

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.estimator.classes_)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """(most likely gesture, its probability between 0 and 1) for one feature vector."""
        proba = self.predict_proba(np.asarray(features, dtype=np.float32)[None, :])[0]
        best = int(np.argmax(proba))
        return self.classes[best], float(proba[best])

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(rows)


@dataclass
class GestureModel:
    pose: TrainedClassifier | None = None
    movement: TrainedClassifier | None = None
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    # Feature sizes the model was trained with; a mismatch means the features
    # changed in a newer version of the app and the model must be retrained.
    num_static_features: int = NUM_STATIC_FEATURES
    num_sequence_features: int = NUM_SEQUENCE_FEATURES
    format: int = MODEL_FORMAT

    @property
    def gestures(self) -> list[str]:
        names = set()
        for clf in (self.pose, self.movement):
            if clf is not None:
                names.update(clf.classes)
        return sorted(names)


def trained_dir() -> Path:
    return user_data_dir() / "trained"


def model_file(root: Path | None = None) -> Path:
    return (root or trained_dir()) / MODEL_FILENAME


def save_model(model: GestureModel, root: Path | None = None) -> Path:
    path = model_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    tmp.replace(path)  # never leave a half-written model behind
    return path


def load_model(root: Path | None = None) -> GestureModel:
    """Load the trained model. Raises ModelError with a user-facing message."""
    path = model_file(root)
    if not path.exists():
        raise ModelError("No trained model yet. Record examples, then click Train.")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelError(
            f"The trained model could not be read ({exc}). Please train again."
        ) from exc
    if not isinstance(model, GestureModel) or model.format != MODEL_FORMAT:
        raise ModelError(
            "The trained model is from another version of the app. Please train again."
        )
    if (
        model.num_static_features != NUM_STATIC_FEATURES
        or model.num_sequence_features != NUM_SEQUENCE_FEATURES
    ):
        raise ModelError(
            "The app's features changed since this model was trained. Please train again."
        )
    return model


def delete_model(root: Path | None = None) -> bool:
    path = model_file(root)
    if path.exists():
        path.unlink()
        return True
    return False
