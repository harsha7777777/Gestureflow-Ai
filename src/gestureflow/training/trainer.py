"""Training and honestly evaluating the gesture classifiers.

For each model (poses, movements):

1. Two candidate algorithms are tried, a random forest and an SVM.
2. Each is evaluated with cross-validation over whole *examples*: the examples
   are split into k folds, the model trains on k-1 folds and is tested on the
   examples it has never seen, k times over. Every example is tested exactly
   once. The result is an accuracy and a confusion matrix *per example*, which
   is what the user recorded and can relate to.
3. The candidate with the higher held-out accuracy wins (ties go to the random
   forest, which is faster), and is then trained on all examples.

With 5 examples per gesture these numbers are rough: one mistake is 20% of a
gesture. They are measured, though, never assumed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gestureflow.storage.samples import NEUTRAL, Sample
from gestureflow.training.dataset import Dataset, movement_dataset, pose_dataset
from gestureflow.training.model import GestureModel, TrainedClassifier

RECOMMENDED_EXAMPLES = 5
MAX_FOLDS = 5
RANDOM_SEED = 0

Progress = Callable[[str], None]


def make_candidates() -> dict[str, object]:
    """Fresh, unfitted candidate classifiers. Both output probabilities."""
    return {
        "Random forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=RANDOM_SEED
        ),
        # SVMs need features on similar scales, hence the scaler. An SVM gives
        # scores, not probabilities; calibration turns them into probabilities.
        "SVM": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(C=10, gamma="scale", class_weight="balanced"), cv=3, ensemble=False
            ),
        ),
    }


@dataclass
class Evaluation:
    """How one model did on examples it had not seen during training."""

    kind: str  # "pose" or "movement"
    algorithm: str  # the one that was chosen
    classes: list[str]
    examples: dict[str, int]  # examples per gesture
    folds: int  # 0 = not enough examples to evaluate
    accuracy: float | None = None  # fraction of held-out examples classified correctly
    confusion: np.ndarray | None = None  # rows = true gesture, columns = predicted
    candidates: dict[str, float] = field(default_factory=dict)  # accuracy of each algorithm

    @property
    def evaluated(self) -> bool:
        return self.accuracy is not None


@dataclass
class TrainingReport:
    pose: Evaluation | None
    movement: Evaluation | None
    warnings: list[str]

    def summary_lines(self) -> list[str]:
        lines = []
        for name, ev in (("Poses", self.pose), ("Movements", self.movement)):
            if ev is None:
                continue
            if ev.evaluated:
                tested = sum(ev.examples.values())
                correct = int(round(ev.accuracy * tested))
                lines.append(
                    f"{name}: {correct} of {tested} held-out examples correct "
                    f"({ev.accuracy:.0%}, {ev.algorithm}, {ev.folds}-fold cross-validation)"
                )
            else:
                lines.append(f"{name}: trained ({ev.algorithm}), not enough examples to test")
        return lines


def split_folds(groups_by_class: dict[str, list[int]], k: int) -> list[set[int]]:
    """Assign examples to k folds so that every fold holds some of every gesture."""
    rng = np.random.default_rng(RANDOM_SEED)
    folds: list[set[int]] = [set() for _ in range(k)]
    for label in sorted(groups_by_class):
        groups = list(groups_by_class[label])
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            folds[i % k].add(group)
    return folds


def _predict_examples(estimator, data: Dataset, test_groups: set[int]) -> dict[int, str]:
    """Predicted gesture per held-out example (pose frames are averaged)."""
    mask = data.original & np.isin(data.groups, list(test_groups))
    proba = estimator.predict_proba(data.X[mask])
    classes = np.array([str(c) for c in estimator.classes_])
    out = {}
    for group in test_groups:
        rows = data.groups[mask] == group
        if rows.any():
            out[group] = str(classes[int(np.argmax(proba[rows].mean(axis=0)))])
    return out


def cross_validate(make_estimator: Callable[[], object], data: Dataset, k: int):
    """(accuracy, confusion matrix, class list) over held-out examples."""
    labels = {int(g): str(y) for g, y in zip(data.groups, data.y, strict=True)}
    by_class: dict[str, list[int]] = {}
    for group, label in labels.items():
        by_class.setdefault(label, []).append(group)
    classes = sorted(by_class)
    index = {c: i for i, c in enumerate(classes)}

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for test_groups in split_folds(by_class, k):
        train = ~np.isin(data.groups, list(test_groups))
        estimator = make_estimator()
        estimator.fit(data.X[train], data.y[train].astype(str))
        for group, predicted in _predict_examples(estimator, data, test_groups).items():
            confusion[index[labels[group]], index[predicted]] += 1
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return accuracy, confusion, classes


def train_one(kind: str, data: Dataset, progress: Progress = print):
    """Evaluate both candidates, then fit the best one on everything.

    Returns (TrainedClassifier, Evaluation), or (None, None) with fewer than two gestures.
    """
    examples = data.examples_per_class()
    if len(examples) < 2:
        return None, None
    k = min(MAX_FOLDS, min(examples.values()))

    scores: dict[str, float] = {}
    best = ("Random forest", None, None)
    if k >= 2:
        for name in make_candidates():
            progress(f"Testing {name} on {kind} examples...")
            accuracy, confusion, classes = cross_validate(
                lambda name=name: make_candidates()[name], data, k
            )
            scores[name] = accuracy
            if best[1] is None or accuracy > scores[best[0]]:
                best = (name, confusion, classes)

    algorithm, confusion, _ = best
    progress(f"Training the {kind} model on all examples...")
    estimator = make_candidates()[algorithm]
    estimator.fit(data.X, data.y.astype(str))
    evaluation = Evaluation(
        kind=kind,
        algorithm=algorithm,
        classes=sorted(examples),
        examples=examples,
        folds=k if k >= 2 else 0,
        accuracy=scores.get(algorithm),
        confusion=confusion,
        candidates=scores,
    )
    return TrainedClassifier(kind, algorithm, estimator), evaluation


def check_examples(samples: list[Sample]) -> list[str]:
    """Plain-language warnings about the recorded examples."""
    warnings = []
    kinds: dict[str, set[str]] = {}
    counts: dict[tuple[str, str], int] = {}
    for s in samples:
        kinds.setdefault(s.gesture, set()).add(s.kind)
        counts[(s.gesture, s.kind)] = counts.get((s.gesture, s.kind), 0) + 1
    for (gesture, kind), n in sorted(counts.items()):
        if gesture != NEUTRAL and n < RECOMMENDED_EXAMPLES:
            what = "pose" if kind == "static" else "movement"
            warnings.append(
                f"'{gesture}' has only {n} {what} example{'s' if n != 1 else ''}; "
                f"{RECOMMENDED_EXAMPLES}-15 work best."
            )
    for gesture, k in sorted(kinds.items()):
        if gesture != NEUTRAL and len(k) > 1:
            warnings.append(
                f"'{gesture}' was recorded both as a pose and as a movement. "
                "They are learned as two separate gestures with the same name."
            )
    has_static = any(s.kind == "static" and s.gesture != NEUTRAL for s in samples)
    has_dynamic = any(s.kind == "dynamic" and s.gesture != NEUTRAL for s in samples)
    if NEUTRAL not in kinds and (has_static or has_dynamic):
        warnings.append(
            "No 'neutral' examples. Record some (hand resting, typing, moving casually) "
            "so the app learns when to do nothing."
        )
    elif has_dynamic and not any(s.kind == "dynamic" and s.gesture == NEUTRAL for s in samples):
        warnings.append(
            "No 'neutral' movement examples. Record a few as Movement (scratching your "
            "head, reaching for the mouse) so ordinary motion isn't mistaken for a swipe."
        )
    return warnings


def train(samples: list[Sample], progress: Progress = print) -> tuple[GestureModel, TrainingReport]:
    """Train both models from recorded examples. Raises ValueError if nothing can be trained."""
    warnings = check_examples(samples)
    pose, pose_eval = train_one("pose", pose_dataset(samples), progress)
    movement, movement_eval = train_one("movement", movement_dataset(samples), progress)

    if pose is None and movement is None:
        raise ValueError(
            "Not enough to train yet. Record at least two different gestures of the same "
            "kind (for example one gesture and 'neutral'), with a few examples each."
        )
    if pose is None and any(s.kind == "static" and s.gesture != NEUTRAL for s in samples):
        warnings.append("Poses were skipped: they need at least one other pose or 'neutral'.")
    if movement is None and any(s.kind == "dynamic" and s.gesture != NEUTRAL for s in samples):
        warnings.append(
            "Movements were skipped: they need at least one other movement or neutral movements."
        )
    model = GestureModel(pose=pose, movement=movement)
    return model, TrainingReport(pose_eval, movement_eval, warnings)
