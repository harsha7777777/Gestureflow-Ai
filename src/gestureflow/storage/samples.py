"""Saving and loading recorded gesture examples.

Each example is one small JSON file containing ONLY hand landmark coordinates
and timestamps, never images or video:

    user_data/gestures/<gesture name>/<number>_<date>.json

    {
      "format": 1,
      "gesture": "thumbs_up",
      "kind": "static",              # or "dynamic" (a movement such as a swipe)
      "created": "2026-10-10T18:40:00+00:00",
      "handedness": "Right",
      "aspect": 1.3333,              # image width / height (see HandLandmarks)
      "frames": [{"t_ms": 0, "points": [[x, y, z], ... 21 points]}, ...]
    }

One file per example keeps things simple: you can open, copy or delete any
example by hand, and a damaged file only affects that one example.
"""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gestureflow.detection.landmarks import NUM_LANDMARKS
from gestureflow.storage.paths import user_data_dir

FORMAT_VERSION = 1
KINDS = ("static", "dynamic")

# Every user should record this class: everyday hand movements that must NOT
# trigger anything. It is what lets the app "do nothing" confidently.
NEUTRAL = "neutral"

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class SampleError(ValueError):
    """An example could not be saved or loaded."""


@dataclass(frozen=True)
class Sample:
    gesture: str
    kind: str
    frames: np.ndarray  # (T, 21, 3)
    timestamps_ms: np.ndarray  # (T,)
    handedness: str = "Unknown"
    aspect: float = 1.0
    created: str = ""
    path: Path | None = None


def gestures_dir() -> Path:
    return user_data_dir() / "gestures"


def clean_gesture_name(name: str) -> str:
    """Turn user input like "Thumbs Up" into a safe name like "thumbs_up".

    Raises SampleError if nothing usable is left. Names become folder names, so
    only lowercase letters, digits, "_" and "-" are allowed (max 40 characters).
    """
    cleaned = re.sub(r"[\s]+", "_", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)
    if not _NAME_PATTERN.match(cleaned):
        raise SampleError(
            f"'{name}' is not a valid gesture name. Use letters, numbers, spaces, "
            "'_' or '-' (up to 40 characters)."
        )
    return cleaned


def save_sample(sample: Sample, root: Path | None = None) -> Path:
    """Write one example to disk and return its path."""
    gesture = clean_gesture_name(sample.gesture)
    if sample.kind not in KINDS:
        raise SampleError(f"kind must be one of {KINDS}, not {sample.kind!r}")
    frames = np.asarray(sample.frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (NUM_LANDMARKS, 3) or len(frames) == 0:
        raise SampleError(f"frames must have shape (T, 21, 3), got {frames.shape}")
    if len(sample.timestamps_ms) != len(frames):
        raise SampleError("need exactly one timestamp per frame")

    now = datetime.now(UTC)
    t0 = int(sample.timestamps_ms[0])
    data = {
        "format": FORMAT_VERSION,
        "gesture": gesture,
        "kind": sample.kind,
        "created": now.isoformat(timespec="seconds"),
        "handedness": sample.handedness,
        "aspect": round(float(sample.aspect), 4),
        "frames": [
            # Timestamps are stored relative to the first frame; 5 decimals is far
            # more precise than the detector itself.
            {"t_ms": int(t) - t0, "points": np.round(f, 5).tolist()}
            for t, f in zip(sample.timestamps_ms, frames, strict=True)
        ],
    }
    folder = (root or gestures_dir()) / gesture
    folder.mkdir(parents=True, exist_ok=True)
    # A running number at the front keeps files in recording order (used by "undo").
    # Timestamps alone can't: on Windows the clock may return the same value for
    # two examples saved a few milliseconds apart.
    path = folder / f"{_next_index(folder):05d}_{now:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return path


def _next_index(folder: Path) -> int:
    numbers = [int(p.name.split("_", 1)[0]) for p in folder.glob("*.json") if p.name[:5].isdigit()]
    return max(numbers, default=0) + 1


def load_sample(path: Path) -> Sample:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION:
            raise SampleError(f"unsupported format {data.get('format')!r}")
        frames = np.array([f["points"] for f in data["frames"]], dtype=np.float32)
        timestamps = np.array([f["t_ms"] for f in data["frames"]], dtype=np.int64)
        if frames.ndim != 3 or frames.shape[1:] != (NUM_LANDMARKS, 3):
            raise SampleError(f"bad landmark shape {frames.shape}")
        if data["kind"] not in KINDS:
            raise SampleError(f"bad kind {data['kind']!r}")
        return Sample(
            gesture=data["gesture"],
            kind=data["kind"],
            frames=frames,
            timestamps_ms=timestamps,
            handedness=data.get("handedness", "Unknown"),
            aspect=float(data.get("aspect", 1.0)),
            created=data.get("created", ""),
            path=Path(path),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, SampleError):
            raise SampleError(f"{path}: {exc}") from None
        raise SampleError(f"{path}: could not read example ({exc})") from exc


def load_samples(gesture: str | None = None, root: Path | None = None):
    """Load examples (all gestures, or one). Returns (samples, errors).

    Damaged files are reported in `errors` instead of stopping everything.
    """
    base = root or gestures_dir()
    folders = [base / clean_gesture_name(gesture)] if gesture else sorted(base.glob("*"))
    samples, errors = [], []
    for folder in folders:
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                samples.append(load_sample(path))
            except SampleError as exc:
                errors.append(str(exc))
    return samples, errors


def count_samples(root: Path | None = None) -> dict[str, int]:
    """Number of saved examples per gesture name, e.g. {"neutral": 12, "thumbs_up": 8}."""
    base = root or gestures_dir()
    if not base.is_dir():
        return {}
    return {
        folder.name: len(list(folder.glob("*.json")))
        for folder in sorted(base.iterdir())
        if folder.is_dir()
    }


def _kind_of(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError):
        return None


def count_by_kind(root: Path | None = None) -> dict[str, dict[str, int]]:
    """Examples per gesture and kind, e.g. {"pause": {"static": 1, "dynamic": 4}}."""
    base = root or gestures_dir()
    if not base.is_dir():
        return {}
    counts: dict[str, dict[str, int]] = {}
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            kind = _kind_of(path)
            if kind in KINDS:
                counts.setdefault(folder.name, {}).setdefault(kind, 0)
                counts[folder.name][kind] += 1
    return counts


def delete_samples_of_kind(gesture: str, kind: str, root: Path | None = None) -> int:
    """Delete the examples of one gesture that have the given kind. Returns how many."""
    if kind not in KINDS:
        raise SampleError(f"kind must be one of {KINDS}, not {kind!r}")
    folder = (root or gestures_dir()) / clean_gesture_name(gesture)
    deleted = 0
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if _kind_of(path) == kind:
            path.unlink()
            deleted += 1
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return deleted


def delete_gesture(gesture: str, root: Path | None = None) -> int:
    """Delete all examples of one gesture. Returns how many were deleted."""
    folder = (root or gestures_dir()) / clean_gesture_name(gesture)
    if not folder.is_dir():
        return 0
    count = len(list(folder.glob("*.json")))
    shutil.rmtree(folder)
    return count


def delete_last_sample(gesture: str, root: Path | None = None) -> Path | None:
    """Delete the most recent example of a gesture (an "undo"). Returns its path."""
    folder = (root or gestures_dir()) / clean_gesture_name(gesture)
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not files:
        return None
    files[-1].unlink()
    if not any(folder.iterdir()):
        folder.rmdir()
    return files[-1]
