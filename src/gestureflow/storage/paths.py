"""Where GestureFlow AI keeps files on disk.

Everything lives inside the project folder you cloned, so deleting that folder
removes every trace of the app:

    Gestureflow-Ai/
        models/      downloaded MediaPipe model (and, later, models you train)
        user_data/   your recordings, profiles and settings

Both folders are listed in .gitignore and are never committed.
"""

import os
from pathlib import Path

# src/gestureflow/storage/paths.py -> parents[3] is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return _PROJECT_ROOT


def models_dir() -> Path:
    """Folder for model files. Override with the GESTUREFLOW_MODELS_DIR variable."""
    override = os.environ.get("GESTUREFLOW_MODELS_DIR")
    return Path(override) if override else _PROJECT_ROOT / "models"


def user_data_dir() -> Path:
    """Folder for personal data. Override with the GESTUREFLOW_USER_DATA_DIR variable."""
    override = os.environ.get("GESTUREFLOW_USER_DATA_DIR")
    return Path(override) if override else _PROJECT_ROOT / "user_data"
