"""Entry point for ``python -m gestureflow``.

python -m gestureflow                    open the live camera window
python -m gestureflow --camera 1         start with a specific camera
python -m gestureflow --download-model   download the hand model and exit
python -m gestureflow --check            check the installation and exit
"""

import argparse
import platform
import sys

from gestureflow import __version__

SUPPORTED_PYTHON = ((3, 12), (3, 13))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gestureflow",
        description="Control your computer with your own hand gestures.",
    )
    parser.add_argument("--version", action="version", version=f"GestureFlow AI {__version__}")
    parser.add_argument(
        "--camera", type=int, metavar="N", help="camera number to start with (default: first found)"
    )
    parser.add_argument(
        "--download-model", action="store_true", help="download the hand model and exit"
    )
    parser.add_argument("--check", action="store_true", help="check the installation and exit")
    return parser


def check_installation() -> int:
    from gestureflow.detection.model_file import is_model_present, model_path

    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        print(
            "Warning: GestureFlow AI is tested on Python 3.12 and 3.13, "
            f"you are running {platform.python_version()}.",
            file=sys.stderr,
        )
    print(f"GestureFlow AI {__version__} is installed correctly.")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if is_model_present():
        print(f"Hand model: found at {model_path()}")
    else:
        print("Hand model: not downloaded yet (run: python -m gestureflow --download-model)")
    return 0


def download() -> int:
    from gestureflow.detection.model_file import ModelDownloadError, download_model

    print("Downloading the MediaPipe hand model (about 8 MB)...")
    try:
        path = download_model()
    except ModelDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved and verified: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return check_installation()
    if args.download_model:
        return download()

    # Imported here so --check and --download-model work even if Qt can't start.
    from gestureflow.ui.app import run_app

    return run_app(camera_index=args.camera)


if __name__ == "__main__":
    sys.exit(main())
