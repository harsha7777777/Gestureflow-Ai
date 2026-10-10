"""Entry point for ``python -m gestureflow``.

python -m gestureflow                    open the live camera window
python -m gestureflow --camera 1         start with a specific camera
python -m gestureflow --download-model   download the hand model and exit
python -m gestureflow --check            check the installation and exit
python -m gestureflow --train            train on your recorded examples and exit
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
    parser.add_argument(
        "--train", action="store_true", help="train on your recorded examples, print results, exit"
    )
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
    from gestureflow.detection.model_file import (
        MODEL_SHA256,
        ModelDownloadError,
        download_model,
        is_model_present,
        model_path,
        sha256_of,
    )

    if is_model_present() and sha256_of(model_path()) == MODEL_SHA256:
        print(f"Hand model already downloaded and verified: {model_path()}")
        return 0
    print("Downloading the MediaPipe hand model (about 8 MB)...")
    try:
        path = download_model()
    except ModelDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved and verified: {path}")
    return 0


def train_from_terminal() -> int:
    from gestureflow.storage.samples import load_samples
    from gestureflow.training.model import save_model
    from gestureflow.training.trainer import train

    samples, errors = load_samples()
    for error in errors:
        print(f"Skipped a damaged file: {error}", file=sys.stderr)
    print(f"Loaded {len(samples)} examples.")
    try:
        model, report = train(samples)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    path = save_model(model)
    print()
    for line in report.summary_lines():
        print(line)
    for ev in (report.pose, report.movement):
        if ev is not None and ev.confusion is not None:
            print(
                f"\n{ev.kind.capitalize()} confusion matrix (rows = recorded, columns = guessed):"
            )
            width = max(len(c) for c in ev.classes) + 2
            print(" " * width + "".join(c[:8].rjust(9) for c in ev.classes))
            for name, row in zip(ev.classes, ev.confusion, strict=True):
                print(name.ljust(width) + "".join(str(n).rjust(9) for n in row))
            others = {k: f"{v:.0%}" for k, v in ev.candidates.items()}
            print(f"Held-out accuracy per algorithm: {others}")
    for warning in report.warnings:
        print(f"Note: {warning}")
    print(f"\nSaved the model to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return check_installation()
    if args.download_model:
        return download()
    if args.train:
        return train_from_terminal()

    # Imported here so --check and --download-model work even if Qt can't start.
    from gestureflow.ui.app import run_app

    return run_app(camera_index=args.camera)


if __name__ == "__main__":
    sys.exit(main())
