"""Entry point for ``python -m gestureflow``.

Milestone 0 only verifies that the package is installed correctly.
The live webcam window arrives in Milestone 1.
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
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)

    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        found = platform.python_version()
        print(
            f"Warning: GestureFlow AI is tested on Python 3.12 and 3.13, you are running {found}.",
            file=sys.stderr,
        )

    print(f"GestureFlow AI {__version__} is installed correctly.")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print("The webcam window is coming in Milestone 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
