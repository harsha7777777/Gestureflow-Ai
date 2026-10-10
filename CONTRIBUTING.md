# Contributing to GestureFlow AI

Thanks for your interest! Contributions of all sizes are welcome.

## Set up a development environment

Follow the install steps in the [README](README.md), using `pip install -e ".[dev]"`.

## Before you open a pull request

```bash
ruff format .
ruff check .
pytest
```

CI runs the same checks on macOS, Windows and Linux.

## Ground rules

- **Respect the architecture.** The ML stages (`detection`, `features`, `training`,
  `recognition`) must never import `actions` or `mapping`, and `actions` must never import the ML
  stages. `tests/test_architecture.py` enforces this. See [docs/architecture.md](docs/architecture.md).
- **Tests without a camera.** New logic should be testable with synthetic or recorded landmark
  data. Tests that truly need a webcam get `@pytest.mark.camera`.
- **Privacy first.** Never save frames or video by default, and never send data off the machine.
- **No made-up numbers.** Only report accuracy or performance figures you actually measured.
- **New dependencies** need a short explanation in the pull request, and an exact pin in
  `requirements.txt`.

## Reporting bugs and ideas

Use the issue templates. For bugs, include your OS, Python version and the full error message.
