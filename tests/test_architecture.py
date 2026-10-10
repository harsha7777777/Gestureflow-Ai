"""Enforces the architecture rules from docs/architecture.md automatically.

The rules are easy to break by accident with one convenient import, so a test
checks them on every push instead of relying on code review alone.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "gestureflow"

# stage -> stages it must never import
FORBIDDEN_IMPORTS = {
    # The ML side knows nothing about OS actions or how gestures are mapped.
    "detection": {"actions", "mapping", "ui"},
    "features": {"actions", "mapping", "ui"},
    "training": {"actions", "mapping", "ui"},
    "recognition": {"actions", "mapping", "ui"},
    # The action layer receives command names and never touches the ML model.
    "actions": {"camera", "detection", "features", "training", "recognition", "ui"},
    # Mapping translates names to names; it needs no model and no OS access.
    "mapping": {"camera", "detection", "features", "training", "actions", "ui"},
}

EXPECTED_STAGES = {
    "camera",
    "detection",
    "features",
    "training",
    "recognition",
    "mapping",
    "actions",
    "storage",
    "ui",
}


def imported_stages(path: Path) -> set[str]:
    """Return the gestureflow stages a source file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:  # relative import, e.g. `from ..actions import x`
                modules.append("gestureflow." + node.module)
            else:
                modules.append(node.module)
    return {m.split(".")[1] for m in modules if m.startswith("gestureflow.")}


def test_all_stages_exist():
    found = {p.name for p in PACKAGE_DIR.iterdir() if (p / "__init__.py").exists()}
    assert found >= EXPECTED_STAGES


@pytest.mark.parametrize("stage", sorted(FORBIDDEN_IMPORTS))
def test_stage_respects_boundaries(stage):
    violations = []
    for path in (PACKAGE_DIR / stage).rglob("*.py"):
        bad = imported_stages(path) & FORBIDDEN_IMPORTS[stage]
        if bad:
            violations.append(f"{path.relative_to(PACKAGE_DIR)} imports {sorted(bad)}")
    assert not violations, "Architecture rule broken:\n" + "\n".join(violations)


def test_boundary_checker_detects_violations(tmp_path):
    # Guards the guard: make sure the checker really catches forbidden imports.
    sample = tmp_path / "bad.py"
    sample.write_text("from gestureflow.actions import windows\nimport gestureflow.ui\n")
    assert imported_stages(sample) == {"actions", "ui"}
