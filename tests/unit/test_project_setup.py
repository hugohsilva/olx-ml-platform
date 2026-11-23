from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import olx_ml_platform


def test_project_package_imports_and_registers_mlplatform() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert callable(olx_ml_platform.main)
    assert scripts["mlplatform"] == "olx_ml_platform:main"


def test_setup_phase_project_structure_exists() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    expected_paths = (
        "src/domain/__init__.py",
        "src/application/__init__.py",
        "src/infrastructure/config.py",
        "src/cli/__init__.py",
        "src/api/__init__.py",
        "examples/iris/train.py",
        "examples/iris/requirements.txt",
        "examples/iris/data/.gitkeep",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "tests/contract",
    )

    assert all((repository_root / path).exists() for path in expected_paths)
