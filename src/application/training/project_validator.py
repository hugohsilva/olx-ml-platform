from __future__ import annotations

import platform
from collections.abc import Mapping
from pathlib import Path

from application.errors import ValidationError
from application.training.input_hashes import hash_config, hash_dataset_files, hash_project_files
from domain.models import TrainingProject

CONTRACT_VERSION = "1"


def validate_training_project(
    project_directory: Path,
    *,
    effective_config: Mapping[str, object],
) -> TrainingProject:
    if not isinstance(project_directory, Path) or project_directory.is_symlink():
        raise ValidationError("The project directory is missing or is a symbolic link.")
    try:
        project_root = project_directory.resolve(strict=True)
    except OSError as exc:
        raise ValidationError("The project directory is missing or unreadable.") from exc
    if not project_root.is_dir():
        raise ValidationError("The project directory is missing or is not a directory.")

    entrypoint = project_root / "train.py"
    dependencies = project_root / "requirements.txt"
    data_directory = project_root / "data"
    _require_readable_file(entrypoint, "train.py", nonempty=True)
    _require_readable_file(dependencies, "requirements.txt", nonempty=False)
    if data_directory.is_symlink() or not data_directory.is_dir():
        raise ValidationError("The required data directory is missing or invalid.")

    code_hash = hash_project_files(project_root, data_directory)
    dataset_hash = hash_dataset_files(data_directory)
    config_hash_value = hash_config(effective_config)

    return TrainingProject(
        project_id=project_root.name,
        entrypoint="train.py",
        dependencies="requirements.txt",
        data_location=data_directory.resolve(strict=True),
        contract_version=CONTRACT_VERSION,
        code_hash=code_hash,
        dataset_identity="data",
        dataset_hash=dataset_hash,
        config_hash=config_hash_value,
        python_version=platform.python_version(),
    )


def _require_readable_file(path: Path, name: str, *, nonempty: bool) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"The required {name} file is missing or invalid.")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"The required {name} file is unreadable.") from exc
    if nonempty and not content.strip():
        raise ValidationError(f"The required {name} file is empty.")
    if b"\x00" in content:
        raise ValidationError(f"The required {name} file contains invalid binary content.")
