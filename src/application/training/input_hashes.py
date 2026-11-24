from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from application.errors import ValidationError

_EXCLUDED_PROJECT_DIRECTORIES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "output",
        "outputs",
        "artifacts",
    }
)


def hash_project_files(project_directory: Path, data_directory: Path) -> str:
    project_root = _resolve_directory(project_directory, "project code")
    data_root = _resolve_directory(data_directory, "dataset")
    if not data_root.is_relative_to(project_root):
        raise ValidationError("The dataset directory must be inside the training project.")

    files = _regular_files(
        project_root,
        excluded_root=data_root,
        excluded_directories=_EXCLUDED_PROJECT_DIRECTORIES,
    )
    return _hash_file_manifest(files, "project code")


def hash_dataset_files(data_directory: Path) -> str:
    data_root = _resolve_directory(data_directory, "dataset")
    files = _regular_files(data_root)
    return _hash_file_manifest(files, "dataset")


def hash_config(config: Any) -> str:
    try:
        canonical_json = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(
            "The effective configuration cannot be serialized as canonical JSON."
        ) from exc
    return hashlib.sha256(canonical_json).hexdigest()


def _resolve_directory(path: Path, input_name: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise ValidationError(f"The {input_name} directory is missing or is a symbolic link.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"The {input_name} directory is missing or unreadable.") from exc
    if not resolved.is_dir():
        raise ValidationError(f"The {input_name} path is not a directory.")
    return resolved


def _regular_files(
    root: Path,
    *,
    excluded_root: Path | None = None,
    excluded_directories: frozenset[str] = frozenset(),
) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []

    def raise_walk_error(error: OSError) -> None:
        raise ValidationError("A required input directory is unreadable.") from error

    try:
        for current_directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=raise_walk_error,
        ):
            current_path = Path(current_directory)
            if excluded_root is not None and current_path == excluded_root:
                directory_names.clear()
                continue

            kept_directories: list[str] = []
            for name in directory_names:
                child = current_path / name
                if child.is_symlink():
                    if name in excluded_directories:
                        continue
                    raise ValidationError("A project input contains a symbolic link.")
                if name in excluded_directories:
                    continue
                if excluded_root is not None and child == excluded_root:
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories

            for name in file_names:
                file_path = current_path / name
                if file_path.is_symlink():
                    raise ValidationError("A project input contains a symbolic link.")
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_bytes()
                except OSError as exc:
                    raise ValidationError("A required input file is unreadable.") from exc
                files.append((file_path.relative_to(root).as_posix(), content))
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("A required input file is unreadable.") from exc

    return sorted(files, key=lambda item: item[0])


def _hash_file_manifest(files: list[tuple[str, bytes]], input_name: str) -> str:
    if not files:
        raise ValidationError(f"The {input_name} input set is empty.")
    manifest = [
        {"path": relative_path, "sha256": hashlib.sha256(content).hexdigest()}
        for relative_path, content in files
    ]
    canonical_json = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()
