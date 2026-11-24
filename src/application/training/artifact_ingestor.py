from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from application.errors import MissingOutputError
from application.ports import ArtifactStore
from domain.models import ModelArtifact, ServingStatus


class ArtifactIngestor:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    def ingest(self, run_id: UUID, output_directory: Path) -> ModelArtifact:
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise MissingOutputError("The training output directory is missing or invalid.")

        output_root = output_directory.resolve(strict=True)
        files = _collect_output_files(output_root)
        model_path = output_root / "model.joblib"
        if model_path not in files:
            raise MissingOutputError("Required training output model.joblib is missing.")

        stored_paths: dict[Path, Path] = {}
        checksums: dict[Path, str] = {}
        stored_names: set[str] = set()
        for path in files:
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise MissingOutputError(f"Training artifact {path.name} is unreadable.") from exc
            if not content and path == model_path:
                raise MissingOutputError("Required training output model.joblib is empty.")
            relative_name = path.relative_to(output_root).as_posix().replace("/", "__")
            if relative_name in stored_names:
                raise MissingOutputError(
                    "Training outputs contain paths that map to the same artifact name."
                )
            stored_names.add(relative_name)
            stored_paths[path] = self._artifact_store.save_artifact(
                run_id,
                relative_name,
                content,
            )
            checksums[path] = hashlib.sha256(content).hexdigest()

        stored_model = stored_paths[model_path]
        artifact = ModelArtifact(
            run_id=run_id,
            artifact_name="model.joblib",
            artifact_location=str(stored_model),
            size_bytes=stored_model.stat().st_size,
            checksum=checksums[model_path],
            created_at=_now_utc(),
            serving_status=ServingStatus.AVAILABLE,
        )
        metadata = {
            "artifact_name": artifact.artifact_name,
            "artifact_location": artifact.artifact_location,
            "checksum": artifact.checksum,
            "created_at": artifact.created_at.isoformat(),
            "run_id": str(artifact.run_id),
            "serving_status": artifact.serving_status.value,
            "size_bytes": artifact.size_bytes,
        }
        self._artifact_store.save_artifact(
            run_id,
            "model-artifact.json",
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
        return artifact


def _collect_output_files(output_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise MissingOutputError(f"Training outputs cannot be enumerated: {error}") from error

    for current, directory_names, file_names in os.walk(
        output_root,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                raise MissingOutputError("Training output contains a symbolic link.")
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                raise MissingOutputError("Training output contains a symbolic link.")
            try:
                if path.is_file():
                    files.append(path)
            except OSError as exc:
                raise MissingOutputError(f"Training output {name} is unreadable.") from exc

    if not files:
        raise MissingOutputError("The training output directory is empty.")
    return tuple(sorted(files, key=lambda path: path.relative_to(output_root).as_posix()))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
