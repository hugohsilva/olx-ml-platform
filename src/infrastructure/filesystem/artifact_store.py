from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from domain.models import ModelArtifact, ServingStatus


class FilesystemArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    def run_directory(self, run_id: UUID) -> Path:
        _validate_run_id(run_id)
        return self._root / "executions" / run_id.hex

    def artifact_directory(self, run_id: UUID) -> Path:
        return self.run_directory(run_id) / "artifacts"

    def log_directory(self, run_id: UUID) -> Path:
        return self.run_directory(run_id) / "logs"

    def ensure_run_directories(self, run_id: UUID) -> Path:
        run_directory = self.run_directory(run_id)
        self.artifact_directory(run_id).mkdir(parents=True, exist_ok=True)
        self.log_directory(run_id).mkdir(parents=True, exist_ok=True)
        return run_directory

    def save_artifact(self, run_id: UUID, artifact_name: str, content: bytes) -> Path:
        if not isinstance(content, bytes):
            raise TypeError("Artifact content must be bytes.")
        target = self.artifact_directory(run_id) / _validate_file_name(artifact_name)
        self._write_atomically(target, content)
        return target

    def save_log(self, run_id: UUID, log_name: str, content: str) -> Path:
        if not isinstance(content, str):
            raise TypeError("Log content must be a string.")
        target = self.log_directory(run_id) / _validate_file_name(log_name)
        self._write_atomically(target, content.encode("utf-8"))
        return target

    def get_model_artifact(self, run_id: UUID) -> ModelArtifact | None:
        metadata_path = self.artifact_directory(run_id) / "model-artifact.json"
        if metadata_path.is_symlink():
            raise ValueError("Model artifact metadata must not be a symbolic link.")
        if not metadata_path.exists():
            return None
        if not metadata_path.is_file():
            raise ValueError("Model artifact metadata must be a regular file.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Model artifact metadata must be a JSON object.")
        try:
            artifact_run_id_value = metadata["run_id"]
            if not isinstance(artifact_run_id_value, str):
                raise ValueError("Model artifact run_id must be a string.")
            artifact_run_id = UUID(artifact_run_id_value)
            artifact_name = metadata["artifact_name"]
            artifact_location = metadata["artifact_location"]
            size_bytes = metadata["size_bytes"]
            checksum = metadata["checksum"]
            created_at = datetime.fromisoformat(metadata["created_at"])
            ServingStatus(metadata["serving_status"])
            artifact_path = self.artifact_directory(run_id) / "model.joblib"
            status = _model_artifact_status(
                artifact_path,
                artifact_name,
                artifact_location,
                size_bytes,
                checksum,
            )
            artifact = ModelArtifact(
                run_id=artifact_run_id,
                artifact_name=artifact_name,
                artifact_location=artifact_location,
                size_bytes=size_bytes,
                checksum=checksum,
                created_at=created_at,
                serving_status=status,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Model artifact metadata is invalid.") from exc
        if artifact.run_id != run_id:
            raise ValueError("Model artifact metadata belongs to another execution.")
        return artifact

    def _write_atomically(self, target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_run_id(run_id: UUID) -> None:
    if not isinstance(run_id, UUID):
        raise ValueError("run_id must be a UUID.")


def _model_artifact_status(
    artifact_path: Path,
    artifact_name: object,
    artifact_location: object,
    size_bytes: object,
    checksum: object,
) -> ServingStatus:
    expected_location = str(artifact_path.resolve())
    if (
        artifact_name != "model.joblib"
        or not isinstance(artifact_location, str)
        or Path(artifact_location).resolve() != Path(expected_location)
    ):
        return ServingStatus.INVALID
    if artifact_path.is_symlink():
        return ServingStatus.INVALID
    if not artifact_path.exists():
        return ServingStatus.MISSING
    if not artifact_path.is_file():
        return ServingStatus.INVALID
    content = artifact_path.read_bytes()
    if size_bytes != len(content) or checksum != hashlib.sha256(content).hexdigest():
        return ServingStatus.INVALID
    return ServingStatus.AVAILABLE


def _validate_file_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("Artifact and log names must be a single file name.")
    return name
