from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4


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
