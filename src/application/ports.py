from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from domain.models import InferenceRequest, JsonScalar, LogRecord, Metric, ModelArtifact, Run


@dataclass(frozen=True, slots=True)
class ImageBuildRequest:
    run_id: UUID
    project_directory: Path
    requirements_file: str
    python_version: str
    base_image_registry: str
    allowed_repositories: tuple[str, ...]
    cpus: float
    memory_limit: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class ImageBuildResult:
    image_id: str
    dependency_versions: tuple[tuple[str, str], ...]
    logs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingExecutionRequest:
    run_id: UUID
    image_id: str
    project_directory: Path
    entrypoint: str
    output_directory: Path
    started_at: datetime
    cpus: float
    memory_limit: str
    timeout_seconds: int
    network_disabled: bool = True


@dataclass(frozen=True, slots=True)
class TrainingExecutionResult:
    exit_code: int
    logs: tuple[LogRecord, ...]
    timed_out: bool = False
    out_of_memory: bool = False


@dataclass(frozen=True, slots=True)
class InferenceExecutionRequest:
    run_id: UUID
    image_id: str
    artifact_location: str
    instances: InferenceRequest
    cpus: float
    memory_limit: str


class ExecutionRepository(Protocol):
    def save_run(self, run: Run) -> None: ...

    def get_run(self, run_id: UUID) -> Run | None: ...

    def list_runs(self) -> tuple[Run, ...]: ...


class MetricsRepository(Protocol):
    def save_metrics(self, run_id: UUID, metrics: tuple[Metric, ...]) -> None: ...

    def get_metrics(self, run_id: UUID) -> tuple[Metric, ...]: ...


class LogRepository(Protocol):
    def save_logs(self, run_id: UUID, logs: tuple[LogRecord, ...]) -> None: ...


class ArtifactStore(Protocol):
    def save_artifact(self, run_id: UUID, artifact_name: str, content: bytes) -> Path: ...

    def save_log(self, run_id: UUID, log_name: str, content: str) -> Path: ...

    def get_model_artifact(self, run_id: UUID) -> ModelArtifact | None: ...


class ImageBuilder(Protocol):
    def build(self, request: ImageBuildRequest) -> ImageBuildResult: ...


class TrainingExecutor(Protocol):
    def execute(self, request: TrainingExecutionRequest) -> TrainingExecutionResult: ...


class ModelExecutor(Protocol):
    def validate_model(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
    ) -> None: ...

    def predict(self, request: InferenceExecutionRequest) -> tuple[JsonScalar, ...]: ...
