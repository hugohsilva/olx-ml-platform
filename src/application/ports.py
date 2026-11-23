from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from domain.models import LogRecord, Metric, Run


@dataclass(frozen=True, slots=True)
class ImageBuildRequest:
    project_directory: Path
    requirements_file: str
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
    image_id: str
    project_directory: Path
    output_directory: Path
    cpus: float
    memory_limit: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class TrainingExecutionResult:
    exit_code: int
    logs: tuple[LogRecord, ...]
    timed_out: bool = False


class ExecutionRepository(Protocol):
    def save_run(self, run: Run) -> None: ...

    def get_run(self, run_id: UUID) -> Run | None: ...


class MetricsRepository(Protocol):
    def save_metrics(self, run_id: UUID, metrics: tuple[Metric, ...]) -> None: ...

    def get_metrics(self, run_id: UUID) -> tuple[Metric, ...]: ...


class ArtifactStore(Protocol):
    def save_artifact(self, run_id: UUID, artifact_name: str, content: bytes) -> Path: ...

    def save_log(self, run_id: UUID, log_name: str, content: str) -> Path: ...


class ImageBuilder(Protocol):
    def build(self, request: ImageBuildRequest) -> ImageBuildResult: ...


class TrainingExecutor(Protocol):
    def execute(self, request: TrainingExecutionRequest) -> TrainingExecutionResult: ...
