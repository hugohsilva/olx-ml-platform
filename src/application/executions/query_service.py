from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from application.errors import ExecutionNotFoundError
from application.ports import ArtifactStore, ExecutionRepository
from domain.models import FailureReason, Metric, ModelArtifact, Run, RunState


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    run_id: UUID
    project_id: str
    state: RunState
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class ExecutionDetail:
    run_id: UUID
    project_id: str
    state: RunState
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    parameters: tuple[tuple[str, object], ...]
    metrics: tuple[Metric, ...]
    log_location: str | None
    image_build_log_location: str | None
    docker_image: str | None
    artifact_location: str | None
    failure_reason: FailureReason | None
    failure_message: str | None


class ExecutionQueryService:
    def __init__(
        self,
        execution_repository: ExecutionRepository,
        artifact_store: ArtifactStore,
    ) -> None:
        self._execution_repository = execution_repository
        self._artifact_store = artifact_store

    def list_executions(self) -> tuple[ExecutionSummary, ...]:
        return tuple(_summary(run) for run in self._execution_repository.list_runs())

    def get_execution(self, run_id: UUID) -> ExecutionDetail:
        run = self._execution_repository.get_run(run_id)
        if run is None:
            raise ExecutionNotFoundError(f"Execution {run_id} was not found.")
        artifact = self._artifact_store.get_model_artifact(run_id)
        return _detail(run, artifact)


def _summary(run: Run) -> ExecutionSummary:
    return ExecutionSummary(
        run_id=run.run_id,
        project_id=run.project_id,
        state=run.state,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _detail(run: Run, artifact: ModelArtifact | None) -> ExecutionDetail:
    return ExecutionDetail(
        run_id=run.run_id,
        project_id=run.project_id,
        state=run.state,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        parameters=run.parameters,
        metrics=run.metrics,
        log_location=run.log_location,
        image_build_log_location=run.image_build_log_location,
        docker_image=run.docker_image,
        artifact_location=artifact.artifact_location if artifact is not None else None,
        failure_reason=run.failure_reason,
        failure_message=run.failure_message,
    )
