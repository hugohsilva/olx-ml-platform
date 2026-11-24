from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from application.errors import ExecutionNotFoundError, ModelUnavailableError
from application.ports import ArtifactStore, ExecutionRepository, ModelExecutor
from domain.models import RunState, ServingStatus


@dataclass(frozen=True, slots=True)
class ServedModel:
    run_id: UUID
    image_id: str
    artifact_location: str
    cpus: float
    memory_limit: str


class ModelService:
    def __init__(
        self,
        execution_repository: ExecutionRepository,
        artifact_store: ArtifactStore,
        model_executor: ModelExecutor,
    ) -> None:
        self._execution_repository = execution_repository
        self._artifact_store = artifact_store
        self._model_executor = model_executor

    def load_model(self, run_id: UUID) -> ServedModel:
        run = self._execution_repository.get_run(run_id)
        if run is None:
            raise ExecutionNotFoundError(f"Execution {run_id} was not found.", run_id=run_id)
        if run.state is not RunState.SUCCEEDED or run.docker_image is None:
            raise ModelUnavailableError(
                "The execution has not succeeded with a built model image.",
                run_id=run_id,
            )

        artifact = self._artifact_store.get_model_artifact(run_id)
        if artifact is None or artifact.serving_status is not ServingStatus.AVAILABLE:
            raise ModelUnavailableError(
                "A valid model artifact is not available for this execution.",
                run_id=run_id,
            )
        self._model_executor.validate_model(
            run.docker_image,
            artifact.artifact_location,
            run.max_cpus,
            run.memory_limit,
        )
        return ServedModel(
            run_id=run_id,
            image_id=run.docker_image,
            artifact_location=artifact.artifact_location,
            cpus=run.max_cpus,
            memory_limit=run.memory_limit,
        )
