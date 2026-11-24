from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from application.errors import ExecutionNotFoundError, ModelUnavailableError
from application.inference.model_service import ModelService
from application.ports import InferenceExecutionRequest
from application.training.artifact_ingestor import ArtifactIngestor
from domain.executions.state_machine import transition_run
from domain.models import Run, RunState, ServingStatus
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


class FakeModelExecutor:
    def __init__(self) -> None:
        self.validated: list[tuple[str, str, float, str]] = []
        self.prediction_requests: list[InferenceExecutionRequest] = []

    def validate_model(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
    ) -> None:
        self.validated.append((image_id, artifact_location, cpus, memory_limit))

    def predict(self, request: InferenceExecutionRequest) -> tuple[object, ...]:
        self.prediction_requests.append(request)
        return (2,)


def test_model_service_selects_and_validates_the_requested_successful_run(tmp_path: Path) -> None:
    run_id = uuid4()
    timestamp = datetime.now(timezone.utc)
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    run = transition_run(
        transition_run(
            Run(
                run_id=run_id,
                project_id="iris",
                docker_image="sha256:" + "a" * 64,
                created_at=timestamp,
            ),
            RunState.RUNNING,
            at=timestamp,
        ),
        RunState.SUCCEEDED,
        at=timestamp,
    )
    repository.save_run(run)
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.joblib").write_bytes(b"serialized model")
    artifact = ArtifactIngestor(artifact_store).ingest(run_id, output)
    executor = FakeModelExecutor()

    served_model = ModelService(repository, artifact_store, executor).load_model(run_id)

    assert served_model.run_id == run_id
    assert served_model.image_id == run.docker_image
    assert served_model.artifact_location == artifact.artifact_location
    assert executor.validated == [
        (run.docker_image, artifact.artifact_location, run.max_cpus, run.memory_limit)
    ]
    connection.close()


def test_model_service_rejects_missing_run_and_unavailable_artifact(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    executor = FakeModelExecutor()
    service = ModelService(repository, artifact_store, executor)

    with pytest.raises(ExecutionNotFoundError):
        service.load_model(uuid4())

    run_id = uuid4()
    timestamp = datetime.now(timezone.utc)
    run = transition_run(
        transition_run(
            Run(
                run_id=run_id,
                project_id="iris",
                docker_image="sha256:" + "b" * 64,
                created_at=timestamp,
            ),
            RunState.RUNNING,
            at=timestamp,
        ),
        RunState.SUCCEEDED,
        at=timestamp,
    )
    repository.save_run(run)

    with pytest.raises(ModelUnavailableError, match="artifact"):
        service.load_model(run_id)

    assert executor.validated == []
    connection.close()


def test_model_service_rejects_runs_that_have_not_succeeded(tmp_path: Path) -> None:
    run_id = uuid4()
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    repository.save_run(Run(run_id=run_id, project_id="iris"))
    executor = FakeModelExecutor()

    with pytest.raises(ModelUnavailableError, match="succeeded"):
        ModelService(
            repository,
            FilesystemArtifactStore(tmp_path / "artifacts"),
            executor,
        ).load_model(run_id)

    assert executor.validated == []
    connection.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_model_service_rejects_missing_or_modified_model_artifacts(
    tmp_path: Path,
    damage: str,
) -> None:
    run_id = uuid4()
    timestamp = datetime.now(timezone.utc)
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    run = transition_run(
        transition_run(
            Run(
                run_id=run_id,
                project_id="iris",
                docker_image="sha256:" + "c" * 64,
                created_at=timestamp,
            ),
            RunState.RUNNING,
            at=timestamp,
        ),
        RunState.SUCCEEDED,
        at=timestamp,
    )
    repository.save_run(run)
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.joblib").write_bytes(b"serialized model")
    ArtifactIngestor(artifact_store).ingest(run_id, output)
    artifact_path = artifact_store.artifact_directory(run_id) / "model.joblib"
    if damage == "missing":
        artifact_path.unlink()
        expected_status = ServingStatus.MISSING
    else:
        artifact_path.write_bytes(b"changed model")
        expected_status = ServingStatus.INVALID
    executor = FakeModelExecutor()

    artifact = artifact_store.get_model_artifact(run_id)

    assert artifact is not None
    assert artifact.serving_status is expected_status
    with pytest.raises(ModelUnavailableError, match="artifact"):
        ModelService(repository, artifact_store, executor).load_model(run_id)
    assert executor.validated == []
    connection.close()
