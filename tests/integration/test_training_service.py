from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from application.errors import (
    ApplicationError,
    ImageBuildError,
    ImageBuildTimeout,
    MissingOutputError,
    OutOfMemoryError,
    ResourceLimitError,
    TrainingError,
    TrainingTimeout,
    ValidationError,
)
from application.ports import (
    ImageBuildRequest,
    ImageBuildResult,
    TrainingExecutionRequest,
    TrainingExecutionResult,
)
from application.training.training_service import TrainingOptions, TrainingService
from domain.models import FailureReason, LogRecord, LogStream, RunState
from infrastructure.config import PlatformConfig
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.log_repository import SQLiteLogRepository
from infrastructure.sqlite.metrics_repository import SQLiteMetricsRepository
from infrastructure.sqlite.schema import initialize_schema


class FakeImageBuilder:
    def __init__(self, *, error: ApplicationError | None = None) -> None:
        self.requests: list[ImageBuildRequest] = []
        self.error = error

    def build(self, request: ImageBuildRequest) -> ImageBuildResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ImageBuildResult(
            image_id="sha256:" + "a" * 64,
            dependency_versions=(("scikit-learn", "1.6.1"),),
            logs=("image build completed",),
        )


class FakeTrainingExecutor:
    def __init__(
        self,
        *,
        timed_out: bool = False,
        out_of_memory: bool = False,
        write_outputs: bool = True,
        exit_code: int = 0,
        error: ApplicationError | None = None,
    ) -> None:
        self.timed_out = timed_out
        self.out_of_memory = out_of_memory
        self.write_outputs = write_outputs
        self.exit_code = exit_code
        self.error = error
        self.requests: list[TrainingExecutionRequest] = []

    def execute(self, request: TrainingExecutionRequest) -> TrainingExecutionResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.write_outputs:
            request.output_directory.mkdir(parents=True, exist_ok=True)
            (request.output_directory / "model.joblib").write_bytes(b"trained-model")
            (request.output_directory / "metrics.json").write_text(
                json.dumps({"accuracy": 0.96}),
                encoding="utf-8",
            )
            (request.output_directory / "parameters.json").write_text(
                json.dumps({"random_state": 7}),
                encoding="utf-8",
            )
        return TrainingExecutionResult(
            exit_code=self.exit_code,
            timed_out=self.timed_out,
            out_of_memory=self.out_of_memory,
            logs=(
                LogRecord(
                    run_id=request.run_id,
                    sequence=0,
                    timestamp=request.started_at,
                    stream=LogStream.STDOUT,
                    message="training completed",
                ),
            ),
        )


def create_project(root: Path) -> Path:
    project = root / "iris-project"
    (project / "data").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "requirements.txt").write_text("scikit-learn==1.6.1\n", encoding="utf-8")
    (project / "data" / "records.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    return project


def create_service(tmp_path: Path, executor: FakeTrainingExecutor):
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    execution_repository = SQLiteExecutionRepository(connection)
    image_builder = FakeImageBuilder()
    artifact_store = FilesystemArtifactStore(tmp_path / "store")
    service = TrainingService(
        execution_repository=execution_repository,
        metrics_repository=SQLiteMetricsRepository(connection),
        log_repository=SQLiteLogRepository(connection),
        artifact_store=artifact_store,
        image_builder=image_builder,
        training_executor=executor,
        config=PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path / "home")}),
    )
    return service, execution_repository, image_builder, artifact_store, connection


def create_service_with_builder_error(
    tmp_path: Path,
    executor: FakeTrainingExecutor,
    error: ApplicationError,
):
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    execution_repository = SQLiteExecutionRepository(connection)
    image_builder = FakeImageBuilder(error=error)
    artifact_store = FilesystemArtifactStore(tmp_path / "store")
    service = TrainingService(
        execution_repository=execution_repository,
        metrics_repository=SQLiteMetricsRepository(connection),
        log_repository=SQLiteLogRepository(connection),
        artifact_store=artifact_store,
        image_builder=image_builder,
        training_executor=executor,
        config=PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path / "home")}),
    )
    return service, execution_repository, executor, artifact_store, connection


def test_training_service_persists_run_hashes_metrics_logs_and_model_artifact(tmp_path) -> None:
    project_path = create_project(tmp_path)
    executor = FakeTrainingExecutor()
    service, repository, image_builder, artifact_store, connection = create_service(
        tmp_path, executor
    )

    run = service.train(project_path)
    persisted = repository.get_run(run.run_id)

    assert persisted is not None
    assert persisted.state is RunState.SUCCEEDED
    assert persisted.code_hash is not None and len(persisted.code_hash) == 64
    assert persisted.dataset_hash is not None and len(persisted.dataset_hash) == 64
    assert persisted.config_hash is not None and len(persisted.config_hash) == 64
    assert persisted.docker_image == "sha256:" + "a" * 64
    assert persisted.dependency_versions == (("scikit-learn", "1.6.1"),)
    assert persisted.metrics[0].name == "accuracy"
    assert persisted.metrics[0].value == 0.96
    assert persisted.started_at is not None
    assert persisted.finished_at is not None
    assert image_builder.requests[0].run_id == run.run_id
    assert executor.requests[0].image_id == persisted.docker_image
    assert executor.requests[0].network_disabled is True
    assert (
        artifact_store.artifact_directory(run.run_id) / "model.joblib"
    ).read_bytes() == b"trained-model"
    artifact_metadata = json.loads(
        (artifact_store.artifact_directory(run.run_id) / "model-artifact.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact_metadata["run_id"] == str(run.run_id)
    assert artifact_metadata["serving_status"] == "available"
    stored_metrics = connection.execute(
        "SELECT name, value_json FROM metrics WHERE run_id = ?",
        (str(run.run_id),),
    ).fetchall()
    assert [(row["name"], row["value_json"]) for row in stored_metrics] == [("accuracy", "0.96")]
    stored_logs = connection.execute(
        "SELECT message FROM logs WHERE run_id = ?",
        (str(run.run_id),),
    ).fetchall()
    assert [row["message"] for row in stored_logs] == [
        "image build completed",
        "training completed",
    ]
    assert persisted.log_location is not None
    assert Path(persisted.log_location).exists()
    connection.close()


def test_training_service_keeps_build_and_training_timeouts_independent(tmp_path) -> None:
    project_path = create_project(tmp_path)
    executor = FakeTrainingExecutor()
    service, _, image_builder, _, connection = create_service(
        tmp_path,
        executor,
    )

    service.train(
        project_path,
        options=TrainingOptions(build_timeout_seconds=45, timeout_seconds=90),
    )

    assert image_builder.requests[0].timeout_seconds == 45
    assert executor.requests[0].timeout_seconds == 90
    connection.close()


def test_training_service_persists_timeout_failure_and_exposes_run_id(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, _, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(timed_out=True, write_outputs=False),
    )

    with pytest.raises(TrainingTimeout) as error:
        service.train(project_path)

    assert error.value.run_id is not None
    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.TIMEOUT
    assert persisted.finished_at is not None
    connection.close()


def test_training_service_persists_training_oom_failure(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, _, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(out_of_memory=True),
    )

    with pytest.raises(OutOfMemoryError, match="memory limit") as error:
        service.train(project_path)

    run = repository.get_run(UUID(str(error.value.run_id)))
    assert run is not None
    assert run.state is RunState.FAILED
    assert run.failure_reason is FailureReason.OOM_KILLED
    connection.close()


def test_training_service_persists_missing_output_failure(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, _, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(write_outputs=False),
    )

    with pytest.raises(MissingOutputError) as error:
        service.train(project_path)

    assert getattr(error.value, "run_id", None) is not None
    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.MISSING_OUTPUT
    connection.close()


def test_training_service_persists_validation_failure_without_building(tmp_path) -> None:
    project_path = create_project(tmp_path)
    (project_path / "train.py").unlink()
    service, repository, image_builder, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(),
    )

    created_runs = []
    with pytest.raises(ValidationError) as error:
        service.train(project_path, on_run_created=created_runs.append)

    assert error.value.run_id is not None
    assert len(created_runs) == 1
    assert created_runs[0].run_id == error.value.run_id
    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.VALIDATION
    assert persisted.code_hash is None
    assert persisted.dataset_hash is None
    assert persisted.config_hash is None
    assert not image_builder.requests
    connection.close()


def test_training_service_persists_build_failure_and_never_starts_training(tmp_path) -> None:
    project_path = create_project(tmp_path)
    executor = FakeTrainingExecutor()
    service, repository, _, artifact_store, connection = create_service_with_builder_error(
        tmp_path,
        executor,
        ImageBuildError("dependency build failed", logs=("pip install failed",)),
    )

    with pytest.raises(ImageBuildError) as error:
        service.train(project_path)

    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.IMAGE_BUILD_ERROR
    assert persisted.image_build_started_at is not None
    assert persisted.image_build_finished_at is not None
    assert persisted.image_build_log_location is not None
    assert Path(persisted.image_build_log_location).exists()
    assert not executor.requests
    assert (artifact_store.log_directory(persisted.run_id) / "image-build.log").read_text(
        encoding="utf-8"
    ).find("pip install failed") >= 0
    connection.close()


def test_training_service_persists_build_oom_as_image_build_error(tmp_path) -> None:
    project_path = create_project(tmp_path)
    executor = FakeTrainingExecutor()
    service, repository, _, artifact_store, connection = create_service_with_builder_error(
        tmp_path,
        executor,
        ImageBuildError(
            "Docker Buildx command failed with status 137.",
            logs=("out of memory",),
        ),
    )

    with pytest.raises(ImageBuildError) as error:
        service.train(project_path)

    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.IMAGE_BUILD_ERROR
    assert persisted.image_build_log_location is not None
    assert "out of memory" in Path(persisted.image_build_log_location).read_text(encoding="utf-8")
    assert not executor.requests
    connection.close()


def test_training_service_persists_build_timeout_and_never_starts_training(tmp_path) -> None:
    project_path = create_project(tmp_path)
    executor = FakeTrainingExecutor()
    service, repository, _, _, connection = create_service_with_builder_error(
        tmp_path,
        executor,
        ImageBuildTimeout("build timed out", logs=("partial build",)),
    )

    with pytest.raises(ImageBuildTimeout) as error:
        service.train(project_path)

    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.IMAGE_BUILD_TIMEOUT
    assert persisted.image_build_log_location is not None
    assert not executor.requests
    connection.close()


def test_training_service_persists_resource_limit_and_container_failures(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, _, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(error=ResourceLimitError("resource limits unavailable")),
    )

    with pytest.raises(ResourceLimitError) as error:
        service.train(project_path)

    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.RESOURCE_LIMIT
    connection.close()


def test_training_service_persists_nonzero_training_exit_as_training_error(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, _, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(exit_code=7),
    )

    with pytest.raises(TrainingError) as error:
        service.train(project_path)

    persisted = repository.get_run(UUID(str(error.value.run_id)))
    assert persisted is not None
    assert persisted.state is RunState.FAILED
    assert persisted.failure_reason is FailureReason.TRAINING_ERROR
    connection.close()


def test_training_service_rejects_resource_limits_before_build(tmp_path) -> None:
    project_path = create_project(tmp_path)
    service, repository, image_builder, _, connection = create_service(
        tmp_path,
        FakeTrainingExecutor(),
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        service.train(project_path, options=TrainingOptions(timeout_seconds=1801))

    assert not image_builder.requests
    assert repository.get_run(uuid4()) is None
    connection.close()
