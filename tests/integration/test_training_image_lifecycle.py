from __future__ import annotations

import platform
import shutil
import sqlite3
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from application.errors import ImageBuildError
from application.training.training_service import TrainingOptions, TrainingService
from domain.models import FailureReason, RunState
from infrastructure.config import PlatformConfig
from infrastructure.docker.image_builder import DockerImageBuilder
from infrastructure.docker.training_executor import DockerTrainingExecutor
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.log_repository import SQLiteLogRepository
from infrastructure.sqlite.metrics_repository import SQLiteMetricsRepository
from infrastructure.sqlite.schema import initialize_schema


def test_successful_image_lifecycle_and_dependency_build_failure(tmp_path: Path) -> None:
    client = _require_docker()
    _require_base_image(client)
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path / "home")})

    successful_project = tmp_path / "iris"
    shutil.copytree(Path("examples/iris"), successful_project)
    trainer_path = successful_project / "train.py"
    trainer_source = trainer_path.read_text(encoding="utf-8")
    future_import = "from __future__ import annotations\n"
    assert trainer_source.startswith(future_import)
    egress_probe = """\
import errno
import socket

try:
    with socket.create_connection(("1.1.1.1", 443), timeout=3):
        raise SystemExit("external training egress unexpectedly succeeded")
except OSError as error:
    if error.errno not in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN}:
        raise
    print(f"controlled external connection blocked: errno={error.errno}")

"""
    trainer_path.write_text(
        future_import + egress_probe + trainer_source[len(future_import) :],
        encoding="utf-8",
    )
    successful_connection = sqlite3.connect(tmp_path / "successful.sqlite3")
    initialize_schema(successful_connection)
    successful_service = _service(successful_connection, config)

    successful_run = successful_service.train(
        successful_project,
        options=TrainingOptions(
            cpus=1.0,
            memory_limit="1g",
            build_timeout_seconds=600,
            timeout_seconds=600,
        ),
    )

    assert successful_run.state is RunState.SUCCEEDED
    assert successful_run.docker_image is not None
    assert successful_run.docker_image.startswith("sha256:")
    assert successful_run.dependency_versions
    assert ("scikit-learn", "1.8.0") in successful_run.dependency_versions
    assert successful_run.log_location is not None
    assert "controlled external connection blocked" in Path(successful_run.log_location).read_text(
        encoding="utf-8"
    )
    success_directory = config.artifacts_dir / "executions" / successful_run.run_id.hex
    assert (success_directory / "artifacts" / "model.joblib").is_file()
    successful_connection.close()

    failing_project = tmp_path / "broken-project"
    (failing_project / "data").mkdir(parents=True)
    (failing_project / "train.py").write_text(
        "raise SystemExit('training must not start')\n",
        encoding="utf-8",
    )
    (failing_project / "requirements.txt").write_text(
        "package-that-does-not-exist-for-mlplatform==1.0.0\n",
        encoding="utf-8",
    )
    (failing_project / "data" / "records.csv").write_text("x\n1\n", encoding="utf-8")

    failing_connection = sqlite3.connect(tmp_path / "failing.sqlite3")
    initialize_schema(failing_connection)
    failing_service = _service(failing_connection, config)

    with pytest.raises(ImageBuildError) as error:
        failing_service.train(failing_project)

    failed = SQLiteExecutionRepository(failing_connection).get_run(error.value.run_id)
    assert failed is not None
    assert failed.state is RunState.FAILED
    assert failed.failure_reason is FailureReason.IMAGE_BUILD_ERROR
    assert failed.docker_image is None
    assert failed.started_at is None
    failing_connection.close()


def _service(connection: sqlite3.Connection, config: PlatformConfig) -> TrainingService:
    return TrainingService(
        execution_repository=SQLiteExecutionRepository(connection),
        metrics_repository=SQLiteMetricsRepository(connection),
        log_repository=SQLiteLogRepository(connection),
        artifact_store=FilesystemArtifactStore(config.artifacts_dir),
        image_builder=DockerImageBuilder(config),
        training_executor=DockerTrainingExecutor(),
        config=config,
    )


def _require_docker():
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


def _require_base_image(client) -> None:
    image = f"docker.io/library/python:{platform.python_version()}-slim"
    try:
        client.images.get(image)
    except ImageNotFound:
        try:
            client.images.pull(image)
        except DockerException as exc:
            pytest.skip(f"Local base image unavailable: {exc}")
