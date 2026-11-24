from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from application.ports import InferenceExecutionRequest
from application.training.artifact_ingestor import ArtifactIngestor
from cli.main import app
from domain.executions.state_machine import transition_run
from domain.models import Run, RunState
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


class FakeModelExecutor:
    def __init__(self) -> None:
        self.validated: list[tuple[str, str, float, str]] = []

    def validate_model(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
    ) -> None:
        self.validated.append((image_id, artifact_location, cpus, memory_limit))

    def predict(self, request: InferenceExecutionRequest) -> tuple[int, ...]:
        return tuple(0 for _ in request.instances.instances)


def test_serve_cli_validates_model_before_starting_http_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = uuid4()
    home = tmp_path / "home"
    home.mkdir()
    database_path = home / "metadata.sqlite3"
    connection = sqlite3.connect(database_path)
    initialize_schema(connection)
    timestamp = datetime.now(timezone.utc)
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
    SQLiteExecutionRepository(connection).save_run(run)
    connection.close()

    artifact_store = FilesystemArtifactStore(home / "artifacts")
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "model.joblib").write_bytes(b"serialized model")
    artifact = ArtifactIngestor(artifact_store).ingest(run_id, output_directory)
    executor = FakeModelExecutor()
    called: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setenv("MLPLATFORM_HOME", str(home))
    monkeypatch.setattr("cli.commands.serve.DockerInferenceExecutor", lambda: executor)
    monkeypatch.setattr(
        "cli.commands.serve.uvicorn.run",
        lambda api, **kwargs: called.append((api, kwargs)),
    )

    result = CliRunner().invoke(app, ["serve", str(run_id), "--port", "8765"])

    assert result.exit_code == 0
    assert f"serving run_id={run_id}" in result.output
    assert executor.validated == [("sha256:" + "a" * 64, artifact.artifact_location, 1.0, "1g")]
    assert len(called) == 1
    assert called[0][1] == {"host": "127.0.0.1", "port": 8765}


def test_serve_cli_does_not_start_server_for_an_unknown_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLPLATFORM_HOME", str(tmp_path))
    server_started = False

    def run_server(*args: object, **kwargs: object) -> None:
        nonlocal server_started
        server_started = True

    monkeypatch.setattr("cli.commands.serve.uvicorn.run", run_server)
    result = CliRunner().invoke(
        app,
        ["serve", "00000000-0000-0000-0000-000000000001"],
    )

    assert result.exit_code == 1
    assert "was not found" in result.output
    assert not server_started
