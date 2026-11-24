from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from uuid import UUID

from typer.testing import CliRunner

from application.training.artifact_ingestor import ArtifactIngestor
from cli.main import app
from domain.executions.state_machine import transition_run
from domain.models import Metric, Run, RunState
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


def test_cli_lists_and_shows_persisted_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLPLATFORM_HOME", str(tmp_path))
    database_path = tmp_path / "metadata.sqlite3"
    connection = sqlite3.connect(database_path)
    initialize_schema(connection)
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = Run(
        run_id=run_id,
        project_id="iris",
        created_at=timestamp,
        metrics=(Metric(run_id=run_id, name="accuracy", value=0.95),),
        log_location="/logs/training.log",
    )
    run = transition_run(
        transition_run(run, RunState.RUNNING, at=timestamp),
        RunState.SUCCEEDED,
        at=timestamp,
    )
    SQLiteExecutionRepository(connection).save_run(run)
    connection.close()

    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "model.joblib").write_bytes(b"test model")
    model_artifact = ArtifactIngestor(artifact_store).ingest(run_id, output_directory)
    assert model_artifact.checksum == hashlib.sha256(b"test model").hexdigest()

    runner = CliRunner()
    listed = runner.invoke(app, ["executions", "list"])
    shown = runner.invoke(app, ["executions", "show", str(run_id)])
    text_shown = runner.invoke(app, ["executions", "show", str(run_id), "--format", "text"])
    missing = runner.invoke(app, ["executions", "show", "00000000-0000-0000-0000-000000000099"])

    assert listed.exit_code == 0
    assert json.loads(listed.output)[0]["run_id"] == str(run_id)
    assert shown.exit_code == 0
    assert json.loads(shown.output)["metrics"] == [
        {"name": "accuracy", "unit": None, "value": 0.95}
    ]
    assert json.loads(shown.output)["artifact_location"] == model_artifact.artifact_location
    assert text_shown.exit_code == 0
    assert f"artifact_location: {model_artifact.artifact_location}" in text_shown.output
    assert missing.exit_code == 1
    assert "was not found" in missing.output
