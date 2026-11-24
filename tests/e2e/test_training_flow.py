from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException
from typer.testing import CliRunner

from cli.main import app
from infrastructure.sqlite.schema import initialize_schema


def test_training_only_iris_flow_persists_successful_run(tmp_path: Path) -> None:
    _require_docker()
    monkeypatch_home = tmp_path / "mlplatform"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "examples/iris",
            "--timeout-seconds",
            "600",
            "--build-timeout-seconds",
            "600",
            "--cpus",
            "1",
            "--memory-limit",
            "1g",
        ],
        env={"MLPLATFORM_HOME": str(monkeypatch_home)},
    )

    assert result.exit_code == 0, result.output
    lines = dict(line.split("=", 1) for line in result.output.splitlines() if "=" in line)
    run_id = lines["run_id"]
    assert lines["state"] == "SUCCEEDED"

    connection = sqlite3.connect(monkeypatch_home / "metadata.sqlite3")
    initialize_schema(connection)
    row = connection.execute(
        "SELECT state, docker_image, dependency_versions_json FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    connection.close()

    assert row is not None
    assert row[0] == "SUCCEEDED"
    assert row[1].startswith("sha256:")
    assert isinstance(json.loads(row[2]), list)

    run_directory = monkeypatch_home / "artifacts" / "executions" / run_id.replace("-", "")
    assert (run_directory / "artifacts" / "model.joblib").is_file()
    assert (run_directory / "artifacts" / "metrics.json").is_file()


def _require_docker() -> None:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")
