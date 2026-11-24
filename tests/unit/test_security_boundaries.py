from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from application.ports import TrainingExecutionRequest
from infrastructure.docker.container_config import build_training_container_config
from infrastructure.docker.image_builder import _redact_log_line
from infrastructure.docker.inference_executor import build_inference_container_config


def test_build_log_redacts_credentials_and_secret_values() -> None:
    log_line = _redact_log_line("https://alice:password@packages.example/token password=hunter2")

    assert log_line == "https://<redacted>@packages.example/token password=<redacted>"
    assert "alice" not in log_line
    assert "hunter2" not in log_line


def test_training_container_receives_no_host_secrets_and_limited_mounts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret")
    request = TrainingExecutionRequest(
        run_id=uuid4(),
        image_id="sha256:" + "a" * 64,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=600,
    )

    config = build_training_container_config(request)

    assert config["environment"] == {
        "HOME": "/tmp",
        "MLPLATFORM_OUTPUT_DIR": "/output",
        "MLPLATFORM_RUN_ID": str(request.run_id),
    }
    assert "host-secret" not in repr(config)
    assert config["volumes"] == {
        str(project.resolve()): {"bind": "/workspace", "mode": "ro"},
        str(output.resolve()): {"bind": "/output", "mode": "rw"},
    }


def test_inference_container_receives_no_host_secrets_or_extra_mounts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "model.joblib"
    input_path = tmp_path / "instances.json"
    artifact.write_bytes(b"model")
    input_path.write_text("[[1.0]]", encoding="utf-8")
    monkeypatch.setenv("API_SECRET", "host-secret")

    config = build_inference_container_config(
        image_id="sha256:" + "a" * 64,
        artifact_path=artifact,
        cpus=1.0,
        memory_limit="1g",
        command=["predict"],
        input_path=input_path,
    )

    assert config["environment"] == {"HOME": "/tmp"}
    assert "host-secret" not in repr(config)
    assert set(config["volumes"]) == {str(artifact.resolve()), str(input_path.resolve())}
    assert all(mount["mode"] == "ro" for mount in config["volumes"].values())
