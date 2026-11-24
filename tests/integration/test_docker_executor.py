from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from application.ports import TrainingExecutionRequest
from infrastructure.docker.container_config import build_training_container_config
from infrastructure.docker.training_executor import DockerTrainingExecutor

_LOCAL_TEST_IMAGE = "postgres:16-alpine"
_PYTHON_TEST_IMAGE = "python:3.11.9-slim"


def test_docker_executor_runs_a_valid_training_fixture(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_local_image(client, _PYTHON_TEST_IMAGE)
    project = tmp_path / "valid-project"
    output = tmp_path / "valid-output"
    project.mkdir()
    output.mkdir()
    (project / "train.py").write_text("print('training succeeded')\n", encoding="utf-8")

    result = DockerTrainingExecutor(docker_client=client).execute(
        _executor_request(project, output)
    )

    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.out_of_memory
    assert any(log.message == "training succeeded" for log in result.logs)


def test_docker_executor_returns_logs_for_a_failing_training_fixture(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_local_image(client, _PYTHON_TEST_IMAGE)
    project = tmp_path / "failing-project"
    output = tmp_path / "failing-output"
    project.mkdir()
    output.mkdir()
    (project / "train.py").write_text(
        "print('intentional training failure')\nraise SystemExit(7)\n",
        encoding="utf-8",
    )

    result = DockerTrainingExecutor(docker_client=client).execute(
        _executor_request(project, output)
    )

    assert result.exit_code == 7
    assert not result.timed_out
    assert any(log.message == "intentional training failure" for log in result.logs)


def _executor_request(project: Path, output: Path) -> TrainingExecutionRequest:
    return TrainingExecutionRequest(
        run_id=uuid4(),
        image_id=_PYTHON_TEST_IMAGE,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=30,
    )


def test_docker_runtime_enforces_security_resource_and_mount_boundaries(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_local_image(client)
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    (project / "read-only.txt").write_text("input", encoding="utf-8")
    request = _request(project, output)
    config = build_training_container_config(request)
    config["entrypoint"] = ["/bin/sh"]
    config["command"] = [
        "-c",
        "set -eu; id -u > /output/effective-uid; touch /output/writable; "
        "touch /tmp/writable; "
        "if touch /workspace/forbidden 2>/dev/null; then exit 21; fi; "
        "if touch /etc/forbidden 2>/dev/null; then exit 22; fi",
    ]
    container = client.containers.create(
        image=_LOCAL_TEST_IMAGE,
        name=f"mlplatform-security-test-{request.run_id.hex}",
        **config,
    )

    try:
        container.start()
        result = container.wait(timeout=30)
        container.reload()
        attributes = container.attrs
        host_config = attributes["HostConfig"]
        mounts = {mount["Destination"]: mount for mount in attributes["Mounts"]}

        assert result["StatusCode"] == 0
        assert (output / "writable").is_file()
        assert (output / "effective-uid").read_text(encoding="utf-8").strip() != "0"
        assert host_config["NetworkMode"] == "none"
        assert host_config["Privileged"] is False
        assert host_config["CapDrop"] == ["ALL"]
        assert "no-new-privileges:true" in host_config["SecurityOpt"]
        assert host_config["ReadonlyRootfs"] is True
        assert host_config["NanoCpus"] == 1_000_000_000
        assert host_config["Memory"] == 1024**3
        assert mounts["/workspace"]["RW"] is False
        assert mounts["/output"]["RW"] is True
        assert "/tmp" in host_config["Tmpfs"]
    finally:
        container.remove(force=True)


def test_docker_runtime_rejects_writes_to_the_read_only_project_mount(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_local_image(client)
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    request = _request(project, output)
    config = build_training_container_config(request)
    config["entrypoint"] = ["/bin/sh"]
    config["command"] = ["-c", "touch /workspace/forbidden"]
    container = client.containers.create(
        image=_LOCAL_TEST_IMAGE,
        name=f"mlplatform-read-only-test-{request.run_id.hex}",
        **config,
    )

    try:
        container.start()
        result = container.wait(timeout=30)

        assert result["StatusCode"] != 0
        assert not (project / "forbidden").exists()
    finally:
        container.remove(force=True)


def _request(project: Path, output: Path) -> TrainingExecutionRequest:
    return TrainingExecutionRequest(
        run_id=uuid4(),
        image_id=_LOCAL_TEST_IMAGE,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=30,
    )


def _get_docker_client():
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


def _require_local_image(client, image: str = _LOCAL_TEST_IMAGE) -> None:
    try:
        client.images.get(image)
    except ImageNotFound:
        pytest.skip(f"Local integration-test image not available: {image}")
