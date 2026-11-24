from __future__ import annotations

from pathlib import Path

import pytest

from application.errors import InferenceError
from infrastructure.docker.inference_executor import (
    DockerInferenceExecutor,
    build_inference_container_config,
)


class FakeContainer:
    def __init__(self, *, exit_code: int) -> None:
        self.attrs = {
            "State": {
                "Status": "created",
                "ExitCode": exit_code,
                "OOMKilled": False,
            }
        }
        self.removed = False

    def start(self) -> None:
        self.attrs["State"]["Status"] = "exited"

    def reload(self) -> None:
        return None

    def logs(self, *, stdout: bool, stderr: bool) -> bytes:
        return b'{"model_loaded":true}'

    def remove(self, *, force: bool) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container

    def create(self, **_: object) -> FakeContainer:
        return self.container


class FakeDockerClient:
    def __init__(self, container: FakeContainer) -> None:
        self.containers = FakeContainers(container)


@pytest.mark.parametrize(("exit_code", "succeeds"), [(0, True), (1, False)])
def test_inference_executor_removes_temporary_container_on_success_and_failure(
    tmp_path: Path,
    monkeypatch,
    exit_code: int,
    succeeds: bool,
) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    container = FakeContainer(exit_code=exit_code)
    executor = DockerInferenceExecutor(docker_client=FakeDockerClient(container))
    monkeypatch.setattr("infrastructure.docker.inference_executor.os.getuid", lambda: 501)

    if succeeds:
        executor.validate_model("sha256:" + "a" * 64, str(artifact), 0.5, "512m")
    else:
        with pytest.raises(InferenceError):
            executor.validate_model("sha256:" + "a" * 64, str(artifact), 0.5, "512m")

    assert container.removed


def test_inference_container_is_isolated_and_mounts_only_read_only_model_and_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    input_path = tmp_path / "instances.json"
    input_path.write_text("[[1.0]]", encoding="utf-8")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-forwarded")

    config = build_inference_container_config(
        image_id="sha256:" + "a" * 64,
        artifact_path=model_path,
        cpus=0.5,
        memory_limit="512m",
        command=["predict"],
        input_path=input_path,
    )

    assert config["network_mode"] == "none"
    assert config["privileged"] is False
    assert config["cap_drop"] == ["ALL"]
    assert config["security_opt"] == ["no-new-privileges:true"]
    assert config["read_only"] is True
    assert config["nano_cpus"] == 500_000_000
    assert config["mem_limit"] == 512 * 1024**2
    assert config["environment"] == {"HOME": "/tmp"}
    assert config["volumes"] == {
        str(model_path.resolve()): {"bind": "/model/model.joblib", "mode": "ro"},
        str(input_path.resolve()): {"bind": "/request/instances.json", "mode": "ro"},
    }


@pytest.mark.parametrize(
    ("image_id", "cpus", "memory_limit", "message"),
    [
        ("latest", 1.0, "1g", "immutable"),
        ("sha256:" + "a" * 64, 0.0, "1g", "CPUs"),
        ("sha256:" + "a" * 64, 1.1, "1g", "CPUs"),
        ("sha256:" + "a" * 64, 1.0, "2g", "memory"),
    ],
)
def test_inference_container_rejects_mutable_images_and_invalid_limits(
    tmp_path: Path,
    image_id: str,
    cpus: float,
    memory_limit: str,
    message: str,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")

    with pytest.raises(ValueError, match=message):
        build_inference_container_config(
            image_id=image_id,
            artifact_path=model_path,
            cpus=cpus,
            memory_limit=memory_limit,
            command=["validate"],
        )
