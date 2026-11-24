from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from application.errors import ResourceLimitError
from application.ports import TrainingExecutionRequest
from infrastructure.docker.training_executor import DockerTrainingExecutor


class FakeContainer:
    def __init__(
        self,
        *,
        out_of_memory: bool = False,
        remains_running: bool = False,
        resource_override: dict[str, int] | None = None,
    ) -> None:
        self.out_of_memory = out_of_memory
        self.remains_running = remains_running
        self.resource_override = resource_override or {}
        self.attrs = {
            "State": {
                "Status": "created",
                "ExitCode": None,
                "OOMKilled": out_of_memory,
            },
            "HostConfig": {},
        }
        self.options: dict[str, object] = {}
        self.removed = False

    def start(self) -> None:
        if not self.remains_running:
            self.attrs["State"]["Status"] = "exited"
            self.attrs["State"]["ExitCode"] = 137 if self.out_of_memory else 0

    def reload(self) -> None:
        pass

    def logs(self, *, stdout: bool, stderr: bool, timestamps: bool) -> bytes:
        if stdout:
            return b"2026-01-01T00:00:00.000000000Z training output\n"
        return b""

    def remove(self, *, force: bool) -> None:
        self.removed = force

    def kill(self) -> None:
        self.attrs["State"]["Status"] = "exited"
        self.attrs["State"]["ExitCode"] = 137


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container

    def create(self, **options: object) -> FakeContainer:
        self.container.options = options
        self.container.attrs["HostConfig"] = {
            "NanoCpus": options["nano_cpus"],
            "Memory": options["mem_limit"],
            "MemorySwap": options["memswap_limit"],
            **self.container.resource_override,
        }
        return self.container


class FakeDockerClient:
    def __init__(self, container: FakeContainer) -> None:
        self.containers = FakeContainers(container)


def test_executor_passes_enforced_runtime_profile_and_collects_logs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    container = FakeContainer()
    executor = DockerTrainingExecutor(docker_client=FakeDockerClient(container))

    result = executor.execute(_request(project, output))

    options = container.options
    assert options["network_mode"] == "none"
    assert options["privileged"] is False
    assert options["cap_drop"] == ["ALL"]
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert options["read_only"] is True
    assert options["nano_cpus"] == 1_000_000_000
    assert options["mem_limit"] == 1024**3
    assert options["user"] != "0:0"
    assert container.removed
    assert result.exit_code == 0
    assert result.logs[0].message == "training output"
    assert result.logs[0].sequence == 0


def test_executor_reports_oom_from_container_state(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    container = FakeContainer(out_of_memory=True)
    executor = DockerTrainingExecutor(docker_client=FakeDockerClient(container))

    result = executor.execute(_request(project, tmp_path / "output"))

    assert result.out_of_memory
    assert result.exit_code == 137


def test_executor_rejects_docker_that_did_not_apply_resource_limits(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    container = FakeContainer(resource_override={"Memory": 1})
    executor = DockerTrainingExecutor(docker_client=FakeDockerClient(container))

    with pytest.raises(ResourceLimitError, match="did not apply"):
        executor.execute(_request(project, tmp_path / "output"))

    assert container.removed


def test_executor_kills_a_container_when_its_runtime_timeout_expires(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from infrastructure.docker import training_executor

    project = tmp_path / "project"
    project.mkdir()
    container = FakeContainer(remains_running=True)
    executor = DockerTrainingExecutor(docker_client=FakeDockerClient(container))
    clock_values = iter((0.0, 0.5, 1.0, 1.1, 1.2))
    monkeypatch.setattr(training_executor.time, "monotonic", lambda: next(clock_values))

    result = executor.execute(_request(project, tmp_path / "output", timeout_seconds=1))

    assert result.timed_out
    assert container.removed


def _request(
    project: Path,
    output: Path,
    *,
    timeout_seconds: int = 600,
) -> TrainingExecutionRequest:
    return TrainingExecutionRequest(
        run_id=uuid4(),
        image_id="sha256:" + "a" * 64,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=timeout_seconds,
    )
