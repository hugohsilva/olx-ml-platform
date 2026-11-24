from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from application.ports import ImageBuildRequest, TrainingExecutionRequest
from infrastructure.config import PlatformConfig
from infrastructure.docker.image_builder import DockerImageBuilder
from infrastructure.docker.training_executor import DockerTrainingExecutor

_PYTHON_IMAGE = "python:3.11.9-slim"


class InspectingImageBuilder(DockerImageBuilder):
    observed_host_config: dict[str, object]

    def _verify_builder_profile(self, builder_name, request, policy) -> None:
        super()._verify_builder_profile(builder_name, request, policy)
        container = self._docker().containers.get(f"buildx_buildkit_{builder_name}0")
        self.observed_host_config = container.attrs["HostConfig"]


def test_buildkit_container_enforces_cpu_and_memory_limits(tmp_path: Path) -> None:
    client = _get_docker_client()
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path / "home")})
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("", encoding="utf-8")
    (project / "train.py").write_text("pass\n", encoding="utf-8")
    request = ImageBuildRequest(
        run_id=uuid4(),
        project_directory=project,
        requirements_file="requirements.txt",
        python_version=platform.python_version(),
        base_image_registry=config.base_image_registry,
        allowed_repositories=config.build_allowed_repositories,
        cpus=0.5,
        memory_limit="512m",
        timeout_seconds=300,
    )
    builder = InspectingImageBuilder(config, docker_client=client)
    image_id = None

    try:
        result = builder.build(request)
        image_id = result.image_id
        host_config = builder.observed_host_config

        assert host_config["CpuPeriod"] == 100_000
        assert host_config["CpuQuota"] == 50_000
        assert host_config["Memory"] == 512 * 1024**2
    finally:
        if image_id is not None:
            client.images.remove(image_id, force=True)


def test_training_container_enforces_cpu_and_memory_limits(tmp_path: Path) -> None:
    client = _get_docker_client()
    try:
        client.images.get(_PYTHON_IMAGE)
    except ImageNotFound:
        pytest.skip(f"Local integration-test image not available: {_PYTHON_IMAGE}")

    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    (project / "train.py").write_text(
        """\
from pathlib import Path

cgroup = Path("/sys/fs/cgroup")
if (cgroup / "cpu.max").is_file():
    quota, period = (cgroup / "cpu.max").read_text().split()
    memory = (cgroup / "memory.max").read_text().strip()
else:
    quota = (cgroup / "cpu/cpu.cfs_quota_us").read_text().strip()
    period = (cgroup / "cpu/cpu.cfs_period_us").read_text().strip()
    memory = (cgroup / "memory/memory.limit_in_bytes").read_text().strip()
Path("/output/effective-limits").write_text(f"{quota} {period} {memory}")
""",
        encoding="utf-8",
    )
    request = TrainingExecutionRequest(
        run_id=uuid4(),
        image_id=_PYTHON_IMAGE,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=0.5,
        memory_limit="512m",
        timeout_seconds=30,
    )

    result = DockerTrainingExecutor(docker_client=client).execute(request)

    quota, period, memory = map(int, (output / "effective-limits").read_text().split())
    assert result.exit_code == 0
    assert quota / period == pytest.approx(0.5)
    assert memory == 512 * 1024**2


def _get_docker_client():
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")
