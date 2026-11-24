from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from application.ports import ImageBuildRequest
from infrastructure.config import PlatformConfig
from infrastructure.docker.egress_policy import DockerBuildEgressPolicy

_PYTHON_IMAGE = "docker.io/library/python:3.11.9-slim"


def test_docker_build_egress_policy_allows_only_verified_proxy_traffic(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_python_image(client)
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path / "home")})
    request = ImageBuildRequest(
        run_id=uuid4(),
        project_directory=tmp_path,
        requirements_file="requirements.txt",
        python_version="3.11.9",
        base_image_registry=config.base_image_registry,
        allowed_repositories=config.build_allowed_repositories,
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=30,
    )
    policy = DockerBuildEgressPolicy.for_run(
        config,
        request,
        _PYTHON_IMAGE,
        client,
    )

    try:
        policy.start()

        assert policy.proxy_url.startswith("http://")
        assert policy.policy_id.startswith("docker-internal-proxy:")
        assert policy._internal_network.attrs["Internal"] is True
        assert policy._outbound_network.attrs["Internal"] is False

        proxy = policy._proxy_container
        assert proxy is not None
        proxy.reload()
        host_config = proxy.attrs["HostConfig"]
        assert host_config["Privileged"] is False
        assert host_config["CapDrop"] == ["ALL"]
        assert "no-new-privileges:true" in host_config["SecurityOpt"]
        assert host_config["ReadonlyRootfs"] is True
        assert host_config["Binds"] is None
        assert "3128/tcp" not in (host_config.get("PortBindings") or {})
    finally:
        policy.close()


def test_docker_training_profile_denies_root_host_writes_and_network(tmp_path: Path) -> None:
    client = _get_docker_client()
    _require_python_image(client)
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    (project / "train.py").write_text(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=2)\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(20)\n"
        "try:\n"
        "    open('/workspace/forbidden', 'w').close()\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(21)\n"
        "open('/output/security-ok', 'w').close()\n",
        encoding="utf-8",
    )

    from datetime import datetime, timezone
    from uuid import uuid4

    from application.ports import TrainingExecutionRequest
    from infrastructure.docker.training_executor import DockerTrainingExecutor

    request = TrainingExecutionRequest(
        run_id=uuid4(),
        image_id=_PYTHON_IMAGE,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=30,
        network_disabled=True,
    )
    result = DockerTrainingExecutor(docker_client=client).execute(request)

    assert result.exit_code == 0
    assert not result.timed_out
    assert not result.out_of_memory
    assert (output / "security-ok").is_file()


def _get_docker_client():
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


def _require_python_image(client) -> None:
    try:
        client.images.get(_PYTHON_IMAGE)
    except ImageNotFound:
        pytest.skip(f"Local integration-test image not available: {_PYTHON_IMAGE}")
