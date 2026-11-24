from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from application.ports import TrainingExecutionRequest
from infrastructure.config import ConfigurationError, PlatformConfig
from infrastructure.docker.container_config import build_training_container_config


@pytest.mark.parametrize(
    "values",
    [
        {"MLPLATFORM_DOCKER_CPUS": "0"},
        {"MLPLATFORM_DOCKER_CPUS": "2"},
        {"MLPLATFORM_DOCKER_MEMORY_LIMIT": "2g"},
        {"MLPLATFORM_BUILD_TIMEOUT_SECONDS": "0"},
        {"MLPLATFORM_BUILD_TIMEOUT_SECONDS": "1801"},
        {"MLPLATFORM_TRAINING_TIMEOUT_SECONDS": "bad"},
    ],
)
def test_platform_configuration_rejects_invalid_resource_limits(
    tmp_path: Path,
    values: dict[str, str],
) -> None:
    with pytest.raises(ConfigurationError):
        PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path), **values})


@pytest.mark.parametrize(
    ("cpus", "memory_limit", "timeout_seconds"),
    [
        (0.0, "1g", 600),
        (1.1, "1g", 600),
        (1.0, "2g", 600),
        (1.0, "1g", 0),
        (1.0, "1g", 1801),
    ],
)
def test_training_container_rejects_limits_before_container_creation(
    tmp_path: Path,
    cpus: float,
    memory_limit: str,
    timeout_seconds: int,
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    request = TrainingExecutionRequest(
        run_id=uuid4(),
        image_id="sha256:" + "a" * 64,
        project_directory=project,
        entrypoint="train.py",
        output_directory=output,
        started_at=datetime.now(timezone.utc),
        cpus=cpus,
        memory_limit=memory_limit,
        timeout_seconds=timeout_seconds,
    )

    with pytest.raises(ValueError):
        build_training_container_config(request)
