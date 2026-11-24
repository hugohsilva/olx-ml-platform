from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from application.ports import TrainingExecutionRequest
from infrastructure.docker.container_config import build_training_container_config


def create_request(
    project_directory: Path,
    output_directory: Path,
    **overrides: object,
) -> TrainingExecutionRequest:
    values: dict[str, object] = {
        "run_id": uuid4(),
        "image_id": "sha256:" + "a" * 64,
        "project_directory": project_directory,
        "entrypoint": "train.py",
        "output_directory": output_directory,
        "started_at": datetime.now(timezone.utc),
        "cpus": 1.0,
        "memory_limit": "1g",
        "timeout_seconds": 600,
        "network_disabled": True,
    }
    values.update(overrides)
    return TrainingExecutionRequest(**values)  # type: ignore[arg-type]


def test_training_container_config_applies_isolation_and_mount_boundaries(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()

    config = build_training_container_config(create_request(project, output))

    assert config["network_mode"] == "none"
    assert config["privileged"] is False
    assert config["cap_drop"] == ["ALL"]
    assert config["security_opt"] == ["no-new-privileges:true"]
    assert config["read_only"] is True
    assert config["user"] != "0:0"
    assert config["nano_cpus"] == 1_000_000_000
    assert config["mem_limit"] == 1024**3
    assert config["memswap_limit"] == 1024**3
    assert config["volumes"] == {
        str(project.resolve()): {"bind": "/workspace", "mode": "ro"},
        str(output.resolve()): {"bind": "/output", "mode": "rw"},
    }
    assert config["tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=64m"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"network_disabled": False}, "without network"),
        ({"entrypoint": "../outside.py"}, "inside the project"),
        ({"cpus": 0.0}, "cpus"),
        ({"timeout_seconds": 1801}, "timeout_seconds"),
        ({"memory_limit": "2g"}, "at most 1g"),
    ],
)
def test_training_container_config_rejects_unsafe_or_invalid_options(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()

    with pytest.raises(ValueError, match=message):
        build_training_container_config(create_request(project, output, **overrides))
