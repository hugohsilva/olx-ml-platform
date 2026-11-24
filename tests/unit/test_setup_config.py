from __future__ import annotations

import pytest

from infrastructure.config import ConfigurationError, PlatformConfig


def test_default_config_has_local_paths_resource_limits_and_exact_repositories() -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": "/tmp/mlplatform-test"})

    assert config.database_path == config.home / "metadata.sqlite3"
    assert config.artifacts_dir == config.home / "artifacts"
    assert config.logs_dir == config.home / "logs"
    assert config.docker.cpus == 1.0
    assert config.docker.memory_limit == "1g"
    assert config.docker.build_timeout_seconds == 600
    assert config.docker.training_timeout_seconds == 600
    assert config.build_allowed_repositories == (
        "pypi.org",
        "files.pythonhosted.org",
        "docker.io",
        "registry-1.docker.io",
        "auth.docker.io",
        "production.cloudfront.docker.com",
    )


def test_config_accepts_lower_resource_limits_and_additional_exact_repository() -> None:
    config = PlatformConfig.from_env(
        {
            "MLPLATFORM_DOCKER_CPUS": "0.5",
            "MLPLATFORM_DOCKER_MEMORY_LIMIT": "512m",
            "MLPLATFORM_BUILD_TIMEOUT_SECONDS": "300",
            "MLPLATFORM_TRAINING_TIMEOUT_SECONDS": "900",
            "MLPLATFORM_BASE_IMAGE_REGISTRY": "registry.example.com:5000",
            "MLPLATFORM_EXTRA_BUILD_REPOSITORIES": "packages.example.com",
        }
    )

    assert config.docker.cpus == 0.5
    assert config.docker.memory_limit == "512m"
    assert config.docker.build_timeout_seconds == 300
    assert config.docker.training_timeout_seconds == 900
    assert config.build_allowed_repositories == (
        "pypi.org",
        "files.pythonhosted.org",
        "registry.example.com:5000",
        "packages.example.com",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLPLATFORM_DOCKER_CPUS", "0"),
        ("MLPLATFORM_DOCKER_CPUS", "1.1"),
        ("MLPLATFORM_DOCKER_CPUS", "nan"),
        ("MLPLATFORM_DOCKER_MEMORY_LIMIT", "2g"),
        ("MLPLATFORM_DOCKER_MEMORY_LIMIT", "0m"),
        ("MLPLATFORM_BUILD_TIMEOUT_SECONDS", "1801"),
        ("MLPLATFORM_TRAINING_TIMEOUT_SECONDS", "0"),
        ("MLPLATFORM_EXTRA_BUILD_REPOSITORIES", "*.example.com"),
    ],
)
def test_config_rejects_invalid_limits_and_repository_wildcards(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        PlatformConfig.from_env({name: value})
