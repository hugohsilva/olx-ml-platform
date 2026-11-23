from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from infrastructure.config import (
    ConfigurationError,
    EgressPolicyError,
    PlatformConfig,
    verify_build_egress_policy,
)


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


def test_egress_preflight_fails_closed_when_no_checker_is_configured() -> None:
    config = PlatformConfig.from_env({})

    with pytest.raises(EgressPolicyError, match="not verifiable"):
        verify_build_egress_policy(config)


def test_egress_preflight_accepts_only_an_exact_enforced_allowlist() -> None:
    config = replace(
        PlatformConfig.from_env({}),
        egress_policy_check=("policy-check",),
    )
    expected_report = {
        "enforced": True,
        "policy_id": "local-firewall-policy-1",
        "allowed_repositories": list(config.build_allowed_repositories),
    }

    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert arguments[:3] == [
            "policy-check",
            "--verify-build-egress",
            "--json",
        ]
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(arguments, 0, json.dumps(expected_report), "")

    assert verify_build_egress_policy(config, runner=runner) == "local-firewall-policy-1"


def test_egress_preflight_rejects_a_mismatched_policy_report() -> None:
    config = replace(
        PlatformConfig.from_env({}),
        egress_policy_check=("policy-check",),
    )
    report = {
        "enforced": True,
        "policy_id": "local-firewall-policy-1",
        "allowed_repositories": ["pypi.org"],
    }

    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, json.dumps(report), "")

    with pytest.raises(EgressPolicyError, match="does not exactly match"):
        verify_build_egress_policy(config, runner=runner)
