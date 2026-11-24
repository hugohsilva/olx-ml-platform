from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

DEFAULT_BUILD_REPOSITORIES = ("pypi.org", "files.pythonhosted.org")
DEFAULT_BASE_IMAGE_REGISTRY = "docker.io"
_DOCKER_HUB_REGISTRY_ENDPOINTS = (
    "registry-1.docker.io",
    "auth.docker.io",
    "production.cloudfront.docker.com",
)
DEFAULT_CPU_LIMIT = 1.0
DEFAULT_MEMORY_LIMIT = "1g"
DEFAULT_TIMEOUT_SECONDS = 600
MAX_CPU_LIMIT = 1.0
MAX_MEMORY_BYTES = 1024**3
MAX_TIMEOUT_SECONDS = 1800


class ConfigurationError(ValueError):
    """Raised when local platform configuration is invalid."""


class EgressPolicyError(RuntimeError):
    """Raised when the configured build egress policy cannot be verified."""


@dataclass(frozen=True)
class DockerResourceLimits:
    cpus: float
    memory_limit: str
    build_timeout_seconds: int
    training_timeout_seconds: int


@dataclass(frozen=True)
class PlatformConfig:
    home: Path
    database_path: Path
    artifacts_dir: Path
    logs_dir: Path
    docker: DockerResourceLimits
    base_image_registry: str
    build_allowed_repositories: tuple[str, ...]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PlatformConfig:
        values = os.environ if environ is None else environ
        home = Path(values.get("MLPLATFORM_HOME", "~/.mlplatform")).expanduser().resolve()
        base_registry = _normalize_repository(
            values.get("MLPLATFORM_BASE_IMAGE_REGISTRY", DEFAULT_BASE_IMAGE_REGISTRY)
        )
        extra_repositories = _parse_repositories(
            values.get("MLPLATFORM_EXTRA_BUILD_REPOSITORIES", "")
        )
        allowed_repositories = tuple(
            dict.fromkeys(
                (
                    *DEFAULT_BUILD_REPOSITORIES,
                    base_registry,
                    *_base_image_registry_endpoints(base_registry),
                    *extra_repositories,
                )
            )
        )

        return cls(
            home=home,
            database_path=_resolve_path(
                values.get("MLPLATFORM_DATABASE_PATH", "metadata.sqlite3"), home
            ),
            artifacts_dir=_resolve_path(values.get("MLPLATFORM_ARTIFACTS_DIR", "artifacts"), home),
            logs_dir=_resolve_path(values.get("MLPLATFORM_LOGS_DIR", "logs"), home),
            docker=DockerResourceLimits(
                cpus=_parse_cpu_limit(values.get("MLPLATFORM_DOCKER_CPUS", "1.0")),
                memory_limit=_parse_memory_limit(
                    values.get("MLPLATFORM_DOCKER_MEMORY_LIMIT", "1g")
                ),
                build_timeout_seconds=_parse_timeout(
                    "MLPLATFORM_BUILD_TIMEOUT_SECONDS",
                    values.get("MLPLATFORM_BUILD_TIMEOUT_SECONDS", "600"),
                ),
                training_timeout_seconds=_parse_timeout(
                    "MLPLATFORM_TRAINING_TIMEOUT_SECONDS",
                    values.get("MLPLATFORM_TRAINING_TIMEOUT_SECONDS", "600"),
                ),
            ),
            base_image_registry=base_registry,
            build_allowed_repositories=allowed_repositories,
        )


def _resolve_path(value: str, home: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else home / path).resolve()


def _parse_cpu_limit(value: str) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("MLPLATFORM_DOCKER_CPUS must be a positive number.") from exc
    if not math.isfinite(limit) or limit <= 0 or limit > MAX_CPU_LIMIT:
        raise ConfigurationError("MLPLATFORM_DOCKER_CPUS must be greater than 0 and at most 1.0.")
    return limit


def _parse_memory_limit(value: str) -> str:
    match = re.fullmatch(r"([0-9]+)([mg])", value.strip().lower())
    if match is None:
        raise ConfigurationError(
            "MLPLATFORM_DOCKER_MEMORY_LIMIT must use a positive M or G value, such as 512m or 1g."
        )
    amount = int(match.group(1))
    multiplier = 1024**2 if match.group(2) == "m" else 1024**3
    if amount <= 0 or amount * multiplier > MAX_MEMORY_BYTES:
        raise ConfigurationError(
            "MLPLATFORM_DOCKER_MEMORY_LIMIT must be greater than 0 and at most 1g."
        )
    return f"{amount}{match.group(2)}"


def _parse_timeout(name: str, value: str) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer number of seconds.") from exc
    if seconds < 1 or seconds > MAX_TIMEOUT_SECONDS:
        raise ConfigurationError(f"{name} must be between 1 and {MAX_TIMEOUT_SECONDS} seconds.")
    return seconds


def _parse_repositories(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    repositories = [repository.strip() for repository in value.split(",")]
    if any(not repository for repository in repositories):
        raise ConfigurationError("Build repositories must be comma-separated exact hostnames.")
    return tuple(dict.fromkeys(_normalize_repository(repository) for repository in repositories))


def _base_image_registry_endpoints(registry: str) -> tuple[str, ...]:
    """Return the exact Docker Hub hosts required to fetch a public base image."""

    return _DOCKER_HUB_REGISTRY_ENDPOINTS if registry == "docker.io" else ()


def _normalize_repository(value: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ConfigurationError("Build repositories must be exact hostnames.")
    if "*" in value or "/" in value or "\\" in value or "://" in value:
        raise ConfigurationError("Build repositories must be exact hostnames without wildcards.")

    parsed = urlsplit(f"//{value}")
    if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ConfigurationError("Build repositories must be exact hostnames without URL paths.")
    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"Invalid build repository hostname: {value}") from exc

    if hostname is None or not _is_valid_hostname(hostname):
        raise ConfigurationError(f"Invalid build repository hostname: {value}")
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError(f"Invalid build repository port: {value}")
    return hostname.lower() + (f":{port}" if port is not None else "")


def _is_valid_hostname(hostname: str) -> bool:
    if len(hostname) > 253:
        return False
    return all(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) is not None
        for label in hostname.rstrip(".").split(".")
    )
