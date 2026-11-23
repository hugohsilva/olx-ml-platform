from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias
from uuid import UUID

JsonScalar: TypeAlias = str | int | float | bool | None
NamedValue: TypeAlias = tuple[str, JsonScalar]
MetricValue: TypeAlias = str | int | float
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MAX_TIMEOUT_SECONDS = 1800
MAX_CPU_LIMIT = 1.0
MAX_MEMORY_BYTES = 1024**3


class RunState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FailureReason(StrEnum):
    VALIDATION = "validation"
    IMAGE_BUILD_ERROR = "image_build_error"
    IMAGE_BUILD_TIMEOUT = "image_build_timeout"
    TRAINING_ERROR = "training_error"
    TIMEOUT = "timeout"
    OOM_KILLED = "oom_killed"
    RESOURCE_LIMIT = "resource_limit"
    CONTAINER_ERROR = "container_error"
    MISSING_OUTPUT = "missing_output"


class ServingStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    INVALID = "invalid"


class LogStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"
    APPLICATION = "application"


@dataclass(frozen=True, slots=True)
class TrainingProject:
    project_id: str
    entrypoint: str
    dependencies: str
    data_location: Path
    contract_version: str
    git_commit: str | None = None
    code_hash: str | None = None
    dataset_identity: str | None = None
    dataset_hash: str | None = None
    config_hash: str | None = None
    python_version: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.project_id, "project_id")
        _require_nonempty(self.entrypoint, "entrypoint")
        _require_nonempty(self.dependencies, "dependencies")
        _require_nonempty(self.contract_version, "contract_version")
        if not isinstance(self.data_location, Path):
            raise ValueError("data_location must be a Path.")
        if self.git_commit is not None:
            _require_nonempty(self.git_commit, "git_commit")
        if self.dataset_identity is not None:
            _require_nonempty(self.dataset_identity, "dataset_identity")
        if self.python_version is not None:
            _require_nonempty(self.python_version, "python_version")
        _validate_optional_hash(self.code_hash, "code_hash")
        _validate_optional_hash(self.dataset_hash, "dataset_hash")
        _validate_optional_hash(self.config_hash, "config_hash")


@dataclass(frozen=True, slots=True)
class Metric:
    run_id: UUID
    name: str
    value: MetricValue
    unit: str | None = None

    def __post_init__(self) -> None:
        _require_uuid(self.run_id, "run_id")
        _require_nonempty(self.name, "name")
        _validate_metric_value(self.value)
        if self.unit is not None:
            _require_nonempty(self.unit, "unit")


@dataclass(frozen=True, slots=True)
class Run:
    run_id: UUID
    project_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: RunState = RunState.CREATED
    git_commit: str | None = None
    code_hash: str | None = None
    dataset_identity: str | None = None
    dataset_hash: str | None = None
    config_hash: str | None = None
    dependency_versions: tuple[tuple[str, str], ...] = ()
    python_version: str | None = None
    docker_image: str | None = None
    build_allowed_repositories: tuple[str, ...] = ()
    image_build_started_at: datetime | None = None
    image_build_finished_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    build_timeout_seconds: int = 600
    timeout_seconds: int = 600
    max_cpus: float = 1.0
    memory_limit: str = "1g"
    parameters: tuple[NamedValue, ...] = ()
    metrics: tuple[Metric, ...] = ()
    log_location: str | None = None
    image_build_log_location: str | None = None
    failure_reason: FailureReason | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        _require_uuid(self.run_id, "run_id")
        _require_nonempty(self.project_id, "project_id")
        _validate_datetime(self.created_at, "created_at")
        for field_name in (
            "image_build_started_at",
            "image_build_finished_at",
            "started_at",
            "finished_at",
        ):
            _validate_optional_datetime(getattr(self, field_name), field_name)

        if not isinstance(self.state, RunState):
            raise ValueError("state must be a RunState.")
        if self.failure_reason is not None and not isinstance(self.failure_reason, FailureReason):
            raise ValueError("failure_reason must be a FailureReason.")
        if self.state is RunState.FAILED and self.failure_reason is None:
            raise ValueError("A failed run requires a failure_reason.")
        if self.state is not RunState.FAILED and self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for a failed run.")
        if self.failure_message is not None:
            _require_nonempty(self.failure_message, "failure_message")
            if self.state is not RunState.FAILED:
                raise ValueError("failure_message is only valid for a failed run.")
        if self.state in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}:
            if self.finished_at is None:
                raise ValueError("A terminal run requires finished_at.")
        elif self.finished_at is not None:
            raise ValueError("A non-terminal run cannot have finished_at.")

        _validate_optional_hash(self.code_hash, "code_hash")
        _validate_optional_hash(self.dataset_hash, "dataset_hash")
        _validate_optional_hash(self.config_hash, "config_hash")
        if self.git_commit is not None:
            _require_nonempty(self.git_commit, "git_commit")
        if self.dataset_identity is not None:
            _require_nonempty(self.dataset_identity, "dataset_identity")
        if self.python_version is not None:
            _require_nonempty(self.python_version, "python_version")
        if self.docker_image is not None:
            _require_nonempty(self.docker_image, "docker_image")
        if self.log_location is not None:
            _require_nonempty(self.log_location, "log_location")
        if self.image_build_log_location is not None:
            _require_nonempty(self.image_build_log_location, "image_build_log_location")

        _validate_timeout(self.build_timeout_seconds, "build_timeout_seconds")
        _validate_timeout(self.timeout_seconds, "timeout_seconds")
        if (
            isinstance(self.max_cpus, bool)
            or not isinstance(self.max_cpus, (int, float))
            or not math.isfinite(self.max_cpus)
            or self.max_cpus <= 0
            or self.max_cpus > MAX_CPU_LIMIT
        ):
            raise ValueError("max_cpus must be greater than 0 and at most 1.0.")
        _validate_memory_limit(self.memory_limit)

        object.__setattr__(
            self,
            "dependency_versions",
            _validate_pairs(self.dependency_versions, "dependency_versions"),
        )
        object.__setattr__(
            self,
            "parameters",
            _validate_named_values(self.parameters, "parameters"),
        )
        repositories = tuple(self.build_allowed_repositories)
        if any(
            not isinstance(repository, str) or not repository.strip() for repository in repositories
        ):
            raise ValueError("build_allowed_repositories must contain non-empty hostnames.")
        if len(set(repositories)) != len(repositories):
            raise ValueError("build_allowed_repositories cannot contain duplicates.")
        object.__setattr__(self, "build_allowed_repositories", repositories)

        metrics = tuple(self.metrics)
        if any(
            not isinstance(metric, Metric) or metric.run_id != self.run_id for metric in metrics
        ):
            raise ValueError("Every metric must belong to this run.")
        if len({metric.name for metric in metrics}) != len(metrics):
            raise ValueError("Metric names must be unique within a run.")
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    run_id: UUID
    artifact_name: str
    artifact_location: str
    size_bytes: int
    checksum: str
    created_at: datetime
    serving_status: ServingStatus

    def __post_init__(self) -> None:
        _require_uuid(self.run_id, "run_id")
        _require_nonempty(self.artifact_name, "artifact_name")
        _require_nonempty(self.artifact_location, "artifact_location")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer.")
        _validate_hash(self.checksum, "checksum")
        _validate_datetime(self.created_at, "created_at")
        if not isinstance(self.serving_status, ServingStatus):
            raise ValueError("serving_status must be a ServingStatus.")


@dataclass(frozen=True, slots=True)
class LogRecord:
    run_id: UUID
    sequence: int
    timestamp: datetime
    stream: LogStream
    message: str

    def __post_init__(self) -> None:
        _require_uuid(self.run_id, "run_id")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer.")
        _validate_datetime(self.timestamp, "timestamp")
        if not isinstance(self.stream, LogStream):
            raise ValueError("stream must be a LogStream.")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string.")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    instances: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        try:
            instances = tuple(tuple(instance) for instance in self.instances)
        except TypeError as exc:
            raise ValueError("instances must be a sequence of numeric rows.") from exc
        if not instances or any(not instance for instance in instances):
            raise ValueError("instances must contain at least one non-empty row.")
        for instance in instances:
            for value in instance:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("instances must contain finite numeric values.")
        object.__setattr__(self, "instances", instances)


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    run_id: UUID
    predictions: tuple[JsonScalar, ...]

    def __post_init__(self) -> None:
        _require_uuid(self.run_id, "run_id")
        predictions = tuple(self.predictions)
        for prediction in predictions:
            if isinstance(prediction, float) and not math.isfinite(prediction):
                raise ValueError("predictions must contain finite values.")
            if not isinstance(prediction, (str, int, float, bool, type(None))):
                raise ValueError("predictions must contain JSON scalar values.")
        object.__setattr__(self, "predictions", predictions)


def _require_uuid(value: UUID, field_name: str) -> None:
    if not isinstance(value, UUID):
        raise ValueError(f"{field_name} must be a UUID.")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")


def _validate_optional_hash(value: str | None, field_name: str) -> None:
    if value is not None:
        _validate_hash(value, field_name)


def _validate_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime.")


def _validate_optional_datetime(value: datetime | None, field_name: str) -> None:
    if value is not None:
        _validate_datetime(value, field_name)


def _validate_timeout(value: int, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(f"{field_name} must be between 1 and {MAX_TIMEOUT_SECONDS} seconds.")


def _validate_memory_limit(value: str) -> None:
    match = re.fullmatch(r"([0-9]+)([mg])", value.lower()) if isinstance(value, str) else None
    if match is None:
        raise ValueError("memory_limit must use M or G units, such as 512m or 1g.")
    amount = int(match.group(1))
    multiplier = 1024**2 if match.group(2) == "m" else 1024**3
    if amount <= 0 or amount * multiplier > MAX_MEMORY_BYTES:
        raise ValueError("memory_limit must be greater than 0 and at most 1g.")


def _validate_pairs(
    pairs: tuple[tuple[str, str], ...], field_name: str
) -> tuple[tuple[str, str], ...]:
    normalized = _normalize_pairs(pairs, field_name)
    if any(
        len(pair) != 2
        or not isinstance(pair[0], str)
        or not pair[0].strip()
        or not isinstance(pair[1], str)
        or not pair[1].strip()
        for pair in normalized
    ):
        raise ValueError(f"{field_name} must contain non-empty name-version pairs.")
    if len({pair[0] for pair in normalized}) != len(normalized):
        raise ValueError(f"{field_name} names must be unique.")
    return normalized


def _validate_named_values(
    pairs: tuple[NamedValue, ...], field_name: str
) -> tuple[NamedValue, ...]:
    normalized = _normalize_pairs(pairs, field_name)
    if any(
        len(pair) != 2 or not isinstance(pair[0], str) or not pair[0].strip() for pair in normalized
    ):
        raise ValueError(f"{field_name} must contain name-value pairs.")
    if len({pair[0] for pair in normalized}) != len(normalized):
        raise ValueError(f"{field_name} names must be unique.")
    for _, value in normalized:
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"{field_name} values must be JSON scalars.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field_name} values must be finite.")
    return normalized


def _validate_metric_value(value: MetricValue) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Metric value must be a number or string.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Metric value must be finite.")


def _normalize_pairs(
    pairs: tuple[tuple[object, object], ...], field_name: str
) -> tuple[tuple[object, object], ...]:
    if not isinstance(pairs, (tuple, list)) or any(
        not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs
    ):
        raise ValueError(f"{field_name} must contain two-item pairs.")
    return tuple((pair[0], pair[1]) for pair in pairs)
