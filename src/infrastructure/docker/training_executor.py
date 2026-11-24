from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import docker
from docker.errors import DockerException

from application.errors import ApplicationError, ContainerExecutionError, ResourceLimitError
from application.ports import TrainingExecutionRequest, TrainingExecutionResult
from domain.models import LogRecord, LogStream
from infrastructure.docker.container_config import (
    build_training_container_config,
    parse_memory_limit,
)

_POLL_INTERVAL_SECONDS = 0.1
_STOP_CONFIRM_SECONDS = 5.0
_DOCKER_TIMESTAMP_PATTERN = re.compile(r"^(\S+)\s(.*)$")


class DockerTrainingExecutor:
    def __init__(self, *, docker_client: Any | None = None) -> None:
        self._docker_client = docker_client

    def execute(self, request: TrainingExecutionRequest) -> TrainingExecutionResult:
        if request.started_at.tzinfo is None or request.started_at.utcoffset() is None:
            raise ContainerExecutionError(
                "Training start timestamp must include a timezone.",
                run_id=request.run_id,
            )
        project_directory = request.project_directory
        output_directory = request.output_directory
        if (
            project_directory.is_symlink()
            or not project_directory.is_dir()
            or output_directory.is_symlink()
        ):
            raise ContainerExecutionError(
                "Training input or output directory is missing or invalid.",
                run_id=request.run_id,
            )
        try:
            project_directory = project_directory.resolve(strict=True)
            output_directory.mkdir(parents=True, exist_ok=True)
            output_directory = output_directory.resolve(strict=True)
        except OSError as exc:
            raise ContainerExecutionError(
                "Training input or output directory is unavailable.",
                run_id=request.run_id,
            ) from exc

        try:
            config = build_training_container_config(request)
        except (TypeError, ValueError) as exc:
            raise ContainerExecutionError(str(exc), run_id=request.run_id) from exc

        container = None
        primary_error: ApplicationError | None = None
        try:
            container = self._docker().containers.create(
                image=request.image_id,
                name=f"mlplatform-training-{request.run_id.hex}",
                labels={"mlplatform.run_id": str(request.run_id)},
                **config,
            )
            container.reload()
            self._verify_resource_limits(container, request)
            container.start()
            timed_out = self._wait_for_completion(container, request)
            container.reload()
            state = container.attrs["State"]
            logs = _collect_logs(container, request)
            exit_code = state.get("ExitCode")
            if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                raise ContainerExecutionError(
                    "Docker returned an invalid training container exit status.",
                    run_id=request.run_id,
                )
            return TrainingExecutionResult(
                exit_code=exit_code,
                logs=logs,
                timed_out=timed_out,
                out_of_memory=state.get("OOMKilled") is True,
            )
        except DockerException as exc:
            primary_error = ContainerExecutionError(
                "Docker could not create or inspect the training container.",
                run_id=request.run_id,
            )
            raise primary_error from exc
        except ApplicationError as exc:
            primary_error = exc
            raise
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except DockerException as cleanup_error:
                    if primary_error is None:
                        raise ContainerExecutionError(
                            "Training container could not be removed.",
                            run_id=request.run_id,
                        ) from cleanup_error
                    primary_error.add_note(
                        f"Training container cleanup also failed: {cleanup_error}"
                    )

    def _wait_for_completion(
        self,
        container: Any,
        request: TrainingExecutionRequest,
    ) -> bool:
        deadline = time.monotonic() + request.timeout_seconds
        while True:
            container.reload()
            state = container.attrs.get("State", {})
            if state.get("Status") in {"exited", "dead"}:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_after_timeout(container, request)
                return True
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def _verify_resource_limits(
        self,
        container: Any,
        request: TrainingExecutionRequest,
    ) -> None:
        host_config = container.attrs.get("HostConfig", {})
        expected_memory = parse_memory_limit(request.memory_limit)
        expected_nano_cpus = round(request.cpus * 1_000_000_000)
        if (
            host_config.get("NanoCpus") != expected_nano_cpus
            or host_config.get("Memory") != expected_memory
            or host_config.get("MemorySwap") != expected_memory
        ):
            raise ResourceLimitError(
                "Docker did not apply the configured CPU and memory limits.",
                run_id=request.run_id,
            )

    def _kill_after_timeout(self, container: Any, request: TrainingExecutionRequest) -> None:
        try:
            container.kill()
        except DockerException as exc:
            try:
                container.reload()
            except DockerException as reload_error:
                raise ContainerExecutionError(
                    "Timed-out training container could not be stopped or inspected.",
                    run_id=request.run_id,
                ) from reload_error
            if container.attrs.get("State", {}).get("Status") not in {"exited", "dead"}:
                raise ContainerExecutionError(
                    "Timed-out training container could not be stopped.",
                    run_id=request.run_id,
                ) from exc

        deadline = time.monotonic() + _STOP_CONFIRM_SECONDS
        while time.monotonic() < deadline:
            container.reload()
            if container.attrs.get("State", {}).get("Status") in {"exited", "dead"}:
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
        raise ContainerExecutionError(
            "Timed-out training container did not stop after termination was requested.",
            run_id=request.run_id,
        )

    def _docker(self) -> Any:
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except DockerException as exc:
                raise ContainerExecutionError("The Docker daemon is unavailable.") from exc
        return self._docker_client


def _collect_logs(container: Any, request: TrainingExecutionRequest) -> tuple[LogRecord, ...]:
    records: list[LogRecord] = []
    for stream_name, stream in (
        (LogStream.STDOUT, "stdout"),
        (LogStream.STDERR, "stderr"),
    ):
        content = container.logs(
            stdout=stream == "stdout",
            stderr=stream == "stderr",
            timestamps=True,
        )
        if isinstance(content, str):
            lines = content.splitlines()
        else:
            lines = content.decode("utf-8", errors="replace").splitlines()
        for line in lines:
            match = _DOCKER_TIMESTAMP_PATTERN.match(line)
            if match is None:
                timestamp = request.started_at
                message = line
            else:
                timestamp = _parse_timestamp(match.group(1), request.started_at)
                message = match.group(2)
            records.append(
                LogRecord(
                    run_id=request.run_id,
                    sequence=len(records),
                    timestamp=timestamp,
                    stream=stream_name,
                    message=message,
                )
            )
    return tuple(records)


def _parse_timestamp(value: str, fallback: datetime) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return fallback
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return fallback
    return timestamp
