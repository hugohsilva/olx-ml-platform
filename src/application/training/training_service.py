from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from application.errors import (
    ApplicationError,
    ImageBuildError,
    ImageBuildTimeout,
    OutOfMemoryError,
    TrainingError,
    TrainingTimeout,
    failure_reason_for,
)
from application.ports import (
    ArtifactStore,
    ExecutionRepository,
    ImageBuilder,
    ImageBuildRequest,
    LogRepository,
    MetricsRepository,
    TrainingExecutionRequest,
    TrainingExecutor,
)
from application.training.artifact_ingestor import ArtifactIngestor
from application.training.log_collector import LogCollector
from application.training.project_validator import validate_training_project
from application.training.training_outputs import parse_training_outputs
from domain.executions.state_machine import transition_run
from domain.models import NamedValue, Run, RunState
from infrastructure.config import PlatformConfig


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    cpus: float = 1.0
    memory_limit: str = "1g"
    build_timeout_seconds: int = 600
    timeout_seconds: int = 600
    parameters: tuple[NamedValue, ...] = ()


class TrainingService:
    def __init__(
        self,
        *,
        execution_repository: ExecutionRepository,
        metrics_repository: MetricsRepository,
        log_repository: LogRepository,
        artifact_store: ArtifactStore,
        image_builder: ImageBuilder,
        training_executor: TrainingExecutor,
        config: PlatformConfig,
    ) -> None:
        self._execution_repository = execution_repository
        self._metrics_repository = metrics_repository
        self._artifact_store = artifact_store
        self._image_builder = image_builder
        self._training_executor = training_executor
        self._config = config
        self._log_collector = LogCollector(artifact_store, log_repository)
        self._artifact_ingestor = ArtifactIngestor(artifact_store)

    def train(
        self,
        project_directory: Path,
        *,
        options: TrainingOptions | None = None,
        on_run_created: Callable[[Run], None] | None = None,
    ) -> Run:
        effective_options = options or TrainingOptions(
            cpus=self._config.docker.cpus,
            memory_limit=self._config.docker.memory_limit,
            build_timeout_seconds=self._config.docker.build_timeout_seconds,
            timeout_seconds=self._config.docker.training_timeout_seconds,
        )
        project_id = project_directory.name.strip() or "training-project"
        run = Run(
            run_id=uuid4(),
            project_id=project_id,
            build_timeout_seconds=effective_options.build_timeout_seconds,
            timeout_seconds=effective_options.timeout_seconds,
            max_cpus=effective_options.cpus,
            memory_limit=effective_options.memory_limit,
            parameters=effective_options.parameters,
            build_allowed_repositories=self._config.build_allowed_repositories,
        )
        self._execution_repository.save_run(run)

        try:
            self._artifact_store.ensure_run_directories(run.run_id)
            if on_run_created is not None:
                on_run_created(run)
            effective_config = self._effective_config(effective_options)
            project = validate_training_project(
                project_directory,
                effective_config=effective_config,
            )
            run = replace(
                run,
                code_hash=project.code_hash,
                dataset_identity=project.dataset_identity,
                dataset_hash=project.dataset_hash,
                config_hash=project.config_hash,
                python_version=project.python_version,
            )
            self._execution_repository.save_run(run)

            run, next_log_sequence = self._build_image(
                run,
                project_directory,
                project.dependencies,
            )
            run = self._execute_training(
                run,
                project_directory,
                project.entrypoint,
                starting_log_sequence=next_log_sequence,
            )
            outputs = parse_training_outputs(
                self._output_directory(run.run_id),
                run.run_id,
            )
            self._artifact_ingestor.ingest(run.run_id, self._output_directory(run.run_id))
            self._metrics_repository.save_metrics(run.run_id, outputs.metrics)
            succeeded = transition_run(
                replace(
                    run,
                    parameters=outputs.parameters,
                    metrics=outputs.metrics,
                ),
                RunState.SUCCEEDED,
            )
            self._execution_repository.save_run(succeeded)
            return succeeded
        except ApplicationError as error:
            self._persist_failure(run, error)
            raise
        except OSError as exc:
            error = TrainingError("A local filesystem operation failed during the training run.")
            self._persist_failure(run, error)
            raise error from exc

    def _persist_failure(self, run: Run, error: ApplicationError) -> None:
        current_run = self._execution_repository.get_run(run.run_id) or run
        failure_message = str(error)
        if isinstance(error, (ImageBuildError, ImageBuildTimeout)) and error.logs:
            try:
                self._log_collector.collect_messages(
                    current_run.run_id,
                    error.logs,
                    file_name="image-build.log",
                )
            except OSError:
                failure_message += " Build logs could not be persisted."
            else:
                current_run = replace(
                    current_run,
                    image_build_log_location=str(
                        self._artifact_store.log_directory(current_run.run_id) / "image-build.log"
                    ),
                )
        failed = transition_run(
            current_run,
            RunState.FAILED,
            failure_reason=failure_reason_for(error),
            failure_message=failure_message,
        )
        error.run_id = current_run.run_id
        self._execution_repository.save_run(failed)

    def _build_image(
        self,
        run: Run,
        project_directory: Path,
        requirements_file: str,
    ) -> tuple[Run, int]:
        build_started_at = datetime.now(timezone.utc)
        running = replace(
            transition_run(run, RunState.RUNNING, at=build_started_at),
            image_build_started_at=build_started_at,
        )
        self._execution_repository.save_run(running)
        request = ImageBuildRequest(
            run_id=running.run_id,
            project_directory=project_directory.resolve(strict=True),
            requirements_file=requirements_file,
            python_version=run.python_version or "3.11",
            base_image_registry=self._config.base_image_registry,
            allowed_repositories=self._config.build_allowed_repositories,
            cpus=running.max_cpus,
            memory_limit=running.memory_limit,
            timeout_seconds=running.build_timeout_seconds,
        )
        try:
            result = self._image_builder.build(request)
        except ApplicationError:
            failed_build = replace(running, image_build_finished_at=datetime.now(timezone.utc))
            self._execution_repository.save_run(failed_build)
            raise

        build_logs = self._log_collector.collect_messages(
            running.run_id,
            result.logs,
            file_name="image-build.log",
        )
        completed_build = replace(
            running,
            docker_image=result.image_id,
            dependency_versions=result.dependency_versions,
            image_build_finished_at=datetime.now(timezone.utc),
            image_build_log_location=str(
                self._artifact_store.log_directory(running.run_id) / "image-build.log"
            ),
        )
        self._execution_repository.save_run(completed_build)
        return completed_build, len(build_logs)

    def _execute_training(
        self,
        run: Run,
        project_directory: Path,
        entrypoint: str,
        *,
        starting_log_sequence: int,
    ) -> Run:
        if not run.docker_image:
            raise ImageBuildError("The training image was not produced by the image build.")
        started_at = datetime.now(timezone.utc)
        output_directory = self._output_directory(run.run_id)
        output_directory.mkdir(parents=True, exist_ok=True)
        executing = replace(
            run,
            docker_image=run.docker_image,
            started_at=started_at,
        )
        self._execution_repository.save_run(executing)
        result = self._training_executor.execute(
            TrainingExecutionRequest(
                run_id=run.run_id,
                image_id=run.docker_image,
                project_directory=project_directory.resolve(strict=True),
                entrypoint=entrypoint,
                output_directory=output_directory,
                started_at=started_at,
                cpus=run.max_cpus,
                memory_limit=run.memory_limit,
                timeout_seconds=run.timeout_seconds,
                network_disabled=True,
            )
        )
        self._log_collector.collect_records(
            run.run_id,
            result.logs,
            file_name="training.log",
            starting_sequence=starting_log_sequence,
        )
        executing = replace(
            executing,
            log_location=str(self._artifact_store.log_directory(run.run_id) / "training.log"),
        )
        self._execution_repository.save_run(executing)
        if result.out_of_memory:
            raise OutOfMemoryError("Training container exceeded the configured memory limit.")
        if result.timed_out:
            raise TrainingTimeout("Training exceeded its configured timeout.")
        if result.exit_code != 0:
            raise TrainingError(f"Training container exited with status {result.exit_code}.")
        return executing

    def _output_directory(self, run_id: UUID) -> Path:
        return self._artifact_store.run_directory(run_id) / "output"

    def _effective_config(self, options: TrainingOptions) -> dict[str, object]:
        return {
            "build_allowed_repositories": list(self._config.build_allowed_repositories),
            "build_timeout_seconds": options.build_timeout_seconds,
            "cpus": options.cpus,
            "memory_limit": options.memory_limit,
            "parameters": {name: value for name, value in options.parameters},
            "timeout_seconds": options.timeout_seconds,
        }
