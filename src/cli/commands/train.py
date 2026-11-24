from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import typer

from application.errors import ApplicationError
from application.training.training_service import TrainingOptions, TrainingService
from domain.models import Run
from infrastructure.config import ConfigurationError, PlatformConfig
from infrastructure.docker.container_config import parse_memory_limit
from infrastructure.docker.image_builder import DockerImageBuilder
from infrastructure.docker.training_executor import DockerTrainingExecutor
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.log_repository import SQLiteLogRepository
from infrastructure.sqlite.metrics_repository import SQLiteMetricsRepository
from infrastructure.sqlite.schema import initialize_schema


def register_commands(application: typer.Typer) -> None:
    application.command(name="train", help="Build and run an isolated training project.")(train)


def train(
    project: Path = typer.Argument(
        ...,
        help="Training project containing train.py, requirements.txt and data/.",
    ),
    timeout_seconds: int = typer.Option(600, min=1, max=1800),
    build_timeout_seconds: int = typer.Option(600, min=1, max=1800),
    cpus: float = typer.Option(1.0, min=0.0, max=1.0),
    memory_limit: str = typer.Option("1g"),
) -> None:
    if not math.isfinite(cpus) or not 0 < cpus <= 1:
        raise typer.BadParameter("--cpus must be greater than 0 and at most 1.0.")
    try:
        parse_memory_limit(memory_limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--memory-limit") from exc

    try:
        config = PlatformConfig.from_env()
    except ConfigurationError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(config.database_path)
    except (OSError, sqlite3.Error) as exc:
        typer.echo("error: local metadata database could not be opened.", err=True)
        raise typer.Exit(code=1) from exc

    run_id_printed = False

    def report_created_run(run: Run) -> None:
        nonlocal run_id_printed
        typer.echo(f"run_id={run.run_id}")
        run_id_printed = True

    try:
        initialize_schema(connection)
        artifact_store = FilesystemArtifactStore(config.artifacts_dir)
        service = TrainingService(
            execution_repository=SQLiteExecutionRepository(connection),
            metrics_repository=SQLiteMetricsRepository(connection),
            log_repository=SQLiteLogRepository(connection),
            artifact_store=artifact_store,
            image_builder=DockerImageBuilder(config),
            training_executor=DockerTrainingExecutor(),
            config=config,
        )
        run = service.train(
            project,
            options=TrainingOptions(
                cpus=cpus,
                memory_limit=memory_limit,
                build_timeout_seconds=build_timeout_seconds,
                timeout_seconds=timeout_seconds,
            ),
            on_run_created=report_created_run,
        )
    except ApplicationError as exc:
        if exc.run_id is not None and not run_id_printed:
            typer.echo(f"run_id={exc.run_id}")
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"invalid training configuration: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except sqlite3.Error as exc:
        typer.echo("error: local metadata operation failed.", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()

    typer.echo(f"state={run.state.value}")
