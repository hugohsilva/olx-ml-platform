from __future__ import annotations

import sqlite3
from uuid import UUID

import typer
import uvicorn

from api.app import create_app
from application.errors import ApplicationError
from application.inference.model_service import ModelService
from application.inference.prediction_service import PredictionService
from infrastructure.config import ConfigurationError, PlatformConfig
from infrastructure.docker.inference_executor import DockerInferenceExecutor
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


def register_commands(application: typer.Typer) -> None:
    application.command(name="serve", help="Serve a trained model over the local REST API.")(serve)


def serve(
    run_id: UUID = typer.Argument(..., help="Successful execution to serve."),
    port: int = typer.Option(8000, min=1, max=65535, help="Local API port."),
) -> None:
    connection: sqlite3.Connection | None = None
    try:
        config = PlatformConfig.from_env()
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(config.database_path)
        initialize_schema(connection)
        executor = DockerInferenceExecutor()
        served_model = ModelService(
            SQLiteExecutionRepository(connection),
            FilesystemArtifactStore(config.artifacts_dir),
            executor,
        ).load_model(run_id)
        api = create_app(PredictionService(served_model, executor))
    except (ApplicationError, ConfigurationError, OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"error: model could not be served: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if connection is not None:
            connection.close()

    typer.echo(f"serving run_id={run_id} url=http://127.0.0.1:{port}")
    try:
        uvicorn.run(api, host="127.0.0.1", port=port)
    except OSError as exc:
        typer.echo(f"error: inference server could not start: {exc}", err=True)
        raise typer.Exit(code=1) from exc
