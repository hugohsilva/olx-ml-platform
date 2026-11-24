from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from uuid import UUID

import typer

from application.errors import ExecutionNotFoundError
from application.executions.query_service import ExecutionDetail, ExecutionQueryService
from cli.formatters import format_execution_detail, format_execution_summaries
from infrastructure.config import ConfigurationError, PlatformConfig
from infrastructure.filesystem.artifact_store import FilesystemArtifactStore
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


def register_commands(application: typer.Typer) -> None:
    executions = typer.Typer(help="List and inspect training executions.")
    executions.command(name="list")(list_executions)
    executions.command(name="show")(show_execution)
    application.add_typer(executions, name="executions")


def list_executions(
    output_format: str = typer.Option("json", "--format", help="Output format: json or text."),
) -> None:
    _validate_output_format(output_format)
    service, connection = _query_service()
    try:
        executions_list = service.list_executions()
    finally:
        connection.close()
    if output_format == "text":
        typer.echo(format_execution_summaries(executions_list))
        return
    payload = [
        {
            "created_at": _format_datetime(execution.created_at),
            "finished_at": _format_optional_datetime(execution.finished_at),
            "project_id": execution.project_id,
            "run_id": str(execution.run_id),
            "started_at": _format_optional_datetime(execution.started_at),
            "state": execution.state.value,
        }
        for execution in executions_list
    ]
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def show_execution(
    run_id: UUID = typer.Argument(..., help="Execution identifier."),
    output_format: str = typer.Option("json", "--format", help="Output format: json or text."),
) -> None:
    _validate_output_format(output_format)
    service, connection = _query_service()
    try:
        detail = service.get_execution(run_id)
    except ExecutionNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError) as exc:
        typer.echo(f"error: model artifact metadata could not be read: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()
    if output_format == "text":
        typer.echo(format_execution_detail(detail))
        return
    typer.echo(json.dumps(_detail_payload(detail), ensure_ascii=False, separators=(",", ":")))


def _query_service() -> tuple[ExecutionQueryService, sqlite3.Connection]:
    try:
        config = PlatformConfig.from_env()
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(config.database_path)
        initialize_schema(connection)
    except (ConfigurationError, OSError, sqlite3.Error) as exc:
        typer.echo(f"error: execution storage could not be opened: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return (
        ExecutionQueryService(
            SQLiteExecutionRepository(connection),
            FilesystemArtifactStore(config.artifacts_dir),
        ),
        connection,
    )


def _detail_payload(detail: ExecutionDetail) -> dict[str, object]:
    payload = asdict(detail)
    payload["run_id"] = str(detail.run_id)
    payload["state"] = detail.state.value
    payload["created_at"] = _format_datetime(detail.created_at)
    payload["started_at"] = _format_optional_datetime(detail.started_at)
    payload["finished_at"] = _format_optional_datetime(detail.finished_at)
    payload["failure_reason"] = (
        detail.failure_reason.value if detail.failure_reason is not None else None
    )
    payload["metrics"] = [
        {"name": metric.name, "unit": metric.unit, "value": metric.value}
        for metric in detail.metrics
    ]
    return payload


def _format_datetime(value: datetime) -> str:
    return value.isoformat()


def _format_optional_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _validate_output_format(value: str) -> None:
    if value not in {"json", "text"}:
        raise typer.BadParameter("Output format must be 'json' or 'text'.", param_hint="--format")
