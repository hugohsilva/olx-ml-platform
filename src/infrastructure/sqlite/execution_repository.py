from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import UUID

from domain.models import FailureReason, Metric, Run, RunState

_RUN_COLUMNS = (
    "run_id, project_id, git_commit, code_hash, dataset_identity, dataset_hash, config_hash, "
    "dependency_versions_json, python_version, docker_image, "
    "build_allowed_repositories_json, state, created_at, image_build_started_at, "
    "image_build_finished_at, started_at, finished_at, build_timeout_seconds, "
    "timeout_seconds, max_cpus, memory_limit, parameters_json, log_location, "
    "image_build_log_location, failure_reason, failure_message"
)


class SQLiteExecutionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def save_run(self, run: Run) -> None:
        values = (
            str(run.run_id),
            run.project_id,
            run.git_commit,
            run.code_hash,
            run.dataset_identity,
            run.dataset_hash,
            run.config_hash,
            _dump_json(run.dependency_versions),
            run.python_version,
            run.docker_image,
            _dump_json(run.build_allowed_repositories),
            run.state.value,
            _dump_datetime(run.created_at),
            _dump_optional_datetime(run.image_build_started_at),
            _dump_optional_datetime(run.image_build_finished_at),
            _dump_optional_datetime(run.started_at),
            _dump_optional_datetime(run.finished_at),
            run.build_timeout_seconds,
            run.timeout_seconds,
            run.max_cpus,
            run.memory_limit,
            _dump_json(run.parameters),
            run.log_location,
            run.image_build_log_location,
            run.failure_reason.value if run.failure_reason else None,
            run.failure_message,
        )
        placeholders = ", ".join("?" for _ in values)
        updates = ", ".join(
            f"{column.strip()} = excluded.{column.strip()}"
            for column in _RUN_COLUMNS.split(",")
            if column.strip() != "run_id"
        )

        with self._connection:
            self._connection.execute(
                f"""
                INSERT INTO runs ({_RUN_COLUMNS})
                VALUES ({placeholders})
                ON CONFLICT(run_id) DO UPDATE SET {updates}
                """,
                values,
            )
            self._connection.execute(
                "DELETE FROM metrics WHERE run_id = ?",
                (str(run.run_id),),
            )
            self._connection.executemany(
                """
                INSERT INTO metrics (run_id, name, value_json, unit)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(metric.run_id),
                        metric.name,
                        _dump_json(metric.value),
                        metric.unit,
                    )
                    for metric in run.metrics
                ],
            )

    def get_run(self, run_id: UUID) -> Run | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            return None

        metric_rows = self._connection.execute(
            "SELECT name, value_json, unit FROM metrics WHERE run_id = ? ORDER BY name",
            (str(run_id),),
        ).fetchall()
        metrics = tuple(
            Metric(
                run_id=run_id,
                name=metric_row["name"],
                value=json.loads(metric_row["value_json"]),
                unit=metric_row["unit"],
            )
            for metric_row in metric_rows
        )
        return Run(
            run_id=UUID(row["run_id"]),
            project_id=row["project_id"],
            git_commit=row["git_commit"],
            code_hash=row["code_hash"],
            dataset_identity=row["dataset_identity"],
            dataset_hash=row["dataset_hash"],
            config_hash=row["config_hash"],
            dependency_versions=tuple(
                tuple(pair) for pair in json.loads(row["dependency_versions_json"])
            ),
            python_version=row["python_version"],
            docker_image=row["docker_image"],
            build_allowed_repositories=tuple(json.loads(row["build_allowed_repositories_json"])),
            state=RunState(row["state"]),
            created_at=_load_datetime(row["created_at"]),
            image_build_started_at=_load_optional_datetime(row["image_build_started_at"]),
            image_build_finished_at=_load_optional_datetime(row["image_build_finished_at"]),
            started_at=_load_optional_datetime(row["started_at"]),
            finished_at=_load_optional_datetime(row["finished_at"]),
            build_timeout_seconds=row["build_timeout_seconds"],
            timeout_seconds=row["timeout_seconds"],
            max_cpus=row["max_cpus"],
            memory_limit=row["memory_limit"],
            parameters=tuple(tuple(pair) for pair in json.loads(row["parameters_json"])),
            metrics=metrics,
            log_location=row["log_location"],
            image_build_log_location=row["image_build_log_location"],
            failure_reason=(
                FailureReason(row["failure_reason"]) if row["failure_reason"] is not None else None
            ),
            failure_message=row["failure_message"],
        )

    def list_runs(self) -> tuple[Run, ...]:
        rows = self._connection.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC, run_id ASC"
        ).fetchall()
        return tuple(run for row in rows if (run := self.get_run(UUID(row["run_id"]))) is not None)


def _dump_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _dump_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _dump_optional_datetime(value: datetime | None) -> str | None:
    return _dump_datetime(value) if value is not None else None


def _load_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_optional_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None
