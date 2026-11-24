from __future__ import annotations

import json
import sqlite3
from uuid import UUID

from domain.models import Metric


class SQLiteMetricsRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def save_metrics(self, run_id: UUID, metrics: tuple[Metric, ...]) -> None:
        if any(metric.run_id != run_id for metric in metrics):
            raise ValueError("Every metric must belong to run_id.")
        with self._connection:
            self._connection.execute("DELETE FROM metrics WHERE run_id = ?", (str(run_id),))
            self._connection.executemany(
                """
                INSERT INTO metrics (run_id, name, value_json, unit)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(metric.run_id),
                        metric.name,
                        json.dumps(
                            metric.value,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                        metric.unit,
                    )
                    for metric in metrics
                ],
            )

    def get_metrics(self, run_id: UUID) -> tuple[Metric, ...]:
        rows = self._connection.execute(
            """
            SELECT name, value_json, unit
            FROM metrics
            WHERE run_id = ?
            ORDER BY name
            """,
            (str(run_id),),
        ).fetchall()
        return tuple(
            Metric(
                run_id=run_id,
                name=row["name"],
                value=json.loads(row["value_json"]),
                unit=row["unit"],
            )
            for row in rows
        )
