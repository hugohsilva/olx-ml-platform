from __future__ import annotations

import sqlite3
from datetime import timezone
from uuid import UUID

from domain.models import LogRecord


class SQLiteLogRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute("PRAGMA foreign_keys = ON")

    def save_logs(self, run_id: UUID, logs: tuple[LogRecord, ...]) -> None:
        if any(record.run_id != run_id for record in logs):
            raise ValueError("Every log record must belong to run_id.")
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO logs (run_id, sequence, timestamp, stream, message)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sequence) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    stream = excluded.stream,
                    message = excluded.message
                """,
                [
                    (
                        str(record.run_id),
                        record.sequence,
                        record.timestamp.astimezone(timezone.utc).isoformat(),
                        record.stream.value,
                        record.message,
                    )
                    for record in logs
                ],
            )
