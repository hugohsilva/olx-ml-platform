from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from application.ports import ArtifactStore, LogRepository
from domain.models import LogRecord, LogStream


class LogCollector:
    def __init__(self, artifact_store: ArtifactStore, log_repository: LogRepository) -> None:
        self._artifact_store = artifact_store
        self._log_repository = log_repository

    def collect_messages(
        self,
        run_id: UUID,
        messages: Sequence[str],
        *,
        file_name: str,
        starting_sequence: int = 0,
    ) -> tuple[LogRecord, ...]:
        now = datetime.now(timezone.utc)
        records = tuple(
            LogRecord(
                run_id=run_id,
                sequence=starting_sequence + index,
                timestamp=now,
                stream=LogStream.APPLICATION,
                message=message,
            )
            for index, message in enumerate(messages)
        )
        return self._persist(run_id, records, file_name)

    def collect_records(
        self,
        run_id: UUID,
        records: Sequence[LogRecord],
        *,
        file_name: str,
        starting_sequence: int = 0,
    ) -> tuple[LogRecord, ...]:
        normalized = tuple(
            LogRecord(
                run_id=run_id,
                sequence=starting_sequence + index,
                timestamp=record.timestamp,
                stream=record.stream,
                message=record.message,
            )
            for index, record in enumerate(records)
        )
        return self._persist(run_id, normalized, file_name)

    def _persist(
        self,
        run_id: UUID,
        records: tuple[LogRecord, ...],
        file_name: str,
    ) -> tuple[LogRecord, ...]:
        content = "".join(
            f"{record.timestamp.isoformat()} {record.stream.value} {record.message}\n"
            for record in records
        )
        self._artifact_store.save_log(run_id, file_name, content)
        self._log_repository.save_logs(run_id, records)
        return records
