from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from uuid import UUID

from domain.executions.state_machine import transition_run
from domain.models import Metric, Run, RunState
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


def test_repository_lists_newest_first_with_run_id_tie_break_and_loads_details() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _succeeded_run("00000000-0000-0000-0000-000000000003", created_at)
    same_time_first = _succeeded_run("00000000-0000-0000-0000-000000000001", created_at)
    newest = _succeeded_run(
        "00000000-0000-0000-0000-000000000002",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    for run in (older, same_time_first, newest):
        repository.save_run(run)

    runs = repository.list_runs()

    assert [str(run.run_id) for run in runs] == [
        str(newest.run_id),
        str(same_time_first.run_id),
        str(older.run_id),
    ]
    assert runs[0].metrics == (Metric(run_id=newest.run_id, name="accuracy", value=0.95),)

    connection.close()


def _succeeded_run(run_id: str, created_at: datetime) -> Run:
    created = Run(
        run_id=UUID(run_id),
        project_id="iris",
        created_at=created_at,
        parameters=(("random_state", 7),),
        metrics=(Metric(run_id=UUID(run_id), name="accuracy", value=0.95),),
        log_location=f"/logs/{run_id}.log",
    )
    running = transition_run(created, RunState.RUNNING, at=created_at)
    return transition_run(running, RunState.SUCCEEDED, at=created_at)
