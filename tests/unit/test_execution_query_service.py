from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from application.errors import ExecutionNotFoundError
from application.executions.query_service import ExecutionQueryService
from domain.executions.state_machine import transition_run
from domain.models import Metric, ModelArtifact, Run, RunState, ServingStatus


def test_query_service_returns_summary_and_full_execution_detail() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = Run(
        run_id=run_id,
        project_id="iris",
        created_at=created_at,
        parameters=(("random_state", 7),),
        metrics=(Metric(run_id=run_id, name="accuracy", value=0.95),),
        log_location="/logs/training.log",
        image_build_log_location="/logs/image-build.log",
    )
    run = transition_run(
        transition_run(created, RunState.RUNNING, at=created_at),
        RunState.SUCCEEDED,
        at=created_at,
    )
    artifact = ModelArtifact(
        run_id=run_id,
        artifact_name="model.joblib",
        artifact_location="/artifacts/model.joblib",
        size_bytes=10,
        checksum="a" * 64,
        created_at=created_at,
        serving_status=ServingStatus.AVAILABLE,
    )
    service = ExecutionQueryService(
        FakeExecutionRepository((run,)),
        FakeArtifactStore(artifact),
    )

    summaries = service.list_executions()
    detail = service.get_execution(run_id)

    assert summaries[0].run_id == run_id
    assert summaries[0].state is RunState.SUCCEEDED
    assert detail.parameters == (("random_state", 7),)
    assert detail.metrics == (Metric(run_id=run_id, name="accuracy", value=0.95),)
    assert detail.log_location == "/logs/training.log"
    assert detail.image_build_log_location == "/logs/image-build.log"
    assert detail.artifact_location == "/artifacts/model.joblib"


def test_query_service_returns_null_artifact_location_when_artifact_is_missing() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    run = Run(run_id=run_id, project_id="iris")
    service = ExecutionQueryService(FakeExecutionRepository((run,)), FakeArtifactStore(None))

    assert service.get_execution(run_id).artifact_location is None


def test_query_service_raises_explicit_not_found_error() -> None:
    service = ExecutionQueryService(FakeExecutionRepository(()), FakeArtifactStore(None))

    with pytest.raises(ExecutionNotFoundError, match="was not found"):
        service.get_execution(uuid4())


class FakeExecutionRepository:
    def __init__(self, runs: tuple[Run, ...]) -> None:
        self._runs = runs

    def list_runs(self) -> tuple[Run, ...]:
        return self._runs

    def get_run(self, run_id: UUID) -> Run | None:
        return next((run for run in self._runs if run.run_id == run_id), None)

    def save_run(self, run: Run) -> None:
        raise AssertionError("Query service must not write executions.")


class FakeArtifactStore:
    def __init__(self, artifact: ModelArtifact | None) -> None:
        self._artifact = artifact

    def get_model_artifact(self, run_id: UUID) -> ModelArtifact | None:
        return self._artifact

    def save_artifact(self, run_id: UUID, artifact_name: str, content: bytes) -> Path:
        raise AssertionError("Query service must not write artifacts.")

    def save_log(self, run_id: UUID, log_name: str, content: str) -> Path:
        raise AssertionError("Query service must not write logs.")
