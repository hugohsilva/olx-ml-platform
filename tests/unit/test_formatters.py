from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from application.executions.query_service import ExecutionDetail
from cli.formatters import format_execution_detail
from domain.models import RunState


def test_execution_detail_formatter_includes_artifact_reference() -> None:
    detail = ExecutionDetail(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        project_id="iris",
        state=RunState.SUCCEEDED,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        started_at=None,
        finished_at=None,
        parameters=(),
        metrics=(),
        log_location=None,
        image_build_log_location=None,
        docker_image="sha256:" + "a" * 64,
        artifact_location="/artifacts/model.joblib",
        failure_reason=None,
        failure_message=None,
    )

    text = format_execution_detail(detail)

    assert "state: SUCCEEDED" in text
    assert "artifact_location: /artifacts/model.joblib" in text
    assert "metrics:\n  none" in text
