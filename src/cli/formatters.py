from __future__ import annotations

import json
from datetime import datetime

from application.executions.query_service import ExecutionDetail, ExecutionSummary


def format_execution_summaries(executions: tuple[ExecutionSummary, ...]) -> str:
    if not executions:
        return "No executions found."
    return "\n".join(
        f"{execution.run_id}  {execution.state.value:<9} {execution.project_id}"
        for execution in executions
    )


def format_execution_detail(detail: ExecutionDetail) -> str:
    lines = [
        f"run_id: {detail.run_id}",
        f"project_id: {detail.project_id}",
        f"state: {detail.state.value}",
        f"created_at: {detail.created_at.isoformat()}",
        f"started_at: {_format_optional_datetime(detail.started_at)}",
        f"finished_at: {_format_optional_datetime(detail.finished_at)}",
        f"artifact_location: {detail.artifact_location or 'unavailable'}",
        f"log_location: {detail.log_location or 'unavailable'}",
        f"image_build_log_location: {detail.image_build_log_location or 'unavailable'}",
        f"docker_image: {detail.docker_image or 'unavailable'}",
        f"parameters: {json.dumps(dict(detail.parameters), ensure_ascii=False, sort_keys=True)}",
        "metrics:",
    ]
    lines.extend(
        f"  {metric.name}: {metric.value}" + (f" {metric.unit}" if metric.unit is not None else "")
        for metric in detail.metrics
    )
    if not detail.metrics:
        lines.append("  none")
    if detail.failure_reason is not None:
        lines.append(f"failure_reason: {detail.failure_reason.value}")
    if detail.failure_message is not None:
        lines.append(f"failure_message: {detail.failure_message}")
    return "\n".join(lines)


def _format_optional_datetime(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "unavailable"
