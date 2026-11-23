from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from domain.models import FailureReason, Run, RunState

_ALLOWED_TRANSITIONS = {
    RunState.CREATED: {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED},
    RunState.RUNNING: {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED},
    RunState.SUCCEEDED: set(),
    RunState.FAILED: set(),
    RunState.CANCELLED: set(),
}


class InvalidStateTransition(ValueError):
    """Raised when an execution attempts an unsupported state transition."""


def transition_run(
    run: Run,
    target: RunState,
    *,
    failure_reason: FailureReason | None = None,
    failure_message: str | None = None,
    at: datetime | None = None,
) -> Run:
    if target not in _ALLOWED_TRANSITIONS[run.state]:
        raise InvalidStateTransition(f"Cannot transition a run from {run.state} to {target}.")
    if target is RunState.FAILED and failure_reason is None:
        raise ValueError("A failed run requires failure_reason.")
    if target is not RunState.FAILED and (
        failure_reason is not None or failure_message is not None
    ):
        raise ValueError("Failure details can only be set when transitioning to FAILED.")

    finished_at = at or datetime.now(timezone.utc)
    return replace(
        run,
        state=target,
        finished_at=finished_at
        if target in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
        else None,
        failure_reason=failure_reason,
        failure_message=failure_message,
    )
