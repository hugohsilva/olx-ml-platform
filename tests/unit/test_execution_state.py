from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from domain.executions.state_machine import InvalidStateTransition, transition_run
from domain.models import FailureReason, Run, RunState


def new_run() -> Run:
    return Run(run_id=uuid4(), project_id="iris")


def test_run_can_transition_from_created_to_running_to_succeeded() -> None:
    created = new_run()
    running = transition_run(created, RunState.RUNNING)
    finished_at = datetime.now(timezone.utc)

    succeeded = transition_run(running, RunState.SUCCEEDED, at=finished_at)

    assert created.state is RunState.CREATED
    assert running.state is RunState.RUNNING
    assert succeeded.state is RunState.SUCCEEDED
    assert succeeded.finished_at == finished_at
    assert succeeded.failure_reason is None


def test_failed_transition_requires_a_reason_and_persists_terminal_state() -> None:
    run = new_run()

    with pytest.raises(ValueError, match="failure_reason"):
        transition_run(run, RunState.FAILED)

    failed = transition_run(
        run,
        RunState.FAILED,
        failure_reason=FailureReason.VALIDATION,
        failure_message="Missing training entrypoint.",
    )

    assert failed.state is RunState.FAILED
    assert failed.failure_reason is FailureReason.VALIDATION
    assert failed.failure_message == "Missing training entrypoint."
    assert failed.finished_at is not None


@pytest.mark.parametrize(
    ("initial_state", "target_state"),
    [
        (RunState.CREATED, RunState.SUCCEEDED),
        (RunState.SUCCEEDED, RunState.RUNNING),
        (RunState.FAILED, RunState.RUNNING),
        (RunState.CANCELLED, RunState.SUCCEEDED),
    ],
)
def test_invalid_run_transitions_are_rejected(
    initial_state: RunState, target_state: RunState
) -> None:
    run = new_run()
    if initial_state is RunState.FAILED:
        run = transition_run(
            run,
            RunState.FAILED,
            failure_reason=FailureReason.TRAINING_ERROR,
        )
    elif initial_state is RunState.CANCELLED:
        run = transition_run(run, RunState.CANCELLED)
    elif initial_state is RunState.SUCCEEDED:
        run = transition_run(run, RunState.RUNNING)
        run = transition_run(run, RunState.SUCCEEDED)

    with pytest.raises(InvalidStateTransition):
        transition_run(run, target_state)
