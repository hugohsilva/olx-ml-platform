from __future__ import annotations

import pytest

from application.errors import (
    ApplicationError,
    ImageBuildTimeout,
    InferenceError,
    ValidationError,
    failure_reason_for,
)
from domain.models import FailureReason


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ValidationError("invalid project"), FailureReason.VALIDATION),
        (ImageBuildTimeout("build timed out"), FailureReason.IMAGE_BUILD_TIMEOUT),
    ],
)
def test_application_errors_have_explicit_failure_reasons(
    error: ApplicationError, reason: FailureReason
) -> None:
    assert failure_reason_for(error) is reason


def test_failure_mapping_rejects_unknown_exceptions() -> None:
    with pytest.raises(TypeError, match="ApplicationError"):
        failure_reason_for(RuntimeError("unexpected"))


def test_inference_errors_are_not_mapped_to_training_run_failure_reasons() -> None:
    error = InferenceError("prediction failed")

    assert error.code == "inference_error"
    with pytest.raises(ValueError, match="not a training run failure"):
        failure_reason_for(error)
