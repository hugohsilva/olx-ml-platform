from __future__ import annotations

from domain.models import FailureReason


class ApplicationError(Exception):
    code = "application_error"
    failure_reason: FailureReason | None = None


class ValidationError(ApplicationError):
    code = "validation_error"
    failure_reason = FailureReason.VALIDATION


class ImageBuildError(ApplicationError):
    code = "image_build_error"
    failure_reason = FailureReason.IMAGE_BUILD_ERROR


class ImageBuildTimeout(ApplicationError):
    code = "image_build_timeout"
    failure_reason = FailureReason.IMAGE_BUILD_TIMEOUT


class TrainingError(ApplicationError):
    code = "training_error"
    failure_reason = FailureReason.TRAINING_ERROR


class TrainingTimeout(ApplicationError):
    code = "training_timeout"
    failure_reason = FailureReason.TIMEOUT


class OutOfMemoryError(ApplicationError):
    code = "out_of_memory"
    failure_reason = FailureReason.OOM_KILLED


class ResourceLimitError(ApplicationError):
    code = "resource_limit"
    failure_reason = FailureReason.RESOURCE_LIMIT


class ContainerExecutionError(ApplicationError):
    code = "container_error"
    failure_reason = FailureReason.CONTAINER_ERROR


class MissingOutputError(ApplicationError):
    code = "missing_output"
    failure_reason = FailureReason.MISSING_OUTPUT


class InferenceError(ApplicationError):
    code = "inference_error"


def failure_reason_for(error: ApplicationError) -> FailureReason:
    if not isinstance(error, ApplicationError):
        raise TypeError("failure_reason_for requires an ApplicationError.")
    if error.failure_reason is None:
        raise ValueError(f"{error.code} is not a training run failure reason.")
    return error.failure_reason
