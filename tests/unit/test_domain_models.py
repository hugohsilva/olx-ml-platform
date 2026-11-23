from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from domain.models import (
    FailureReason,
    InferenceRequest,
    InferenceResponse,
    LogRecord,
    LogStream,
    Metric,
    ModelArtifact,
    Run,
    RunState,
    ServingStatus,
    TrainingProject,
)


def test_training_project_validates_required_references_and_sha256_values() -> None:
    project = TrainingProject(
        project_id="iris",
        entrypoint="train.py",
        dependencies="requirements.txt",
        data_location=Path("data"),
        contract_version="1",
        code_hash="a" * 64,
        dataset_hash="b" * 64,
        config_hash="c" * 64,
    )

    assert project.project_id == "iris"
    with pytest.raises(FrozenInstanceError):
        project.project_id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="code_hash"):
        replace(project, code_hash="invalid")


def test_run_rejects_invalid_hashes_and_resource_limits() -> None:
    run = Run(run_id=uuid4(), project_id="iris")

    with pytest.raises(ValueError, match="dataset_hash"):
        replace(run, dataset_hash="not-sha256")
    with pytest.raises(ValueError, match="max_cpus"):
        replace(run, max_cpus=1.1)
    with pytest.raises(ValueError, match="timeout_seconds"):
        replace(run, timeout_seconds=1801)
    with pytest.raises(ValueError, match="dependency_versions"):
        replace(run, dependency_versions=(("invalid-pair",),))


def test_run_requires_failure_reason_when_created_as_failed() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        Run(
            run_id=uuid4(),
            project_id="iris",
            state=RunState.FAILED,
            finished_at=datetime.now(timezone.utc),
        )

    failed = Run(
        run_id=uuid4(),
        project_id="iris",
        state=RunState.FAILED,
        failure_reason=FailureReason.VALIDATION,
        finished_at=datetime.now(timezone.utc),
    )
    assert failed.state is RunState.FAILED


def test_model_artifact_and_log_record_validate_identity_and_metadata() -> None:
    run_id = uuid4()
    now = datetime.now(timezone.utc)
    artifact = ModelArtifact(
        run_id=run_id,
        artifact_name="model.joblib",
        artifact_location="artifacts/model.joblib",
        size_bytes=20,
        checksum="d" * 64,
        created_at=now,
        serving_status=ServingStatus.AVAILABLE,
    )
    log = LogRecord(
        run_id=run_id,
        sequence=0,
        timestamp=now,
        stream=LogStream.STDOUT,
        message="training started",
    )

    assert artifact.serving_status is ServingStatus.AVAILABLE
    assert log.sequence == 0
    with pytest.raises(ValueError, match="size_bytes"):
        replace(artifact, size_bytes=-1)
    with pytest.raises(ValueError, match="sequence"):
        replace(log, sequence=-1)


def test_metric_rejects_non_finite_values_and_boolean_values() -> None:
    run_id = uuid4()
    with pytest.raises(ValueError, match="finite"):
        Metric(run_id=run_id, name="score", value=float("nan"))
    with pytest.raises(ValueError, match="value"):
        Metric(run_id=run_id, name="score", value=True)


def test_inference_entities_are_immutable_and_validate_instances() -> None:
    request = InferenceRequest(instances=((5.1, 3.5, 1.4, 0.2),))
    response = InferenceResponse(run_id=uuid4(), predictions=(0,))

    assert request.instances[0][0] == 5.1
    assert response.predictions == (0,)
    with pytest.raises(ValueError, match="instances"):
        InferenceRequest(instances=())
    with pytest.raises(ValueError, match="finite"):
        InferenceRequest(instances=((float("inf"),),))
