from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from application.errors import (
    ImageBuildError,
    ImageBuildTimeout,
    ResourceLimitError,
)
from application.ports import (
    ImageBuildRequest,
    ImageBuildResult,
    TrainingExecutionRequest,
    TrainingExecutionResult,
)
from cli.main import app


class FakeImageBuilder:
    def __init__(self, *, failure: str) -> None:
        self.failure = failure

    def build(self, request: ImageBuildRequest) -> ImageBuildResult:
        if self.failure == "build_timeout":
            raise ImageBuildTimeout("build timed out", run_id=request.run_id)
        if self.failure == "build_error":
            raise ImageBuildError("image build failed", run_id=request.run_id)
        return ImageBuildResult(
            image_id="sha256:" + "a" * 64,
            dependency_versions=(("scikit-learn", "1.8.0"),),
            logs=(),
        )


class FakeTrainingExecutor:
    def __init__(self, *, failure: str) -> None:
        self.failure = failure
        self.called = False

    def execute(self, request: TrainingExecutionRequest) -> TrainingExecutionResult:
        self.called = True
        if self.failure == "resource_limit":
            raise ResourceLimitError("resource limits unavailable", run_id=request.run_id)
        return TrainingExecutionResult(
            exit_code=7 if self.failure == "training_error" else 0,
            logs=(),
            timed_out=self.failure == "training_timeout",
            out_of_memory=self.failure == "training_oom",
        )


@pytest.mark.parametrize(
    ("failure", "expected_reason", "training_started"),
    [
        ("build_timeout", "image_build_timeout", False),
        ("build_error", "image_build_error", False),
        ("training_timeout", "timeout", True),
        ("training_oom", "oom_killed", True),
        ("training_error", "training_error", True),
        ("resource_limit", "resource_limit", True),
    ],
)
def test_cli_persists_training_failures_with_explicit_reason(
    tmp_path: Path,
    monkeypatch,
    failure: str,
    expected_reason: str,
    training_started: bool,
) -> None:
    project = tmp_path / "iris-project"
    (project / "data").mkdir(parents=True)
    (project / "train.py").write_text("print('training')\n", encoding="utf-8")
    (project / "requirements.txt").write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    (project / "data" / "rows.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    home = tmp_path / "home"
    executor = FakeTrainingExecutor(failure=failure)
    monkeypatch.setattr(
        "cli.commands.train.DockerImageBuilder",
        lambda config: FakeImageBuilder(failure=failure),
    )
    monkeypatch.setattr("cli.commands.train.DockerTrainingExecutor", lambda: executor)

    result = CliRunner().invoke(
        app,
        ["train", str(project)],
        env={"MLPLATFORM_HOME": str(home)},
    )

    assert result.exit_code == 1
    run_id = next(
        line.split("=", 1)[1] for line in result.output.splitlines() if line.startswith("run_id=")
    )
    connection = sqlite3.connect(home / "metadata.sqlite3")
    state, reason, started_at = connection.execute(
        "SELECT state, failure_reason, started_at FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    connection.close()

    assert state == "FAILED"
    assert reason == expected_reason
    assert (started_at is not None) is training_started
    assert executor.called is training_started
