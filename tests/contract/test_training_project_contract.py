from __future__ import annotations

from pathlib import Path

import pytest

from application.errors import ValidationError
from application.training.project_validator import validate_training_project


def make_project(root: Path) -> Path:
    project = root / "iris-project"
    (project / "data").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "requirements.txt").write_text("joblib==1.5.2\n", encoding="utf-8")
    (project / "data" / "records.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    return project


def test_training_project_contract_accepts_required_files_and_builds_fingerprints(tmp_path) -> None:
    project_path = make_project(tmp_path)

    project = validate_training_project(
        project_path,
        effective_config={"cpus": 1.0, "timeout_seconds": 600},
    )

    assert project.project_id == "iris-project"
    assert project.entrypoint == "train.py"
    assert project.dependencies == "requirements.txt"
    assert project.data_location == project_path.resolve() / "data"
    assert project.contract_version == "1"
    assert len(project.code_hash or "") == 64
    assert len(project.dataset_hash or "") == 64
    assert len(project.config_hash or "") == 64
    assert project.dataset_identity == "data"
    assert project.python_version is not None


@pytest.mark.parametrize(
    ("relative_path", "kind"),
    [
        ("train.py", "file"),
        ("requirements.txt", "file"),
        ("data", "directory"),
    ],
)
def test_training_project_contract_rejects_missing_required_inputs(
    tmp_path, relative_path: str, kind: str
) -> None:
    project = make_project(tmp_path)
    target = project / relative_path
    if kind == "directory":
        (target / "records.csv").unlink()
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(ValidationError, match="required"):
        validate_training_project(project, effective_config={"timeout_seconds": 600})


def test_training_project_contract_rejects_empty_dataset(tmp_path) -> None:
    project = make_project(tmp_path)
    (project / "data" / "records.csv").unlink()

    with pytest.raises(ValidationError, match="empty"):
        validate_training_project(project, effective_config={"timeout_seconds": 600})
