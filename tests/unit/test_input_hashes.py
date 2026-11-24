from __future__ import annotations

import hashlib
import json

import pytest

from application.errors import ValidationError
from application.training.input_hashes import (
    hash_config,
    hash_dataset_files,
    hash_project_files,
)


def test_project_hash_excludes_dataset_generated_files_and_environment_directories(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data = project / "data"
    data.mkdir()
    (project / "train.py").write_text("print('first')\n", encoding="utf-8")
    (project / "requirements.txt").write_text("joblib==1.5.2\n", encoding="utf-8")
    (data / "records.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("ignored", encoding="utf-8")
    (project / ".venv").mkdir()
    (project / ".venv" / "ignored.py").write_text("ignored", encoding="utf-8")
    (project / "artifacts").mkdir()
    (project / "artifacts" / "model.joblib").write_bytes(b"generated")

    original_hash = hash_project_files(project, data)
    (data / "records.csv").write_text("x,y\n2,1\n", encoding="utf-8")
    (project / ".git" / "config").write_text("changed", encoding="utf-8")
    (project / ".venv" / "ignored.py").write_text("changed", encoding="utf-8")
    (project / "artifacts" / "model.joblib").write_bytes(b"changed")

    assert hash_project_files(project, data) == original_hash


def test_dataset_hash_uses_sorted_relative_paths_and_file_contents(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        (directory / "nested").mkdir(parents=True)
    (first / "z.csv").write_text("z\n", encoding="utf-8")
    (first / "nested" / "a.csv").write_text("a\n", encoding="utf-8")
    (second / "nested" / "a.csv").write_text("a\n", encoding="utf-8")
    (second / "z.csv").write_text("z\n", encoding="utf-8")

    assert hash_dataset_files(first) == hash_dataset_files(second)
    (second / "z.csv").write_text("changed\n", encoding="utf-8")
    assert hash_dataset_files(first) != hash_dataset_files(second)


def test_canonical_config_hash_is_key_order_independent_and_sha256(tmp_path) -> None:
    config = {"limits": {"cpus": 1.0, "memory": "1g"}, "parameters": {"seed": 7}}
    expected = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert hash_config(config) == expected
    assert hash_config({"parameters": {"seed": 7}, "limits": {"memory": "1g", "cpus": 1.0}}) == (
        expected
    )


@pytest.mark.parametrize("kind", ["code", "dataset"])
def test_empty_or_missing_input_sets_are_rejected(tmp_path, kind: str) -> None:
    path = tmp_path / kind
    path.mkdir()
    if kind == "code":
        (path / "data").mkdir()

    with pytest.raises(ValidationError, match="empty"):
        if kind == "code":
            hash_project_files(path, path / "data")
        else:
            hash_dataset_files(path)


def test_project_hash_rejects_symlinked_files_to_avoid_external_inputs(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data = project / "data"
    data.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("external", encoding="utf-8")
    (project / "train.py").symlink_to(outside)

    with pytest.raises(ValidationError, match="symbolic link"):
        hash_project_files(project, data)
