from __future__ import annotations

from uuid import uuid4

import pytest

from infrastructure.filesystem.artifact_store import FilesystemArtifactStore


def test_filesystem_store_persists_artifacts_and_logs_under_their_run(tmp_path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    run_id = uuid4()

    artifact_path = store.save_artifact(run_id, "model.joblib", b"serialized-model")
    log_path = store.save_log(run_id, "training.log", "training completed\n")

    assert artifact_path.read_bytes() == b"serialized-model"
    assert log_path.read_text(encoding="utf-8") == "training completed\n"
    assert artifact_path.parent.parent == log_path.parent.parent
    assert run_id.hex in str(artifact_path)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "subdir/model.joblib", ""])
def test_filesystem_store_rejects_artifact_names_outside_the_run_directory(
    tmp_path, name: str
) -> None:
    store = FilesystemArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="file name"):
        store.save_artifact(uuid4(), name, b"content")
