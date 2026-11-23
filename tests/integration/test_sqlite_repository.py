from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from domain.models import Metric, Run
from infrastructure.sqlite.execution_repository import SQLiteExecutionRepository
from infrastructure.sqlite.schema import initialize_schema


def test_sqlite_repository_persists_run_and_metrics() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)
    repository = SQLiteExecutionRepository(connection)
    run_id = uuid4()
    run = Run(
        run_id=run_id,
        project_id="iris",
        git_commit="abc123",
        code_hash="a" * 64,
        dataset_identity="iris-public",
        dataset_hash="b" * 64,
        config_hash="c" * 64,
        dependency_versions=(("scikit-learn", "1.6.1"),),
        python_version="3.11.9",
        docker_image="sha256:" + "d" * 64,
        build_allowed_repositories=("pypi.org", "files.pythonhosted.org", "docker.io"),
        created_at=datetime.now(timezone.utc),
        parameters=(("random_state", 42),),
        metrics=(Metric(run_id=run_id, name="accuracy", value=0.95),),
    )

    repository.save_run(run)

    assert repository.get_run(run_id) == run
    assert repository.get_run(uuid4()) is None
    connection.close()


def test_sqlite_schema_enforces_run_foreign_keys_for_metric_and_log_records() -> None:
    connection = sqlite3.connect(":memory:")
    initialize_schema(connection)

    try:
        connection.execute(
            "INSERT INTO metrics (run_id, name, value_json, unit) VALUES (?, ?, ?, ?)",
            (str(uuid4()), "accuracy", "0.95", None),
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("metrics without a parent run must be rejected")
    finally:
        connection.close()
