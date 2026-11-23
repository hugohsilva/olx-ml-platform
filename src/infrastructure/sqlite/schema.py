from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL,
    git_commit TEXT,
    code_hash TEXT CHECK (code_hash IS NULL OR length(code_hash) = 64),
    dataset_identity TEXT,
    dataset_hash TEXT CHECK (dataset_hash IS NULL OR length(dataset_hash) = 64),
    config_hash TEXT CHECK (config_hash IS NULL OR length(config_hash) = 64),
    dependency_versions_json TEXT NOT NULL,
    python_version TEXT,
    docker_image TEXT,
    build_allowed_repositories_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('CREATED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')
    ),
    created_at TEXT NOT NULL,
    image_build_started_at TEXT,
    image_build_finished_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    build_timeout_seconds INTEGER NOT NULL CHECK (
        build_timeout_seconds BETWEEN 1 AND 1800
    ),
    timeout_seconds INTEGER NOT NULL CHECK (timeout_seconds BETWEEN 1 AND 1800),
    max_cpus REAL NOT NULL CHECK (max_cpus > 0 AND max_cpus <= 1.0),
    memory_limit TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    log_location TEXT,
    image_build_log_location TEXT,
    failure_reason TEXT CHECK (
        failure_reason IS NULL OR failure_reason IN (
            'validation', 'image_build_error', 'image_build_timeout',
            'training_error', 'timeout', 'oom_killed', 'resource_limit',
            'container_error', 'missing_output'
        )
    ),
    failure_message TEXT,
    CHECK (
        (state IN ('SUCCEEDED', 'FAILED', 'CANCELLED') AND finished_at IS NOT NULL)
        OR (state IN ('CREATED', 'RUNNING') AND finished_at IS NULL)
    ),
    CHECK ((state = 'FAILED' AND failure_reason IS NOT NULL) OR
           (state != 'FAILED' AND failure_reason IS NULL))
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    unit TEXT,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS logs (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    timestamp TEXT NOT NULL,
    stream TEXT NOT NULL CHECK (stream IN ('stdout', 'stderr', 'application')),
    message TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at
    ON runs(created_at DESC, run_id ASC);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_SQL)
