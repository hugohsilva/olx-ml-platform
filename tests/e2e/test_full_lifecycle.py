from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException
from typer.testing import CliRunner

from cli.main import app


def test_full_training_query_serve_and_prediction_flow(tmp_path: Path) -> None:
    _require_docker()
    home = tmp_path / "mlplatform"
    trained = CliRunner().invoke(
        app,
        ["train", "examples/iris"],
        env={"MLPLATFORM_HOME": str(home)},
    )
    assert trained.exit_code == 0, trained.output
    train_output = dict(line.split("=", 1) for line in trained.output.splitlines() if "=" in line)
    run_id = train_output["run_id"]
    assert train_output["state"] == "SUCCEEDED"

    listed = CliRunner().invoke(
        app,
        ["executions", "show", run_id],
        env={"MLPLATFORM_HOME": str(home)},
    )
    assert listed.exit_code == 0, listed.output
    detail = json.loads(listed.output)
    assert detail["state"] == "SUCCEEDED"
    assert detail["artifact_location"].endswith("/model.joblib")

    unavailable = CliRunner().invoke(
        app,
        ["serve", "00000000-0000-0000-0000-000000000001"],
        env={"MLPLATFORM_HOME": str(home)},
    )
    assert unavailable.exit_code == 1
    assert "was not found" in unavailable.output

    port = _available_port()
    environment = os.environ.copy()
    environment["MLPLATFORM_HOME"] = str(home)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    server_log = tmp_path / "serve.log"
    with server_log.open("w+", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from cli.main import main; main()",
                "serve",
                run_id,
                "--port",
                str(port),
            ],
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(process, port, log_file, timeout_seconds=120)
            status, response = _post_prediction(
                f"http://127.0.0.1:{port}/predict",
                {"instances": [[5.1, 3.5, 1.4, 0.2]]},
            )
            assert status == 200
            assert response["run_id"] == run_id
            assert len(response["predictions"]) == 1

            for payload in (
                {},
                {"instances": []},
                {"instances": [[True]]},
                {"instances": [["5.1"]]},
                {"instances": [[float("inf")]]},
                {"instances": [[1.0]], "run_id": run_id},
                {"instances": [[1.0]], "extra": True},
            ):
                status, response = _post_prediction(
                    f"http://127.0.0.1:{port}/predict",
                    payload,
                )
                assert status == 422
                assert response["error"]["code"] == "invalid_payload"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _require_docker() -> None:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(
    process: subprocess.Popen,
    port: int,
    log_file,
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/openapi.json"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.flush()
            raise AssertionError(f"serve exited before startup:\n{Path(log_file.name).read_text()}")
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise AssertionError("serve did not start before the timeout.")


def _post_prediction(url: str, payload: object) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
