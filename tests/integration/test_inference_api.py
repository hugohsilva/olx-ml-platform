from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from uuid import UUID

import uvicorn

from api.app import create_app
from application.errors import InferenceError
from application.inference.model_service import ServedModel
from application.inference.prediction_service import PredictionService
from application.ports import InferenceExecutionRequest

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeModelExecutor:
    def __init__(self, *, error: bool = False) -> None:
        self.error = error
        self.requests: list[InferenceExecutionRequest] = []

    def validate_model(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
    ) -> None:
        return None

    def predict(self, request: InferenceExecutionRequest) -> tuple[int, ...]:
        self.requests.append(request)
        if self.error:
            raise InferenceError("prediction failed")
        return tuple(0 for _ in request.instances.instances)


def test_inference_api_returns_predictions_and_rejects_invalid_payloads() -> None:
    executor = FakeModelExecutor()
    app = create_app(
        PredictionService(
            ServedModel(
                run_id=RUN_ID,
                image_id="sha256:" + "a" * 64,
                artifact_location="/artifacts/model.joblib",
                cpus=1.0,
                memory_limit="1g",
            ),
            executor,
        )
    )
    server, thread, url = _start_server(app)

    try:
        status, response = _post(url, {"instances": [[5.1, 3.5, 1.4, 0.2]]})
        assert status == 200
        assert response == {"run_id": str(RUN_ID), "predictions": [0]}
        assert executor.requests[0].run_id == RUN_ID

        for payload in (
            {},
            {"instances": []},
            {"instances": [[True]]},
            {"instances": [["sensitive-value"]]},
            {"instances": [[1.0]], "run_id": str(RUN_ID)},
        ):
            status, response = _post(url, payload)
            assert status == 422
            assert response["error"]["code"] == "invalid_payload"
            assert "sensitive-value" not in json.dumps(response)
    finally:
        _stop_server(server, thread)


def test_inference_api_returns_explicit_error_when_prediction_fails() -> None:
    executor = FakeModelExecutor(error=True)
    app = create_app(
        PredictionService(
            ServedModel(
                run_id=RUN_ID,
                image_id="sha256:" + "a" * 64,
                artifact_location="/artifacts/model.joblib",
                cpus=1.0,
                memory_limit="1g",
            ),
            executor,
        )
    )
    server, thread, url = _start_server(app)

    try:
        status, response = _post(url, {"instances": [[1.0]]})

        assert status == 500
        assert response["error"]["code"] == "inference_error"
    finally:
        _stop_server(server, thread)


def _start_server(application):
    server = uvicorn.Server(
        uvicorn.Config(application, host="127.0.0.1", port=0, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started or not server.servers:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("The inference API did not start.")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}/predict"


def _stop_server(server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)
    assert not thread.is_alive()


def _post(url: str, payload: object) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
