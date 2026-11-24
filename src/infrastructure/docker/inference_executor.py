from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import docker
from docker.errors import DockerException

from application.errors import InferenceError
from application.ports import InferenceExecutionRequest
from domain.models import JsonScalar
from infrastructure.docker.container_config import parse_memory_limit

_INFERENCE_TIMEOUT_SECONDS = 30
_POLL_INTERVAL_SECONDS = 0.1
_STOP_CONFIRM_SECONDS = 5.0
_MODEL_RUNNER = """\
import json
import math
import sys
import joblib

model = joblib.load("/model/model.joblib")
if not callable(getattr(model, "predict", None)):
    raise TypeError("model does not expose predict")
if sys.argv[1] == "validate":
    print('{"model_loaded":true}')
else:
    with open("/request/instances.json", encoding="utf-8") as file:
        instances = json.load(file)
    predictions = model.predict(instances)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    predictions = list(predictions)
    normalized = []
    for prediction in predictions:
        if hasattr(prediction, "item"):
            prediction = prediction.item()
        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, (str, int, float))
            or isinstance(prediction, float) and not math.isfinite(prediction)
        ):
            raise TypeError("model returned a non-scalar prediction")
        normalized.append(prediction)
    print(json.dumps(normalized, allow_nan=False, separators=(",", ":")))
"""


class DockerInferenceExecutor:
    def __init__(self, *, docker_client: Any | None = None) -> None:
        self._docker_client = docker_client

    def validate_model(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
    ) -> None:
        self._run(
            image_id,
            artifact_location,
            cpus,
            memory_limit,
            command=["validate"],
        )

    def predict(self, request: InferenceExecutionRequest) -> tuple[JsonScalar, ...]:
        with tempfile.TemporaryDirectory(prefix="mlplatform-inference-") as directory:
            input_path = Path(directory) / "instances.json"
            input_path.write_text(
                json.dumps(
                    request.instances.instances,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            output = self._run(
                request.image_id,
                request.artifact_location,
                request.cpus,
                request.memory_limit,
                command=["predict"],
                input_path=input_path,
                run_id=request.run_id,
            )
        try:
            predictions = json.loads(output)
        except json.JSONDecodeError as exc:
            raise InferenceError("The model returned an invalid prediction response.") from exc
        if not isinstance(predictions, list):
            raise InferenceError("The model returned an invalid prediction response.")
        return tuple(predictions)

    def _run(
        self,
        image_id: str,
        artifact_location: str,
        cpus: float,
        memory_limit: str,
        *,
        command: list[str],
        input_path: Path | None = None,
        run_id=None,
    ) -> str:
        artifact_path = Path(artifact_location)
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise InferenceError("The model artifact is missing or invalid.", run_id=run_id)
        try:
            artifact_path = artifact_path.resolve(strict=True)
            config = build_inference_container_config(
                image_id=image_id,
                artifact_path=artifact_path,
                cpus=cpus,
                memory_limit=memory_limit,
                command=command,
                input_path=input_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise InferenceError(
                "The model artifact or inference limits are invalid.", run_id=run_id
            ) from exc

        container = None
        primary_error: InferenceError | None = None
        try:
            container = self._docker().containers.create(
                name=f"mlplatform-inference-{uuid4().hex}",
                labels={
                    "mlplatform.inference": "true",
                    **({"mlplatform.run_id": str(run_id)} if run_id is not None else {}),
                },
                **config,
            )
            container.start()
            timed_out = self._wait_for_completion(container, run_id)
            container.reload()
            state = container.attrs.get("State", {})
            if timed_out:
                raise InferenceError("Model inference exceeded its 30-second limit.", run_id=run_id)
            if state.get("OOMKilled") is True:
                raise InferenceError("Model inference exceeded its memory limit.", run_id=run_id)
            exit_code = state.get("ExitCode")
            if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
                raise InferenceError(
                    "The model could not be loaded or prediction failed.", run_id=run_id
                )
            output = container.logs(stdout=True, stderr=False)
            return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        except DockerException as exc:
            primary_error = InferenceError(
                "Docker could not execute model inference.", run_id=run_id
            )
            raise primary_error from exc
        except InferenceError as exc:
            primary_error = exc
            raise
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except DockerException as cleanup_error:
                    if primary_error is None:
                        raise InferenceError(
                            "The inference container could not be removed.",
                            run_id=run_id,
                        ) from cleanup_error
                    primary_error.add_note(
                        f"Inference container cleanup also failed: {cleanup_error}"
                    )

    def _wait_for_completion(self, container: Any, run_id) -> bool:
        deadline = time.monotonic() + _INFERENCE_TIMEOUT_SECONDS
        while True:
            container.reload()
            state = container.attrs.get("State", {})
            if state.get("Status") in {"exited", "dead"}:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    container.kill()
                except DockerException as exc:
                    raise InferenceError(
                        "Timed-out inference container could not be stopped.",
                        run_id=run_id,
                    ) from exc
                stop_deadline = time.monotonic() + _STOP_CONFIRM_SECONDS
                while time.monotonic() < stop_deadline:
                    container.reload()
                    if container.attrs.get("State", {}).get("Status") in {"exited", "dead"}:
                        return True
                    time.sleep(_POLL_INTERVAL_SECONDS)
                raise InferenceError(
                    "Timed-out inference container did not stop.",
                    run_id=run_id,
                )
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def _docker(self) -> Any:
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except DockerException as exc:
                raise InferenceError(
                    "The Docker daemon is unavailable for model inference."
                ) from exc
        return self._docker_client


def build_inference_container_config(
    *,
    image_id: str,
    artifact_path: Path,
    cpus: float,
    memory_limit: str,
    command: list[str],
    input_path: Path | None = None,
) -> dict[str, object]:
    if not artifact_path.is_absolute() or artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("The model artifact must be an available regular file.")
    if not image_id.startswith("sha256:"):
        raise ValueError("Inference requires the immutable training image ID.")
    if isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or not math.isfinite(cpus):
        raise ValueError("Inference CPUs must be finite and positive.")
    if not 0 < cpus <= 1:
        raise ValueError("Inference CPUs must be greater than 0 and at most 1.")
    if os.getuid() == 0:
        raise ValueError("The inference executor must run as a non-root host user.")
    volumes: dict[str, dict[str, str]] = {
        str(artifact_path): {"bind": "/model/model.joblib", "mode": "ro"}
    }
    if input_path is not None:
        if input_path.is_symlink() or not input_path.is_file():
            raise ValueError("Inference input must be an available regular file.")
        volumes[str(input_path.resolve(strict=True))] = {
            "bind": "/request/instances.json",
            "mode": "ro",
        }
    return {
        "image": image_id,
        "command": ["python", "-c", _MODEL_RUNNER, *command],
        "cap_drop": ["ALL"],
        "detach": True,
        "environment": {"HOME": "/tmp"},
        "mem_limit": parse_memory_limit(memory_limit),
        "memswap_limit": parse_memory_limit(memory_limit),
        "nano_cpus": round(cpus * 1_000_000_000),
        "network_mode": "none",
        "privileged": False,
        "read_only": True,
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
        "user": f"{os.getuid()}:{os.getgid()}",
        "volumes": volumes,
    }
