from __future__ import annotations

import math
import os
import re
from pathlib import Path

from application.ports import TrainingExecutionRequest

TRAINING_PROJECT_PATH = "/workspace"
TRAINING_OUTPUT_PATH = "/output"
TRAINING_TEMP_PATH = "/tmp"


def build_training_container_config(request: TrainingExecutionRequest) -> dict[str, object]:
    if not request.network_disabled:
        raise ValueError("Training containers must run without network access.")
    if request.project_directory.is_symlink() or request.output_directory.is_symlink():
        raise ValueError("Training mounts must not be symbolic links.")
    try:
        project_directory = request.project_directory.resolve(strict=True)
        output_directory = request.output_directory.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Training mounts must be available directories.") from exc
    if not project_directory.is_dir() or not output_directory.is_dir():
        raise ValueError("Training mounts must be directories.")
    if Path(request.entrypoint).is_absolute() or ".." in Path(request.entrypoint).parts:
        raise ValueError("Training entrypoint must remain inside the project directory.")
    if os.getuid() == 0:
        raise ValueError("The training executor must run as a non-root host user.")
    if (
        isinstance(request.cpus, bool)
        or not isinstance(request.cpus, (int, float))
        or not math.isfinite(request.cpus)
        or not 0 < request.cpus <= 1
    ):
        raise ValueError("cpus must be greater than 0 and at most 1.")
    if (
        isinstance(request.timeout_seconds, bool)
        or not isinstance(request.timeout_seconds, int)
        or not 1 <= request.timeout_seconds <= 1800
    ):
        raise ValueError("timeout_seconds must be between 1 and 1800.")

    memory_bytes = parse_memory_limit(request.memory_limit)
    nano_cpus = round(request.cpus * 1_000_000_000)
    if nano_cpus < 1:
        raise ValueError("cpus is too small to represent as a Docker CPU limit.")
    return {
        "command": ["python", f"{TRAINING_PROJECT_PATH}/{request.entrypoint}"],
        "cap_drop": ["ALL"],
        "detach": True,
        "environment": {
            "HOME": TRAINING_TEMP_PATH,
            "MLPLATFORM_OUTPUT_DIR": TRAINING_OUTPUT_PATH,
            "MLPLATFORM_RUN_ID": str(request.run_id),
        },
        "mem_limit": memory_bytes,
        "memswap_limit": memory_bytes,
        "nano_cpus": nano_cpus,
        "network_mode": "none",
        "privileged": False,
        "read_only": True,
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": {
            TRAINING_TEMP_PATH: "rw,noexec,nosuid,nodev,size=64m",
        },
        "user": f"{os.getuid()}:{os.getgid()}",
        "volumes": {
            str(project_directory): {
                "bind": TRAINING_PROJECT_PATH,
                "mode": "ro",
            },
            str(output_directory): {
                "bind": TRAINING_OUTPUT_PATH,
                "mode": "rw",
            },
        },
        "working_dir": TRAINING_PROJECT_PATH,
    }


def parse_memory_limit(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([mg])", value.lower()) if isinstance(value, str) else None
    if match is None:
        raise ValueError("memory limit must use M or G units.")
    amount = int(match.group(1))
    multiplier = 1024**2 if match.group(2) == "m" else 1024**3
    if amount <= 0 or amount > 1024**3 // multiplier:
        raise ValueError("memory limit must be greater than 0 and at most 1g.")
    return amount * multiplier
