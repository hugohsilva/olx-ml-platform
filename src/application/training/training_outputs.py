from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from application.errors import MissingOutputError
from domain.models import JsonScalar, Metric, NamedValue


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    parameters: tuple[NamedValue, ...]
    metrics: tuple[Metric, ...]


def parse_training_outputs(output_directory: Path, run_id: UUID) -> TrainingOutputs:
    parameters = _read_object(output_directory / "parameters.json")
    metric_values = _read_object(output_directory / "metrics.json")
    try:
        if not metric_values:
            raise ValueError("metrics.json must contain at least one metric.")
        metrics = tuple(
            Metric(run_id=run_id, name=name, value=metric_values[name])
            for name in sorted(metric_values)
        )
        parameter_values: tuple[NamedValue, ...] = tuple(
            (name, parameters[name]) for name in sorted(parameters)
        )
    except (TypeError, ValueError) as exc:
        raise MissingOutputError(f"Training output metadata is invalid: {exc}") from exc
    return TrainingOutputs(parameters=parameter_values, metrics=metrics)


def _read_object(path: Path) -> dict[str, JsonScalar]:
    if path.is_symlink() or not path.is_file():
        raise MissingOutputError(f"Required training output {path.name} is missing.")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MissingOutputError(f"Required training output {path.name} is invalid: {exc}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MissingOutputError(f"Required training output {path.name} must be a JSON object.")
    if any(not isinstance(item, (str, int, float, bool, type(None))) for item in value.values()):
        raise MissingOutputError(
            f"Required training output {path.name} values must be JSON scalars."
        )
    return value


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value} is not allowed.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result
