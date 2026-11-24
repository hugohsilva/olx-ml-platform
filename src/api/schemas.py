from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from domain.models import JsonScalar


class InferenceRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[list[float]]

    @field_validator("instances", mode="before")
    @classmethod
    def validate_instances(cls, value: Any) -> Any:
        if not isinstance(value, list) or not value:
            raise ValueError("instances must contain at least one numeric row.")
        for row in value:
            if not isinstance(row, list) or not row:
                raise ValueError("Each instance must be a non-empty numeric row.")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in row
            ):
                raise ValueError("instances must contain finite numeric values.")
        return value


class InferenceResponseSchema(BaseModel):
    run_id: UUID
    predictions: list[JsonScalar]
