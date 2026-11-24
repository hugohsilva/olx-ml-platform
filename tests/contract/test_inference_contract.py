from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import InferenceRequestSchema


def test_inference_contract_accepts_instances_only() -> None:
    request = InferenceRequestSchema.model_validate({"instances": [[5.1, 3.5, 1.4, 0.2]]})

    assert request.instances == [[5.1, 3.5, 1.4, 0.2]]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"instances": []},
        {"instances": [[]]},
        {"instances": [[True]]},
        {"instances": [["5.1"]]},
        {"instances": [[float("inf")]]},
        {"instances": [[1.0]], "run_id": "00000000-0000-0000-0000-000000000001"},
        {"instances": [[1.0]], "extra": True},
    ],
)
def test_inference_contract_rejects_invalid_or_additional_fields(payload: object) -> None:
    with pytest.raises(ValidationError):
        InferenceRequestSchema.model_validate(payload)
