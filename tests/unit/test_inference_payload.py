from __future__ import annotations

import math

import pytest

from domain.models import InferenceRequest


@pytest.mark.parametrize(
    "instances",
    [
        (),
        ((),),
        ((True,),),
        (("1.0",),),
        ((math.inf,),),
        ((math.nan,),),
    ],
)
def test_inference_request_rejects_empty_or_non_numeric_instances(instances) -> None:
    with pytest.raises(ValueError):
        InferenceRequest(instances=instances)


def test_inference_request_normalizes_numeric_rows() -> None:
    request = InferenceRequest(instances=((1, 2.5),))

    assert request.instances == ((1, 2.5),)
