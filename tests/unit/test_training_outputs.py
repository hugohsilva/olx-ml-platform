from __future__ import annotations

import json
from uuid import uuid4

import pytest

from application.errors import MissingOutputError
from application.training.training_outputs import parse_training_outputs
from domain.models import Metric


def test_training_outputs_parse_metrics_and_parameters(tmp_path) -> None:
    run_id = uuid4()
    (tmp_path / "metrics.json").write_text(
        json.dumps({"accuracy": 0.96, "model_name": "logistic-regression"}),
        encoding="utf-8",
    )
    (tmp_path / "parameters.json").write_text(
        json.dumps({"random_state": 7, "normalize": True}),
        encoding="utf-8",
    )

    output = parse_training_outputs(tmp_path, run_id)

    assert output.metrics == (
        Metric(run_id=run_id, name="accuracy", value=0.96),
        Metric(run_id=run_id, name="model_name", value="logistic-regression"),
    )
    assert output.parameters == (("normalize", True), ("random_state", 7))


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("metrics.json", "[]"),
        ("metrics.json", '{"score": NaN}'),
        ("parameters.json", '{"x": [1, 2]}'),
        ("parameters.json", "not-json"),
    ],
)
def test_training_outputs_reject_invalid_json_contract(
    tmp_path, filename: str, content: str
) -> None:
    (tmp_path / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")
    (tmp_path / "parameters.json").write_text('{"seed": 7}', encoding="utf-8")
    (tmp_path / filename).write_text(content, encoding="utf-8")

    with pytest.raises(MissingOutputError):
        parse_training_outputs(tmp_path, uuid4())


def test_training_outputs_require_both_metadata_files(tmp_path) -> None:
    (tmp_path / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")

    with pytest.raises(MissingOutputError, match="parameters.json"):
        parse_training_outputs(tmp_path, uuid4())
