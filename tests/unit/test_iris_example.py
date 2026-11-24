from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_TRAINER_PATH = Path(__file__).resolve().parents[2] / "examples" / "iris" / "train.py"
_TRAINER_SPEC = spec_from_file_location("iris_example_train", _TRAINER_PATH)
if _TRAINER_SPEC is None or _TRAINER_SPEC.loader is None:
    raise RuntimeError("The Iris example trainer could not be loaded.")
_TRAINER_MODULE = module_from_spec(_TRAINER_SPEC)
_TRAINER_SPEC.loader.exec_module(_TRAINER_MODULE)


def test_iris_example_loads_the_attributed_public_dataset() -> None:
    features, labels = _TRAINER_MODULE.load_dataset()
    attribution = (_TRAINER_PATH.parent / "README.md").read_text(encoding="utf-8")

    assert len(features) == 150
    assert len(labels) == 150
    assert all(len(row) == 4 for row in features)
    assert set(labels) == {0, 1, 2}
    assert "UCI Machine Learning Repository" in attribution
    assert "CC BY 4.0" in attribution
