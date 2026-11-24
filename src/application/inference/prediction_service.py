from __future__ import annotations

from application.errors import InferenceError
from application.inference.model_service import ServedModel
from application.ports import InferenceExecutionRequest, ModelExecutor
from domain.models import InferenceRequest, InferenceResponse


class PredictionService:
    def __init__(self, served_model: ServedModel, model_executor: ModelExecutor) -> None:
        self._served_model = served_model
        self._model_executor = model_executor

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        execution_request = InferenceExecutionRequest(
            run_id=self._served_model.run_id,
            image_id=self._served_model.image_id,
            artifact_location=self._served_model.artifact_location,
            instances=request,
            cpus=self._served_model.cpus,
            memory_limit=self._served_model.memory_limit,
        )
        predictions = self._model_executor.predict(execution_request)
        if len(predictions) != len(request.instances):
            raise InferenceError("The model returned a different number of predictions.")
        try:
            return InferenceResponse(
                run_id=self._served_model.run_id,
                predictions=predictions,
            )
        except ValueError as exc:
            raise InferenceError("The model returned invalid prediction values.") from exc
