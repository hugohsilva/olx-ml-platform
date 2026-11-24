from __future__ import annotations

from fastapi import FastAPI

from api.errors import register_error_handlers
from api.schemas import InferenceRequestSchema, InferenceResponseSchema
from application.inference.prediction_service import PredictionService
from domain.models import InferenceRequest


def create_app(prediction_service: PredictionService) -> FastAPI:
    application = FastAPI(title="OLX ML Platform Inference API")
    register_error_handlers(application)

    @application.post("/predict", response_model=InferenceResponseSchema)
    def predict(payload: InferenceRequestSchema) -> InferenceResponseSchema:
        result = prediction_service.predict(
            InferenceRequest(
                instances=tuple(tuple(value for value in row) for row in payload.instances)
            )
        )
        return InferenceResponseSchema(
            run_id=result.run_id,
            predictions=list(result.predictions),
        )

    return application
