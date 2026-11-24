from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from application.errors import InferenceError


def register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def invalid_payload(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {
                "location": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_payload",
                    "message": "The prediction payload does not match the inference contract.",
                    "details": details,
                }
            },
        )

    @application.exception_handler(InferenceError)
    async def inference_failed(request: Request, error: InferenceError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": error.code,
                    "message": str(error),
                }
            },
        )
