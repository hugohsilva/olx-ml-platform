from __future__ import annotations

import logging
import sys
from typing import TextIO
from uuid import UUID

_LOGGER_NAME = "olx_ml_platform"
_HANDLER_MARKER = "_olx_ml_platform_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s run_id=%(run_id)s %(name)s: %(message)s"


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return True


def configure_application_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        handler.addFilter(RunIdFilter())
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)

    return logger


def run_logger(run_id: UUID, *, logger: logging.Logger | None = None) -> logging.LoggerAdapter:
    if not isinstance(run_id, UUID):
        raise ValueError("run_id must be a UUID.")
    return logging.LoggerAdapter(
        logger or logging.getLogger(_LOGGER_NAME),
        {"run_id": str(run_id)},
    )
