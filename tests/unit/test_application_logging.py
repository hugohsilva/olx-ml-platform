from __future__ import annotations

import io
from uuid import uuid4

from infrastructure.logging import configure_application_logging, run_logger


def test_application_logs_include_run_id_correlation() -> None:
    stream = io.StringIO()
    logger = configure_application_logging(stream=stream)
    run_id = uuid4()

    run_logger(run_id, logger=logger).info("training started")

    output = stream.getvalue()
    assert str(run_id) in output
    assert "training started" in output

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
