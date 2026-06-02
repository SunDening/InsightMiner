"""TraceMiddleware — injects trace_id into every request for end-to-end tracing."""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, Request, Response

logger = logging.getLogger(__name__)


class TraceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that adds a trace_id to each request."""

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex[:12])
        start = time.monotonic()

        response: Response = await call_next(request)

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        response.headers["X-Trace-Id"] = trace_id

        # Structured log entry
        log_data = {
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        }
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))

        return response
