"""MetricsMiddleware — exposes Prometheus metrics at /metrics."""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, Request, Response

from insight_miner.config import METRICS_ENABLED

# ── Metrics definitions ──

http_requests_total = Counter(
    "insight_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "insight_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

llm_calls_total = Counter(
    "insight_llm_calls_total",
    "Total LLM calls",
    ["provider", "status"],
)

llm_call_duration_seconds = Histogram(
    "insight_llm_call_duration_seconds",
    "LLM call duration",
    ["provider"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

retrieval_channel_hits = Counter(
    "insight_retrieval_channel_hits_total",
    "Total retrieval results per channel",
    ["channel"],
)

cache_hits_total = Counter(
    "insight_cache_hits_total",
    "Total cache hits",
    ["level"],
)

circuit_breaker_state = Counter(
    "insight_circuit_breaker_state_changes",
    "Circuit breaker state changes",
    ["name", "state"],
)


# ── Middleware ──

class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that records HTTP metrics."""

    async def dispatch(self, request: Request, call_next):
        if not METRICS_ENABLED:
            return await call_next(request)

        # Special case: /metrics endpoint
        if request.url.path == "/metrics":
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        http_requests_total.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            path=request.url.path,
        ).observe(duration)

        return response
