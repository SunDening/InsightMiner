"""Rate limiter — Token Bucket + FastAPI middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from insight_miner.config import RATE_LIMIT_BURST, RATE_LIMIT_DEFAULT, RATE_LIMIT_ENABLED

logger = logging.getLogger(__name__)


class TokenBucket:
    """Per-key token bucket for rate limiting."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class RateLimiter:
    """Global rate limiter with per-IP buckets."""

    def __init__(
        self,
        default_rate: float = RATE_LIMIT_DEFAULT / 60.0,  # convert rpm to rps
        burst: int = RATE_LIMIT_BURST,
        enabled: bool = RATE_LIMIT_ENABLED,
    ) -> None:
        self._default_rate = default_rate
        self._burst = burst
        self._enabled = enabled
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(default_rate, burst),
        )

    def check(self, key: str) -> bool:
        if not self._enabled:
            return True
        return self._buckets[key].consume()

    def reset(self, key: str) -> None:
        self._buckets.pop(key, None)


# ── FastAPI Middleware ──

def rate_limit_middleware(app: FastAPI, limiter: RateLimiter | None = None) -> None:
    """Install rate limiting middleware on a FastAPI app.

    Limits by client IP. Skips health check endpoints.
    """
    if limiter is None:
        limiter = RateLimiter()

    @app.middleware("http")
    async def _rate_limit_mw(request: Request, call_next):
        # Skip health checks
        if request.url.path in ("/api/system/health", "/metrics", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if not limiter.check(client_ip):
            logger.warning("rate_limit exceeded ip=%s path=%s", client_ip, request.url.path)
            return Response(
                content='{"error":"Too Many Requests","code":429}',
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "10"},
            )

        return await call_next(request)
