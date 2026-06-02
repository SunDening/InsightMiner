"""Circuit breaker for LLM calls — sliding window failure counting.

States:
  CLOSED  → normal operation
  OPEN    → failing fast, no calls through
  HALF_OPEN → probing if service recovered
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from insight_miner.config import (
    CIRCUIT_BREAKER_ENABLED,
    CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
    CIRCUIT_BREAKER_WINDOW,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Sliding-window circuit breaker for LLM calls."""

    def __init__(
        self,
        name: str = "llm",
        threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        window: int = CIRCUIT_BREAKER_WINDOW,
        timeout: int = CIRCUIT_BREAKER_TIMEOUT,
        enabled: bool = CIRCUIT_BREAKER_ENABLED,
    ) -> None:
        self._name = name
        self._threshold = threshold
        self._window = window
        self._timeout = timeout
        self._enabled = enabled

        self._state = CircuitState.CLOSED
        self._failures: list[float] = []
        self._last_open_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if not self._enabled:
            return CircuitState.CLOSED
        return self._state

    def can_call(self) -> bool:
        """Check if a call is allowed through the circuit breaker."""
        if not self._enabled:
            return True

        if self._state == CircuitState.CLOSED:
            self._prune_failures()
            return True

        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_open_time >= self._timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("circuit %s → HALF_OPEN (probing)", self._name)
                return True
            return False

        # HALF_OPEN — allow one probe
        return True

    def record_success(self) -> None:
        """Record a successful call."""
        if not self._enabled:
            return
        if self._state == CircuitState.HALF_OPEN:
            logger.info("circuit %s → CLOSED (recovered)", self._name)
        self._state = CircuitState.CLOSED
        self._failures.clear()

    def record_failure(self) -> None:
        """Record a failed call, may trip the breaker."""
        if not self._enabled:
            return

        now = time.monotonic()
        self._failures.append(now)
        self._prune_failures()

        if len(self._failures) >= self._threshold and self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._last_open_time = now
            logger.warning(
                "circuit %s → OPEN (%d failures in %ds)",
                self._name, len(self._failures), self._window,
            )

    def _prune_failures(self) -> None:
        now = time.monotonic()
        cutoff = now - self._window
        self._failures = [f for f in self._failures if f > cutoff]
