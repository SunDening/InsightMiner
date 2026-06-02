"""DegradationManager — controls service quality levels.

Levels:
  0 = Full pipeline (intent → rewrite → multi-channel → rerank → LLM)
  1 = No rerank (skip cross-encoder)
  2 = No LLM (return retrieved docs only)
  3 = Cache only (return cached/default response)
"""

from __future__ import annotations

import logging

from insight_miner.config import DEGRADATION_LEVEL

logger = logging.getLogger(__name__)


class DegradationManager:
    """Controls service quality level and provides helpers for each level."""

    def __init__(self, level: int = DEGRADATION_LEVEL) -> None:
        self._level = self._clamp(level)

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        self._level = self._clamp(value)
        logger.info("degradation level set to %d", self._level)

    @property
    def full_pipeline(self) -> bool:
        """Level 0: Full RAG pipeline."""
        return self._level <= 0

    @property
    def skip_rerank(self) -> bool:
        """Level 1+: Skip cross-encoder reranking."""
        return self._level >= 1

    @property
    def skip_llm(self) -> bool:
        """Level 2+: Skip LLM generation, return docs only."""
        return self._level >= 2

    @property
    def cache_only(self) -> bool:
        """Level 3+: Only return cached results."""
        return self._level >= 3

    def describe(self) -> str:
        descriptions = {
            0: "Full pipeline (intent → rewrite → retrieve → rerank → LLM)",
            1: "Reduced (skip rerank)",
            2: "Docs only (skip LLM generation)",
            3: "Cache only",
        }
        return descriptions.get(self._level, f"Unknown level {self._level}")

    @staticmethod
    def _clamp(level: int) -> int:
        return max(0, min(3, level))
