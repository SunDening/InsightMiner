"""SearchPostProcessor — abstract interface for result post-processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from insight_miner.core.retrieval.search_context import SearchContext


class SearchPostProcessor(ABC):
    """Post-processes merged results from all channels (dedup, rerank, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Processor name for logging."""

    @property
    def order(self) -> int:
        """Execution order (lower = runs first)."""
        return 100

    @abstractmethod
    async def process(
        self,
        chunks: list[tuple[int, float]],  # [(chunk_index, score), …]
        context: SearchContext,
    ) -> list[tuple[int, float]]:
        """Return processed chunk list."""
