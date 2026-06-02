"""SearchChannel — abstract interface for a single retrieval strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from insight_miner.core.retrieval.search_context import SearchContext
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult


class SearchChannel(ABC):
    """A single retrieval strategy (dense, BM25, graph, …).

    Channels are registered in a MultiChannelEngine and run in parallel.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name for logging and metrics."""

    @property
    def priority(self) -> int:
        """Lower = higher priority (used by dedup)."""
        return 10

    def is_enabled(self, context: SearchContext) -> bool:
        """Whether this channel should run for the given context."""
        return True

    @abstractmethod
    async def search(self, context: SearchContext) -> SearchChannelResult:
        """Execute retrieval and return scored chunks."""
