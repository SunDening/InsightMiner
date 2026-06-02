"""DenseChannel — vector similarity search via ChromaDB."""

from __future__ import annotations

import logging

from insight_miner.core.retrieval.search_channel import SearchChannel
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult
from insight_miner.core.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)


class DenseChannel(SearchChannel):
    """Semantic search via ChromaDB embedding similarity."""

    @property
    def name(self) -> str:
        return "dense"

    @property
    def priority(self) -> int:
        return 1  # highest priority

    async def search(self, context: SearchContext) -> SearchChannelResult:
        kb = context.kb_index
        if kb is None:
            return SearchChannelResult(channel_name=self.name, chunks=[])

        results = kb.dense_search(context.query, k=context.dense_top_k)
        logger.info(
            "dense_channel query=%.50s hits=%d top_score=%.3f",
            context.query, len(results),
            results[0][1] if results else 0,
        )
        return SearchChannelResult(channel_name=self.name, chunks=results)
