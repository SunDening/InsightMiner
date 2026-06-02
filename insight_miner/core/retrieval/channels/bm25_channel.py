"""BM25Channel — keyword sparse retrieval via BM25."""

from __future__ import annotations

import logging

from insight_miner.core.retrieval.search_channel import SearchChannel
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult
from insight_miner.core.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)


class BM25Channel(SearchChannel):
    """Keyword-based retrieval using BM25 (sparse)."""

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def priority(self) -> int:
        return 2

    async def search(self, context: SearchContext) -> SearchChannelResult:
        kb = context.kb_index
        if kb is None:
            return SearchChannelResult(channel_name=self.name, chunks=[])

        results = kb.bm25_search(context.query, k=context.bm25_top_k)
        logger.info(
            "bm25_channel query=%.50s hits=%d", context.query, len(results),
        )
        return SearchChannelResult(channel_name=self.name, chunks=results)
