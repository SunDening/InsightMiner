"""GraphChannel — entity‑graph based retrieval via NetworkX."""

from __future__ import annotations

import logging

from insight_miner.core.retrieval.search_channel import SearchChannel
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult
from insight_miner.core.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)


class GraphChannel(SearchChannel):
    """Entity graph traversal retrieval."""

    @property
    def name(self) -> str:
        return "graph"

    @property
    def priority(self) -> int:
        return 3

    async def search(self, context: SearchContext) -> SearchChannelResult:
        kb = context.kb_index
        if kb is None:
            return SearchChannelResult(channel_name=self.name, chunks=[])

        results = kb.graph_search(
            context.query,
            k=context.graph_top_k,
            query_entities=context.entities or None,
        )
        logger.info(
            "graph_channel query=%.50s hits=%d entities=%s",
            context.query, len(results), context.entities,
        )
        return SearchChannelResult(channel_name=self.name, chunks=results)
