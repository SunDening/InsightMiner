"""DedupProcessor — remove duplicate chunks, keep highest score."""

from __future__ import annotations

import logging

from insight_miner.core.retrieval.post_processors.interface import SearchPostProcessor
from insight_miner.core.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)


class DedupProcessor(SearchPostProcessor):
    """Deduplicate chunks by index, keeping the highest score per chunk."""

    @property
    def name(self) -> str:
        return "dedup"

    @property
    def order(self) -> int:
        return 1  # Run first

    async def process(
        self,
        chunks: list[tuple[int, float]],
        context: SearchContext,
    ) -> list[tuple[int, float]]:
        seen: dict[int, float] = {}
        for idx, score in chunks:
            if idx not in seen or score > seen[idx]:
                seen[idx] = score

        result = sorted(seen.items(), key=lambda x: -x[1])
        logger.info(
            "dedup %d → %d chunks", len(chunks), len(result),
        )
        return result
