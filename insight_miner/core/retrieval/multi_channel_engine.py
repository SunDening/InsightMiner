"""MultiChannelEngine — orchestrates parallel retrieval and post-processing.

Reference: ragent MultiChannelRetrievalEngine pattern.

Flow:
  1. Build SearchContext
  2. Run enabled channels in parallel  (asyncio.gather)
  3. RRF fusion across channel results
  4. Run post-processor chain        (dedup → rerank)
  5. Return final chunks
"""

from __future__ import annotations

import logging

from insight_miner.core.document_processor import KnowledgeBaseIndex
from insight_miner.core.retrieval.search_channel import SearchChannel
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult
from insight_miner.core.retrieval.search_context import SearchContext
from insight_miner.core.retrieval.post_processors.interface import SearchPostProcessor
from insight_miner.core.retrieval.post_processors import DedupProcessor, RerankProcessor
from insight_miner.core.retrieval.channels import DenseChannel, BM25Channel, GraphChannel

logger = logging.getLogger(__name__)


class MultiChannelEngine:
    """Orchestrates multiple retrieval channels and result post-processing."""

    def __init__(
        self,
        channels: list[SearchChannel] | None = None,
        processors: list[SearchPostProcessor] | None = None,
    ):
        self._channels = channels or self._default_channels()
        self._processors = processors or self._default_processors()

    # ── Public API ──

    async def retrieve(self, context: SearchContext) -> list[tuple[int, float]]:
        """Full pipeline: parallel channels → RRF fusion → post-processors."""
        # Stage 1: parallel search
        channel_results = await self._run_channels(context)
        if not channel_results:
            return []

        # Stage 2: RRF fusion across channels
        fused = self._rrf_fusion(channel_results, k=context.rrf_k)
        top_candidates = fused[: context.rrf_candidates]

        logger.info(
            "multi_channel channels=%d total_raw=%d fused=%d top=%d",
            len(channel_results),
            sum(len(r.chunks) for r in channel_results),
            len(fused),
            len(top_candidates),
        )

        # Stage 3: post-processor chain
        chunks = top_candidates
        for proc in sorted(self._processors, key=lambda p: p.order):
            chunks = await proc.process(chunks, context)

        return chunks

    # ── Channel management ──

    def add_channel(self, channel: SearchChannel) -> None:
        self._channels.append(channel)

    def add_processor(self, processor: SearchPostProcessor) -> None:
        self._processors.append(processor)

    # ── Internal ──

    async def _run_channels(self, context: SearchContext) -> list[SearchChannelResult]:
        import asyncio

        enabled = [ch for ch in self._channels if ch.is_enabled(context)]
        if not enabled:
            return []

        tasks = [ch.search(context) for ch in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[SearchChannelResult] = []
        for ch, res in zip(enabled, results):
            if isinstance(res, Exception):
                logger.error("channel %s failed: %s", ch.name, res)
            elif isinstance(res, SearchChannelResult):
                valid.append(res)
        return valid

    @staticmethod
    def _rrf_fusion(
        results: list[SearchChannelResult],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion across multiple channel results."""
        scores: dict[int, float] = {}
        for result in results:
            for rank, (idx, _) in enumerate(result.chunks):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    @staticmethod
    def _default_channels() -> list[SearchChannel]:
        return [DenseChannel(), BM25Channel(), GraphChannel()]

    @staticmethod
    def _default_processors() -> list[SearchPostProcessor]:
        return [DedupProcessor(), RerankProcessor()]
