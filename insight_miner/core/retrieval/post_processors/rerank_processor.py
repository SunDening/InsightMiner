"""RerankProcessor — re-rank top candidates via CrossEncoder."""

from __future__ import annotations

import logging

from insight_miner.core.retrieval.post_processors.interface import SearchPostProcessor
from insight_miner.core.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)


class RerankProcessor(SearchPostProcessor):
    """Cross-encoder re-ranking on the merged chunk list."""

    @property
    def name(self) -> str:
        return "rerank"

    @property
    def order(self) -> int:
        return 10  # Run after dedup

    async def process(
        self,
        chunks: list[tuple[int, float]],
        context: SearchContext,
    ) -> list[tuple[int, float]]:
        kb = context.kb_index
        if kb is None or not chunks:
            return chunks

        # Get chunk texts for reranking
        if not hasattr(kb, "chunk_texts") or not kb.chunk_texts:
            return chunks

        doc_texts = [kb.chunk_texts[idx] for idx, _ in chunks if idx < len(kb.chunk_texts)]
        if not doc_texts:
            return chunks

        reranked = kb.rerank(context.query, doc_texts, top_k=len(doc_texts))
        # Map reranked texts back to chunk indices
        score_map = {kb.chunk_texts[idx]: (idx, score) for idx, score in chunks if idx < len(kb.chunk_texts)}
        result: list[tuple[int, float]] = []
        for text, score in reranked:
            if text in score_map:
                result.append((score_map[text][0], score))

        # Dynamic threshold: keep items with score >= top_score * 0.7, at least 1, max 5
        if result:
            top_score = result[0][1]
            threshold = top_score * 0.7
            result = [r for r in result if r[1] >= threshold][:5]

        logger.info(
            "rerank %d → %d chunks top_score=%.3f", len(chunks), len(result),
            result[0][1] if result else 0,
        )
        return result
