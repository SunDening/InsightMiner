"""PGVectorStore — vector storage and similarity search via PGvector."""

from __future__ import annotations

import logging
from typing import Any

from insight_miner.config import VECTOR_DIMENSION
from insight_miner.core.store.database import DatabasePool

logger = logging.getLogger(__name__)


class PGVectorStore:
    """Stores and queries embeddings in PostgreSQL via the pgvector extension.

    Uses L2 distance (<->). All operations are async via asyncpg.
    """

    async def add_texts(
        self,
        kb_id: str,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert chunk embeddings."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            logger.warning("PGVectorStore: no database pool, skipping insert")
            return

        async with pool.acquire() as conn:
            # Use parameterized batch insert
            values = []
            for i, cid in enumerate(ids):
                emb = embeddings[i]
                text = texts[i] if i < len(texts) else ""
                values.append((cid, kb_id, text, emb))

            await conn.executemany(
                "INSERT INTO chunk_embedding (chunk_id, kb_id, content, embedding) "
                "VALUES ($1, $2, $3, $4::vector) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "  content = EXCLUDED.content, embedding = EXCLUDED.embedding",
                values,
            )
        logger.info("PGVectorStore: inserted %d chunks for kb=%s", len(ids), kb_id)

    async def similarity_search(
        self,
        kb_id: str,
        embedding: list[float],
        k: int = 20,
    ) -> list[tuple[int, float]]:
        """Search by L2 distance. Returns [(chunk_index_in_kb, score)], score = 1/(1+L2)."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chunk_id, 1.0 / (1.0 + (embedding <-> $1::vector)) AS score "
                "FROM chunk_embedding "
                "WHERE kb_id = $2 "
                "ORDER BY embedding <-> $1::vector "
                "LIMIT $3",
                embedding, kb_id, k,
            )

        # Return as list of (index_in_chunk_texts, score) for compatibility
        # Caller maps chunk_id to internal index via chunk_ids
        result: list[tuple[str, float]] = [(r["chunk_id"], float(r["score"])) for r in rows]
        return result  # type: ignore[return-value]

    async def delete_by_filename(self, kb_id: str, filename: str) -> None:
        """Remove all chunk embeddings for a given document."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            # chunk_id format: {doc_id}_{idx:06d}, need to match by doc_id prefix
            # doc_id = md5(filename)[:8]
            import hashlib
            doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
            await conn.execute(
                "DELETE FROM chunk_embedding WHERE kb_id = $1 AND chunk_id LIKE $2",
                kb_id, f"{doc_id}%",
            )
        logger.info("PGVectorStore: deleted chunks for kb=%s file=%s", kb_id, filename)

    async def delete_by_kb(self, kb_id: str) -> None:
        """Remove all chunk embeddings for a knowledge base."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM chunk_embedding WHERE kb_id = $1", kb_id,
            )
        logger.info("PGVectorStore: deleted all chunks for kb=%s", kb_id)
