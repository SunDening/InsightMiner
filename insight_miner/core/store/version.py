"""Document version management — supports rollback and gray release."""

from __future__ import annotations

import logging

from insight_miner.core.store.database import DatabasePool

logger = logging.getLogger(__name__)


class DocumentVersionManager:
    """Manage document version history, enabling rollback."""

    async def create_version(self, doc_id: int, changelog: str = "") -> int | None:
        """Create a new version entry. Returns version number."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            # Get current version
            row = await conn.fetchrow(
                "SELECT version FROM document WHERE id=$1", doc_id,
            )
            if row is None:
                return None
            new_version = row["version"] + 1

            await conn.execute(
                "INSERT INTO document_version (document_id, version, changelog) VALUES ($1, $2, $3)",
                doc_id, new_version, changelog,
            )
            await conn.execute(
                "UPDATE document SET version=$1, updated_at=NOW() WHERE id=$2",
                new_version, doc_id,
            )
            return new_version

    async def get_version_history(self, doc_id: int) -> list[dict]:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_version WHERE document_id=$1 ORDER BY version DESC",
                doc_id,
            )
            return [dict(r) for r in rows]

    async def rollback_to(self, doc_id: int, target_version: int) -> bool:
        """Rollback a document to a previous version."""
        pool = await DatabasePool.get_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE document SET version=$1, updated_at=NOW() WHERE id=$2",
                target_version, doc_id,
            )
            logger.info("rollback doc_id=%d to v=%d", doc_id, target_version)
            return True
