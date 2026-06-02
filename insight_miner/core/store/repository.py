"""StoreRepository — CRUD operations for all stored entities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from insight_miner.core.store.database import DatabasePool

logger = logging.getLogger(__name__)


class StoreRepository:
    """High-level data access layer over PostgreSQL."""

    def __init__(self) -> None:
        self._pool_available = False

    async def ensure_ready(self) -> bool:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return False
        await DatabasePool.ensure_schema()
        self._pool_available = True
        return True

    # ── KB ──

    async def create_kb(self, kb_id: str, name: str = "", description: str = "") -> bool:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO kb (id, name, description) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                    kb_id, name, description,
                )
                return True
            except Exception as e:
                logger.warning("create_kb error: %s", e)
                return False

    async def list_kbs(self) -> list[dict]:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM kb ORDER BY created_at DESC")
            return [dict(r) for r in rows]

    # ── Document ──

    async def add_document(self, kb_id: str, filename: str, size_bytes: int, hash: str = "") -> int | None:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO document (kb_id, filename, size_bytes, hash)
                   VALUES ($1, $2, $3, $4)
                   RETURNING id""",
                kb_id, filename, size_bytes, hash,
            )
            return row["id"] if row else None

    async def update_document_status(self, doc_id: int, status: str):
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE document SET status=$1, updated_at=NOW() WHERE id=$2",
                status, doc_id,
            )

    async def list_documents(self, kb_id: str) -> list[dict]:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document WHERE kb_id=$1 ORDER BY created_at DESC",
                kb_id,
            )
            return [dict(r) for r in rows]

    # ── Thread / Message ──

    async def get_thread_kb_id(self, thread_id: str) -> str | None:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT kb_id FROM thread WHERE id=$1", thread_id)
            return row["kb_id"] if row else None

    async def create_thread(self, thread_id: str, kb_id: str = "default") -> bool:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO thread (id, kb_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    thread_id, kb_id,
                )
                return True
            except Exception:
                return False

    async def save_message(self, thread_id: str, role: str, content: str, tokens: int = 0):
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO message (thread_id, role, content, tokens) VALUES ($1, $2, $3, $4)",
                thread_id, role, content[:10000], tokens,
            )
            await conn.execute(
                "UPDATE thread SET message_count = message_count + 1, updated_at = NOW() WHERE id = $1",
                thread_id,
            )

    async def get_history(self, thread_id: str, limit: int = 50) -> list[dict]:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, created_at AS timestamp FROM message WHERE thread_id=$1 ORDER BY id ASC LIMIT $2",
                thread_id, limit,
            )
            return [dict(r) for r in rows]

    async def list_threads(self, kb_id: str | None = None) -> list[dict]:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            if kb_id:
                rows = await conn.fetch(
                    "SELECT id AS thread_id, title, message_count, updated_at FROM thread WHERE kb_id=$1 ORDER BY updated_at DESC",
                    kb_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT id AS thread_id, title, message_count, updated_at FROM thread ORDER BY updated_at DESC",
                )
            return [dict(r) for r in rows]

    async def delete_thread(self, thread_id: str) -> None:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM message WHERE thread_id=$1", thread_id)
            await conn.execute("DELETE FROM thread WHERE id=$1", thread_id)

    async def update_thread_title(self, thread_id: str, title: str) -> None:
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE thread SET title=$1, updated_at=NOW() WHERE id=$2",
                title[:200], thread_id,
            )

    # ── Audit ──

    async def audit_log(self, actor: str, action: str, resource: str, detail: str = ""):
        pool = await DatabasePool.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (actor, action, resource, detail) VALUES ($1, $2, $3, $4)",
                actor, action, resource, detail,
            )
