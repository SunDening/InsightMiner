"""DatabasePool — async PostgreSQL connection pool management."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from insight_miner.config import PG_DATABASE_URL

logger = logging.getLogger(__name__)


class DatabasePool:
    """Singleton asyncpg connection pool."""

    _instance: asyncpg.Pool | None = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool | None:
        if cls._instance is None:
            try:
                cls._instance = await asyncpg.create_pool(
                    PG_DATABASE_URL,
                    min_size=2,
                    max_size=10,
                    command_timeout=30,
                )
                logger.info("PostgreSQL pool created: %s", PG_DATABASE_URL)
            except Exception as e:
                logger.warning("PostgreSQL unavailable, running without store: %s", e)
                return None
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            logger.info("PostgreSQL pool closed")

    @classmethod
    async def ensure_schema(cls) -> None:
        """Create tables if they don't exist."""
        pool = await cls.get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            # PGvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kb (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    tenant_id   TEXT NOT NULL DEFAULT 'default',
                    status      TEXT NOT NULL DEFAULT 'active',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS document (
                    id          SERIAL PRIMARY KEY,
                    kb_id       TEXT NOT NULL REFERENCES kb(id),
                    filename    TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    version     INT NOT NULL DEFAULT 1,
                    hash        TEXT NOT NULL DEFAULT '',
                    size_bytes  BIGINT NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS document_version (
                    id          SERIAL PRIMARY KEY,
                    document_id INT NOT NULL REFERENCES document(id),
                    version     INT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'archived',
                    changelog   TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS thread (
                    id          TEXT PRIMARY KEY,
                    kb_id       TEXT NOT NULL DEFAULT 'default',
                    tenant_id   TEXT NOT NULL DEFAULT 'default',
                    title       TEXT NOT NULL DEFAULT '',
                    message_count INT NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS message (
                    id          SERIAL PRIMARY KEY,
                    thread_id   TEXT NOT NULL REFERENCES thread(id),
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    tokens      INT NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id          SERIAL PRIMARY KEY,
                    actor       TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    resource    TEXT NOT NULL,
                    detail      TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS intent_node (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    kind        TEXT NOT NULL DEFAULT 'kb',
                    parent_id   TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS chunk_embedding (
                    chunk_id    TEXT PRIMARY KEY,
                    kb_id       TEXT NOT NULL,
                    content     TEXT NOT NULL DEFAULT '',
                    embedding   vector(384) NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_document_kb ON document(kb_id);
                CREATE INDEX IF NOT EXISTS idx_thread_kb ON thread(kb_id);
                CREATE INDEX IF NOT EXISTS idx_message_thread ON message(thread_id);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
                CREATE INDEX IF NOT EXISTS idx_chunk_embedding_kb ON chunk_embedding(kb_id);
            """)
            # HNSW index (add after data exists):
            # CREATE INDEX ON chunk_embedding USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);
            logger.info("PostgreSQL schema ensured")
