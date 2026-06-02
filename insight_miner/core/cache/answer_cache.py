"""Multi-level answer cache with L1 (memory) and L2 (Redis).

Cache key scheme:
  answer:{kb_id}:{sha256(question)}  →  cached answer + evidences

Usage:
  cache = AnswerCache()
  await cache.get("default", "用户问题")
  await cache.set("default", "用户问题", {"answer": "...", ...})
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

from insight_miner.config import (
    CACHE_ENABLED,
    CACHE_TTL_SHORT,
    CACHE_TTL_LONG,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


class AnswerCache:
    """Two-level cache with automatic L1 ← L2 backfill."""

    def __init__(self) -> None:
        self._enabled = CACHE_ENABLED
        # L1: process-memory cache
        self._memory: TTLCache[str, dict] = TTLCache(
            maxsize=512,
            ttl=CACHE_TTL_SHORT,
        )
        # L2: Redis (lazy init)
        self._redis = None
        self._redis_url = REDIS_URL

    # ── Public API ──

    async def get(self, kb_id: str, question: str) -> dict | None:
        """Try L1 → L2. Returns cached result dict or None."""
        if not self._enabled:
            return None

        key = self._make_key(kb_id, question)

        # L1: memory
        hit = self._memory.get(key)
        if hit is not None:
            logger.debug("cache L1 hit kb=%s q=%.30s", kb_id, question)
            return hit

        # L2: Redis
        try:
            redis = await self._get_redis()
            if redis is not None:
                raw = await redis.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    # Backfill L1
                    self._memory[key] = data
                    logger.debug("cache L2 hit kb=%s q=%.30s", kb_id, question)
                    return data
        except Exception as e:
            logger.warning("cache L2 get failed: %s", e)

        return None

    async def set(
        self,
        kb_id: str,
        question: str,
        data: dict,
        ttl: int | None = None,
    ) -> None:
        """Write to L1 + L2."""
        if not self._enabled:
            return

        key = self._make_key(kb_id, question)
        ttl = ttl or CACHE_TTL_LONG

        # L1
        self._memory[key] = data

        # L2
        try:
            redis = await self._get_redis()
            if redis is not None:
                await redis.setex(key, ttl, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.warning("cache L2 set failed: %s", e)

    async def invalidate(self, kb_id: str) -> None:
        """Invalidate all cache entries for a KB (on doc update)."""
        # L1: clear all (simple approach)
        self._memory.clear()
        # L2: pattern delete would be expensive; skip for now
        logger.info("cache invalidated for kb=%s", kb_id)

    # ── Internal ──

    @staticmethod
    def _make_key(kb_id: str, question: str) -> str:
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
        return f"answer:{kb_id}:{q_hash}"

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Ping to verify connectivity
                await self._redis.ping()
                logger.info("Redis connected: %s", self._redis_url)
            except Exception as e:
                logger.warning("Redis unavailable, running without cache: %s", e)
                self._redis = None  # Don't retry on every call
        return self._redis
