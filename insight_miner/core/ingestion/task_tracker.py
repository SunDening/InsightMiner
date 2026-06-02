"""TaskTracker — tracks ingestion task status via Redis.

States:
  pending → processing → done | failed
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from insight_miner.config import CACHE_ENABLED, INGESTION_TASK_TTL, REDIS_URL

logger = logging.getLogger(__name__)


class TaskTracker:
    """Track ingestion task status in Redis."""

    def __init__(self) -> None:
        self._redis = None
        self._redis_url = REDIS_URL

    async def create_task(self, kb_id: str, filename: str) -> str:
        """Create a new task and return task_id."""
        task_id = uuid.uuid4().hex[:12]
        task = {
            "task_id": task_id,
            "kb_id": kb_id,
            "filename": filename,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": "",
        }
        redis = await self._get_redis()
        if redis:
            key = f"task:{task_id}"
            await redis.setex(key, INGESTION_TASK_TTL, json.dumps(task))
        return task_id

    async def update_status(
        self,
        task_id: str,
        status: str,
        error: str = "",
    ) -> None:
        redis = await self._get_redis()
        if not redis:
            return
        key = f"task:{task_id}"
        raw = await redis.get(key)
        if raw is None:
            return
        task = json.loads(raw)
        task["status"] = status
        task["updated_at"] = datetime.now(timezone.utc).isoformat()
        if error:
            task["error"] = error
        await redis.setex(key, INGESTION_TASK_TTL, json.dumps(task))

    async def get_task(self, task_id: str) -> dict | None:
        redis = await self._get_redis()
        if not redis:
            return None
        raw = await redis.get(f"task:{task_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def get_redis(self):
        return await self._get_redis()

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                await self._redis.ping()
            except Exception as e:
                logger.warning("Redis unavailable for task tracking: %s", e)
                self._redis = None
        return self._redis
