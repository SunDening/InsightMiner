"""PostgreSQL-persistent conversation history via StoreRepository."""

from __future__ import annotations

from insight_miner.core.store.repository import StoreRepository


class MemoryService:
    """Stores conversations in PostgreSQL, accessed via StoreRepository."""

    def __init__(self, store: StoreRepository | None = None):
        self._store = store or StoreRepository()

    async def get_thread_kb_id(self, thread_id: str) -> str | None:
        return await self._store.get_thread_kb_id(thread_id)

    async def create_thread(self, thread_id: str, kb_id: str = "default") -> bool:
        return await self._store.create_thread(thread_id, kb_id)

    async def save_message(self, thread_id: str, role: str, content: str):
        await self._store.save_message(thread_id, role, content)

    async def get_history(self, thread_id: str, limit: int = 50) -> list[dict]:
        return await self._store.get_history(thread_id, limit)

    async def list_threads(self, kb_id: str | None = None) -> list[dict]:
        return await self._store.list_threads(kb_id)

    async def delete_thread(self, thread_id: str):
        await self._store.delete_thread(thread_id)

    async def update_title(self, thread_id: str, title: str):
        await self._store.update_thread_title(thread_id, title)
