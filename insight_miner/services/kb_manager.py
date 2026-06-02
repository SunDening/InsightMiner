"""Knowledge base management — CRUD for KBs and documents."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from insight_miner.config import (
    RABBITMQ_QUEUE,
    SUPPORTED_EXTS,
    get_chroma_dir,
    get_docs_dir,
    get_kb_dir,
    get_manifest_path,
)
from insight_miner.core.document_processor import KnowledgeBaseIndex
from insight_miner.core.ingestion.publisher import MessagePublisher
from insight_miner.core.ingestion.task_tracker import TaskTracker

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """Manages multiple knowledge bases and their document lifecycles.

    Single-user mode: kb_id defaults to "default".
    Multi-user extension: kb_id becomes user_{uid}_kb_{name}.
    """

    def __init__(self):
        self._indices: dict[str, KnowledgeBaseIndex] = {}
        self._publisher = MessagePublisher()
        self._task_tracker = TaskTracker()

    async def _get_index(self, kb_id: str) -> KnowledgeBaseIndex:
        if kb_id not in self._indices:
            logger.info("Initializing index for kb=%s", kb_id)
            idx = KnowledgeBaseIndex(kb_id)
            idx.load_models()
            await idx.initialize()
            self._indices[kb_id] = idx
        return self._indices[kb_id]

    # ── KB management ──

    def list_knowledge_bases(self) -> list[dict]:
        kb_dir = get_kb_dir()
        parent = kb_dir.parent
        if not parent.exists():
            return []
        result = []
        for d in sorted(parent.iterdir()):
            if d.is_dir():
                docs_dir = d / "documents"
                doc_count = len(
                    [f for f in docs_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
                ) if docs_dir.exists() else 0
                created = datetime.fromtimestamp(d.stat().st_ctime, tz=timezone.utc).isoformat() if hasattr(d.stat(), 'st_ctime') else None
                result.append({
                    "kb_id": d.name,
                    "document_count": doc_count,
                    "created_at": created,
                })
        return result

    def create_knowledge_base(self, kb_id: str) -> bool:
        path = get_kb_dir(kb_id)
        if path.exists():
            logger.warning("create_knowledge_base already_exists kb=%s", kb_id)
            return False
        path.mkdir(parents=True, exist_ok=True)
        (path / "documents").mkdir(exist_ok=True)
        logger.info("create_knowledge_base kb=%s", kb_id)
        return True

    def delete_knowledge_base(self, kb_id: str):
        if kb_id in self._indices:
            self._indices.pop(kb_id)
        path = get_kb_dir(kb_id)
        if path.exists():
            shutil.rmtree(str(path))
            logger.info("delete_knowledge_base kb=%s", kb_id)

    # ── Document management ──

    async def upload_document(self, kb_id: str, filename: str, content: bytes) -> dict:
        """Upload a document and index it synchronously (legacy path)."""
        return await self._do_upload(kb_id, filename, content, sync=True)

    async def upload_document_async(self, kb_id: str, filename: str, content: bytes) -> dict:
        """Upload a document asynchronously via RabbitMQ. Returns immediately with task_id."""
        return await self._do_upload(kb_id, filename, content, sync=False)

    async def _do_upload(self, kb_id: str, filename: str, content: bytes, sync: bool) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            logger.warning("upload unsupported_ext kb=%s file=%s ext=%s", kb_id, filename, ext)
            return {"success": False, "error": f"Unsupported file type: {ext}"}

        docs_dir = get_docs_dir(kb_id)
        docs_dir.mkdir(parents=True, exist_ok=True)
        fpath = docs_dir / filename

        # Avoid name collision
        if fpath.exists():
            stem = fpath.stem
            suffix = fpath.suffix
            counter = 1
            while fpath.exists():
                fpath = docs_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        fpath.write_bytes(content)

        if sync:
            # ── Synchronous (legacy) path ──
            try:
                idx = await self._get_index(kb_id)
            except Exception as e:
                logger.error("upload index_init_fail kb=%s file=%s error=%s", kb_id, filename, e)
                if fpath.exists():
                    fpath.unlink()
                return {"success": False, "error": f"Index init failed: {e}"}
            try:
                success = await idx.add_document(fpath.name)
                if success:
                    idx.finalize()
                status = "indexed" if success else "error"
                logger.info("upload sync kb=%s file=%s status=%s", kb_id, fpath.name, status)
                return {"success": success, "filename": fpath.name, "size_bytes": len(content), "status": status}
            except Exception as e:
                if fpath.exists():
                    fpath.unlink()
                logger.error("upload exception kb=%s file=%s error=%s", kb_id, fpath.name, e)
                return {"success": False, "error": str(e)}
        else:
            # ── Async path: publish to RabbitMQ ──
            try:
                task_id = await self._task_tracker.create_task(kb_id, fpath.name)
                ok = await self._publisher.publish(RABBITMQ_QUEUE, {
                    "task_id": task_id,
                    "kb_id": kb_id,
                    "filename": fpath.name,
                })
                if ok:
                    logger.info("upload_async kb=%s file=%s task=%s", kb_id, fpath.name, task_id)
                    return {
                        "success": True,
                        "filename": fpath.name,
                        "size_bytes": len(content),
                        "status": "pending",
                        "task_id": task_id,
                    }
                # Fallback: sync if publish failed
                logger.warning("upload_async publish_fail, falling back to sync kb=%s file=%s", kb_id, fpath.name)
                idx = await self._get_index(kb_id)
                success = await idx.add_document(fpath.name)
                if success:
                    idx.finalize()
                return {"success": success, "filename": fpath.name, "size_bytes": len(content),
                        "status": "indexed" if success else "error"}
            except Exception as e:
                logger.error("upload_async exception kb=%s file=%s error=%s", kb_id, fpath.name, e)
                return {"success": False, "error": str(e)}

    async def get_task_status(self, task_id: str) -> dict | None:
        """Get the status of an async ingestion task."""
        return await self._task_tracker.get_task(task_id)

    async def delete_document(self, kb_id: str, filename: str) -> bool:
        docs_dir = get_docs_dir(kb_id)
        fpath = docs_dir / filename
        if not fpath.exists():
            logger.warning("delete_document not_found kb=%s file=%s", kb_id, filename)
            return False

        try:
            idx = await self._get_index(kb_id)
            await idx.remove_document(filename)
            idx.finalize()
        except Exception as e:
            logger.warning("delete_document index_error kb=%s file=%s error=%s", kb_id, filename, e)

        fpath.unlink(missing_ok=True)
        logger.info("delete_document ok kb=%s file=%s", kb_id, filename)
        return True

    def list_documents(self, kb_id: str) -> list[dict]:
        docs_dir = get_docs_dir(kb_id)
        if not docs_dir.exists():
            return []

        # Get indexed filenames from manifest
        manifest_path = get_manifest_path(kb_id)
        indexed_files: set[str] = set()
        if manifest_path.exists():
            import json
            try:
                manifest = json.loads(manifest_path.read_text())
                indexed_files = set(manifest.get("files", {}).keys())
            except (json.JSONDecodeError, KeyError):
                pass

        result = []
        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix.lower() in SUPPORTED_EXTS:
                status = "indexed" if fpath.name in indexed_files else "pending"
                created = datetime.fromtimestamp(fpath.stat().st_ctime, tz=timezone.utc).isoformat() if hasattr(fpath.stat(), 'st_ctime') else None
                result.append({
                    "filename": fpath.name,
                    "size_bytes": fpath.stat().st_size,
                    "status": status,
                    "created_at": created,
                })
        return result

    # ── RAG engine access ──

    async def get_index(self, kb_id: str) -> KnowledgeBaseIndex:
        return await self._get_index(kb_id)

    def shutdown(self):
        """Persist all dirty indices."""
        logger.info("Shutting down %d indices", len(self._indices))
        for idx in self._indices.values():
            idx.finalize()
