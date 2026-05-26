"""Knowledge base management — CRUD for KBs and documents."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from insight_miner.config import (
    SUPPORTED_EXTS,
    get_chroma_dir,
    get_docs_dir,
    get_kb_dir,
    get_manifest_path,
)
from insight_miner.core.document_processor import KnowledgeBaseIndex


class KnowledgeBaseManager:
    """Manages multiple knowledge bases and their document lifecycles.

    Single-user mode: kb_id defaults to "default".
    Multi-user extension: kb_id becomes user_{uid}_kb_{name}.
    """

    def __init__(self):
        self._indices: dict[str, KnowledgeBaseIndex] = {}

    def _get_index(self, kb_id: str) -> KnowledgeBaseIndex:
        if kb_id not in self._indices:
            idx = KnowledgeBaseIndex(kb_id)
            idx.load_models()
            idx.initialize()
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
            return False
        path.mkdir(parents=True, exist_ok=True)
        (path / "documents").mkdir(exist_ok=True)
        return True

    def delete_knowledge_base(self, kb_id: str):
        if kb_id in self._indices:
            self._indices.pop(kb_id)
        path = get_kb_dir(kb_id)
        if path.exists():
            shutil.rmtree(str(path))

    # ── Document management ──

    async def upload_document(self, kb_id: str, filename: str, content: bytes) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
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

        # Initialize index BEFORE saving file (avoids double-indexing in full_rebuild)
        try:
            idx = self._get_index(kb_id)
        except Exception as e:
            return {"success": False, "error": f"Index init failed: {e}"}

        fpath.write_bytes(content)

        try:
            success = idx.add_document(fpath.name)
            if success:
                idx.finalize()
            return {
                "success": success,
                "filename": fpath.name,
                "size_bytes": len(content),
                "status": "indexed" if success else "error",
            }
        except Exception as e:
            if fpath.exists():
                fpath.unlink()
            return {"success": False, "error": str(e)}

    async def delete_document(self, kb_id: str, filename: str) -> bool:
        docs_dir = get_docs_dir(kb_id)
        fpath = docs_dir / filename
        if not fpath.exists():
            return False

        try:
            idx = self._get_index(kb_id)
            idx.remove_document(filename)
            idx.finalize()
        except Exception:
            pass

        fpath.unlink(missing_ok=True)
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

    def get_index(self, kb_id: str) -> KnowledgeBaseIndex:
        return self._get_index(kb_id)

    def shutdown(self):
        """Persist all dirty indices."""
        for idx in self._indices.values():
            idx.finalize()
