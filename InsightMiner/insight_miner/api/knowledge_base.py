"""Knowledge base management API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from insight_miner.models.schemas import DocumentItem, KnowledgeBase
from insight_miner.services.kb_manager import KnowledgeBaseManager

router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge-base"])
logger = logging.getLogger(__name__)


def get_kb_manager() -> KnowledgeBaseManager:
    """Dependency — injected via main.py."""
    raise NotImplementedError("Overridden in main.py")


def _kb_manager_dep(kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)]):
    return kb_manager


@router.get("")
async def list_knowledge_bases(
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
) -> list[KnowledgeBase]:
    bases = kb_manager.list_knowledge_bases()
    logger.info("list_knowledge_bases count=%d", len(bases))
    return [KnowledgeBase(**b) for b in bases]


@router.post("")
async def create_knowledge_base(
    kb_id: str,
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
) -> KnowledgeBase:
    success = kb_manager.create_knowledge_base(kb_id)
    if not success:
        from fastapi import HTTPException
        logger.warning("create_knowledge_base conflict kb=%s", kb_id)
        raise HTTPException(status_code=409, detail=f"Knowledge base '{kb_id}' already exists")
    logger.info("create_knowledge_base kb=%s", kb_id)
    return KnowledgeBase(kb_id=kb_id, document_count=0)


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
):
    kb_manager.delete_knowledge_base(kb_id)
    logger.info("delete_knowledge_base kb=%s", kb_id)
    return {"ok": True}


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
) -> list[DocumentItem]:
    docs = kb_manager.list_documents(kb_id)
    logger.info("list_documents kb=%s count=%d", kb_id, len(docs))
    return [DocumentItem(**d) for d in docs]


@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: str,
    file: UploadFile,
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
) -> DocumentItem:
    content = await file.read()
    result = await kb_manager.upload_document(kb_id, file.filename or "upload", content)
    if not result.get("success"):
        from fastapi import HTTPException
        detail = result.get("error", "Upload failed")
        logger.warning("upload_document fail kb=%s file=%s error=%s", kb_id, file.filename, detail)
        raise HTTPException(status_code=400, detail=detail)
    logger.info("upload_document kb=%s file=%s size=%d", kb_id, result["filename"], result["size_bytes"])
    return DocumentItem(
        filename=result["filename"],
        size_bytes=result["size_bytes"],
        status=result["status"],
    )


@router.delete("/{kb_id}/documents/{filename:path}")
async def delete_document(
    kb_id: str,
    filename: str,
    kb_manager: Annotated[KnowledgeBaseManager, Depends(get_kb_manager)],
):
    success = await kb_manager.delete_document(kb_id, filename)
    if not success:
        from fastapi import HTTPException
        logger.warning("delete_document not_found kb=%s file=%s", kb_id, filename)
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    logger.info("delete_document kb=%s file=%s", kb_id, filename)
    return {"ok": True}
