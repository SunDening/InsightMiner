"""Chat API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from insight_miner.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    MessageItem,
    MindMapResponse,
)
from insight_miner.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def get_chat_service() -> ChatService:
    raise NotImplementedError("Overridden in main.py")


@router.post("")
async def chat(
    req: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    logger.info("chat question=%.50s thread=%s kb=%s", req.question, req.thread_id, req.kb_id)
    return await chat_service.chat(
        question=req.question,
        thread_id=req.thread_id,
        kb_id=req.kb_id,
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    logger.info("stream question=%.50s thread=%s kb=%s", req.question, req.thread_id, req.kb_id)
    generator = chat_service.stream_chat(
        question=req.question,
        thread_id=req.thread_id,
        kb_id=req.kb_id,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{thread_id}/mindmap")
async def generate_mindmap(
    thread_id: str,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> MindMapResponse:
    logger.info("mindmap thread=%s", thread_id)
    try:
        data = await chat_service.generate_mindmap(thread_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return MindMapResponse(**data)
    except Exception as e:
        logger.error("mindmap pydantic validation error: %s | data=%.200s", e, str(data))
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=f"Mind map data validation failed: {e}")


@router.get("/history")
async def list_conversations(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    kb_id: str | None = None,
) -> list[ConversationSummary]:
    threads = await chat_service.list_threads(kb_id)
    logger.info("list_conversations kb=%s count=%d", kb_id, len(threads))
    return [ConversationSummary(**t) for t in threads]


@router.get("/history/{thread_id}")
async def get_conversation(
    thread_id: str,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> list[MessageItem]:
    messages = await chat_service.get_history(thread_id)
    logger.info("get_conversation thread=%s messages=%d", thread_id, len(messages))
    return [MessageItem(**m) for m in messages]


@router.delete("/history/{thread_id}")
async def delete_conversation(
    thread_id: str,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    await chat_service.delete_thread(thread_id)
    logger.info("delete_conversation thread=%s", thread_id)
    return {"ok": True}
