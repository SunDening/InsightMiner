"""Pydantic models for API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Knowledge Base ──

class KnowledgeBase(BaseModel):
    kb_id: str
    document_count: int
    created_at: datetime | None = None


class DocumentItem(BaseModel):
    filename: str
    size_bytes: int
    status: Literal["indexed", "pending", "error"] = "pending"
    created_at: datetime | None = None


# ── Chat ──

class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    kb_id: str = "default"


class EvidenceItem(BaseModel):
    content: str = Field(description="证据原文片段")
    score: float = Field(description="CrossEncoder 原始分数")
    confidence_pct: float = Field(description="映射到 0-100 的百分比置信度")
    source_document: str = Field(description="来源文档文件名")
    chunk_index: int = 0


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    rewritten_query: str = ""
    evidences: list[EvidenceItem] = []
    kb_confidence: float = 0.0
    intent: str = "kb"


class ConversationSummary(BaseModel):
    thread_id: str
    title: str = ""
    message_count: int = 0
    updated_at: datetime | None = None


class MessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None
