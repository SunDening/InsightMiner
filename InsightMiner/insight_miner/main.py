"""FastAPI application entry point."""

from __future__ import annotations

import os

# 强制离线：必须在任何模型导入之前设置
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from insight_miner.api import chat as chat_api
from insight_miner.api import knowledge_base as kb_api
from insight_miner.services.chat_service import ChatService
from insight_miner.services.kb_manager import KnowledgeBaseManager
from insight_miner.services.memory_service import MemoryService

load_dotenv()


# ── Services (application-scoped singletons) ──

memory = MemoryService()
kb_manager = KnowledgeBaseManager()
chat_service = ChatService(kb_manager, memory)


def _inject_deps(app: FastAPI):
    """Override placeholder dependencies in API routers with real instances."""

    async def _get_kb_manager():
        return kb_manager

    async def _get_chat_service():
        return chat_service

    # Copy the dependency overrides into the app's dependency graph
    app.dependency_overrides[kb_api.get_kb_manager] = _get_kb_manager
    app.dependency_overrides[chat_api.get_chat_service] = _get_chat_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    _inject_deps(app)
    yield
    # Shutdown: persist dirty indices
    kb_manager.shutdown()


app = FastAPI(
    title="InsightMiner",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kb_api.router)
app.include_router(chat_api.router)


@app.get("/api/system/health")
async def health():
    return {"status": "ok"}
