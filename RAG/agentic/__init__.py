"""Agentic RAG v2 — 面向 Access .mdb 结构化数据库的智能问答系统（意图路由版）

公共 API：
  - chat(question, thread_id)   多轮对话入口
  - clear_thread(thread_id)     清空会话历史
  - get_history(thread_id)      获取会话历史
  - RagService                  服务类（高级用法）

使用：
  from agentic import chat

  answer = await chat("列出所有GSO卫星")
"""

import json
import asyncio
from datetime import datetime, timezone

# ⚠️ config 必须先于所有其他 agentic 模块导入，确保 load_dotenv() 在
# huggingface_hub / transformers 被任何模块 import 之前设置好离线环境变量
from .config import (
    logger, KB_DIR, CHROMA_DIR, SUMMARY_CACHE_PATH,
    SCHEMA_JSON_PATH, TABLE_DESC_JSON_PATH, SCHEMA_CHROMA_DIR,
    QUERY_MEMORY_PATH,
)

from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .knowledge_base import RagKnowledgeBase
from .schema_indexer import SchemaIndexer
from .query_memory import QueryMemory
from .entity_router import EntityRouter
from .graph_builder import build_agent_graph


class RagService:
    """RAG 服务单例，持有 KB 实例、SchemaIndexer 和 LangGraph agent。

    Usage:
        service = RagService()
        await service.initialize()
        answer = await service.chat("问题")
    """

    def __init__(self):
        self.kb: RagKnowledgeBase | None = None
        self.schema_indexer: SchemaIndexer | None = None
        self.query_memory: QueryMemory | None = None
        self.entity_router: EntityRouter | None = None
        self.agent: CompiledStateGraph | None = None
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """惰性初始化：加载模型、构建 KB + Schema 索引、编译图。线程安全。"""
        async with self._init_lock:
            if self.agent is not None:
                return  # 已初始化

            self.kb = RagKnowledgeBase(
                kb_dir=KB_DIR,
                chroma_dir=CHROMA_DIR,
                summary_cache_path=SUMMARY_CACHE_PATH,
            )
            await self.kb.initialize()

            # SchemaIndexer 复用 KB 已加载的 embeddings_model 和 reranker_model
            self.schema_indexer = SchemaIndexer(
                schema_json_path=SCHEMA_JSON_PATH,
                table_desc_path=TABLE_DESC_JSON_PATH,
                chroma_dir=SCHEMA_CHROMA_DIR,
                embeddings_model=self.kb.embeddings_model,
                reranker_model=self.kb.reranker_model,
            )
            await self.schema_indexer.initialize()

            # QueryMemory 复用 KB 的 embeddings_model
            self.query_memory = QueryMemory(
                persist_dir=QUERY_MEMORY_PATH,
                embeddings_model=self.kb.embeddings_model,
            )
            self.query_memory.initialize()

            # EntityRouter 无需模型，直接实例化
            self.entity_router = EntityRouter()

            self.agent = build_agent_graph(self.kb, self.schema_indexer,
                                           self.query_memory,
                                           self.entity_router)
            logger.info("[Service] RagService 初始化完成")

    async def chat(self, question: str, thread_id: str = "default") -> str:
        """多轮对话入口。返回 JSON 字符串。"""
        if self.agent is None:
            await self.initialize()

        config = {"configurable": {"thread_id": thread_id}}
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        input_state = {
            "question": question,
            "current_time": current_time,
            "intent": "",
            "rewritten_query": "",
            "schema_context": "",
            "sql": "",
            "query_result": "",
            "retry_count": 0,
            "error_info": "",
            "retrieved_docs": "",
            "kb_confidence": 0.0,
            "web_results": "",
            "final_answer": "",
            "review_count": 0,
            "review_feedback": "",
            "messages": [HumanMessage(content=question)],
        }

        final_state = await self.agent.ainvoke(input_state, config)
        return final_state.get("final_answer", json.dumps({
            "answer": "系统未能生成回答，请重试。",
            "rewritten_query": "",
            "data_sources": []
        }, ensure_ascii=False))

    async def clear_thread(self, thread_id: str) -> None:
        """清空指定会话的历史记录。"""
        if self.agent is None:
            return
        try:
            self.agent.checkpointer.delete_thread(thread_id)
            logger.info(f"[Session] 已清空会话 {thread_id}")
        except Exception as e:
            logger.warning(f"[Session] 清空会话失败: {e}")

    async def get_history(self, thread_id: str) -> list[dict]:
        """获取指定会话的对话历史。"""
        if self.agent is None:
            return []
        checkpoint = self.agent.checkpointer.get({"configurable": {"thread_id": thread_id}})
        if not checkpoint:
            return []
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        if not messages:
            return []
        result = []
        for msg in messages:
            if not msg.content:
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result


# ── 模块级单例 ─────────────────────────────────────────────────────────

_service: RagService | None = None
_service_lock = asyncio.Lock()


async def get_service() -> RagService:
    """获取或创建 RagService 单例。"""
    global _service
    if _service is not None:
        return _service
    async with _service_lock:
        if _service is not None:
            return _service
        _service = RagService()
        await _service.initialize()
        return _service


async def chat(question: str, thread_id: str = "default") -> str:
    """多轮对话入口（模块级便捷函数）。"""
    svc = await get_service()
    return await svc.chat(question, thread_id)


async def clear_thread(thread_id: str) -> None:
    """清空指定会话的历史记录（模块级便捷函数）。"""
    svc = await get_service()
    await svc.clear_thread(thread_id)


async def get_history(thread_id: str) -> list[dict]:
    """获取指定会话的对话历史（模块级便捷函数）。"""
    svc = await get_service()
    return await svc.get_history(thread_id)
