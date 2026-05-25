"""KB RAG — 知识库 + 网络搜索 + 闲聊 智能问答系统。

入口点，提供模块级便捷函数和 RagService 类。

支持两种运行方式：
  python -m kb_rag.kb_rag        （从项目根目录以模块运行）
  python kb_rag/kb_rag.py        （直接运行脚本）
"""

import json
import sys
import os
import asyncio
from datetime import datetime, timezone

# 直接运行脚本时的兼容处理
if __name__ == "__main__":
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _parent)

try:
    from .config import logger, KB_DIR, CHROMA_DIR, SUMMARY_CACHE_PATH
    from .knowledge_base import RagKnowledgeBase
    from .graph_builder import build_agent_graph
except ImportError:
    from kb_rag.config import logger, KB_DIR, CHROMA_DIR, SUMMARY_CACHE_PATH  # type: ignore
    from kb_rag.knowledge_base import RagKnowledgeBase  # type: ignore
    from kb_rag.graph_builder import build_agent_graph  # type: ignore

from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

try:
    from .watcher import KbDirWatcher as _KbDirWatcher
    _HAS_WATCHER = True
except ImportError:
    _KbDirWatcher = None  # type: ignore
    _HAS_WATCHER = False


class RagService:
    """RAG 服务单例，持有 KB 实例和 LangGraph agent。"""

    def __init__(self):
        self.kb: RagKnowledgeBase | None = None
        self.agent: CompiledStateGraph | None = None
        self._init_lock = asyncio.Lock()
        self._watcher: object | None = None

    async def initialize(self) -> None:
        """惰性初始化：加载模型、构建 KB、编译图。"""
        async with self._init_lock:
            if self.agent is not None:
                return

            self.kb = RagKnowledgeBase(
                kb_dir=KB_DIR,
                chroma_dir=CHROMA_DIR,
                summary_cache_path=SUMMARY_CACHE_PATH,
            )
            await self.kb.initialize()
            self.agent = build_agent_graph(self.kb)

            # 启动目录监控（增量更新）
            if _HAS_WATCHER:
                try:
                    self._watcher = _KbDirWatcher(
                        self.kb,
                        loop=asyncio.get_event_loop(),
                        kb_dir=self.kb.kb_dir,
                        debounce_sec=2.0,
                    )
                    self._watcher.start()
                except Exception as e:
                    logger.warning(f"[Service] 目录监控启动失败: {e}")
                    self._watcher = None

            logger.info("[Service] KB RAG 初始化完成")

    async def chat(self, question: str, thread_id: str = "default") -> str:
        """多轮对话入口。返回 JSON 字符串。"""
        if self.agent is None:
            await self.initialize()

        config = {"configurable": {"thread_id": thread_id}}
        current_time = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        input_state = {
            "question": question,
            "current_time": current_time,
            "intent": "",
            "rewritten_query": "",
            "retrieved_docs": "",
            "kb_confidence": 0.0,
            "web_results": "",
            "query_entities": [],
            "final_answer": "",
            "review_count": 0,
            "review_feedback": "",
            "messages": [HumanMessage(content=question)],
        }

        final_state = await self.agent.ainvoke(input_state, config)
        return final_state.get("final_answer", json.dumps({
            "answer": "系统未能生成回答，请重试。",
            "rewritten_query": "",
            "data_sources": [],
        }, ensure_ascii=False))

    async def shutdown(self) -> None:
        """释放资源，停止后台监控。"""
        if self._watcher is not None:
            try:
                self._watcher.stop()
            except Exception as e:
                logger.warning(f"[Service] 停止监控失败: {e}")
            self._watcher = None
        logger.info("[Service] 服务已关闭")

    async def clear_thread(self, thread_id: str) -> None:
        if self.agent is None:
            return
        try:
            self.agent.checkpointer.delete_thread(thread_id)
            logger.info(f"[Session] 已清空会话 {thread_id}")
        except Exception as e:
            logger.warning(f"[Session] 清空会话失败: {e}")

    async def get_history(self, thread_id: str) -> list[dict]:
        if self.agent is None:
            return []
        checkpoint = self.agent.checkpointer.get(
            {"configurable": {"thread_id": thread_id}}
        )
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
    svc = await get_service()
    return await svc.chat(question, thread_id)


async def clear_thread(thread_id: str) -> None:
    svc = await get_service()
    await svc.clear_thread(thread_id)


async def get_history(thread_id: str) -> list[dict]:
    svc = await get_service()
    return await svc.get_history(thread_id)


if __name__ == "__main__":
    try:
        from .cli import main
    except ImportError:
        from kb_rag.cli import main  # type: ignore  # noqa: E402
    asyncio.run(main())
