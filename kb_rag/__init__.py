"""KB RAG — 知识库 + 网络搜索 + 闲聊 智能问答系统。

从 Agentic RAG v2 裁剪而来，移除了 SQL 数据库路径。
"""

from .kb_rag import chat, clear_thread, get_history, RagService  # noqa: F401
