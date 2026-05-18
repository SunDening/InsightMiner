"""
Agentic RAG v2 — 面向 MySQL 结构化数据库的智能问答系统（意图路由版）

实现已迁移至 RAG/agentic/ 包。
本文件保留为兼容性入口，所有功能通过 agentic 包提供。
"""

import sys
import os

# 确保 RAG/ 在 sys.path 中，使 from agentic import ... 可用
_rag_dir = os.path.dirname(os.path.abspath(__file__))
if _rag_dir not in sys.path:
    sys.path.insert(0, _rag_dir)

from agentic import chat, clear_thread, get_history, RagService  # noqa: E402
from agentic.cli import main  # noqa: E402

__all__ = ["chat", "clear_thread", "get_history", "main", "RagService"]

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
