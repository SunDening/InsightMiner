"""Agentic RAG v2 — 配置常量、环境变量、日志系统。

面向 Access .mdb 数据库（ITU-R SNS），Schema 元数据由 SchemaIndexer 从 JSON 动态加载。
所有路径均相对于 RAG/ 项目根目录。
"""

import os
import re
import json
import logging
import threading
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

# ── 项目根目录 ────────────────────────────────────────────────────────
# config.py 位于 RAG/agentic/，上溯两级即 RAG/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 日志（每 10 秒批量写入文件，每次启动清空）──────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "agentic_rag.log")
LOG_INTERVAL = 10

# 启动时清空旧日志
with open(LOG_FILE, "w", encoding="utf-8") as _f:
    _f.write("")

_log_buffer: list[str] = []
_log_lock = threading.Lock()


def _flush_logs():
    with _log_lock:
        if _log_buffer:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(_log_buffer) + "\n")
            _log_buffer.clear()


class _TimedBufferingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_lock:
                _log_buffer.append(msg)
        except Exception:
            self.handleError(record)


def _start_log_timer():
    def _loop():
        while True:
            _flush_logs()
            threading.Event().wait(LOG_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


_start_log_timer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[_TimedBufferingHandler()],
)
logger = logging.getLogger("agentic_rag_v2")

# ── Access .mdb 数据库 ─────────────────────────────────────────────────

MDB_PATH = os.getenv("MDB_PATH", r"C:\Users\Sacr\Documents\test.mdb")

# ── Schema 元数据 JSON ─────────────────────────────────────────────────

SCHEMA_JSON_PATH = os.path.join(PROJECT_ROOT, "..", "tmp", "schema.json")
TABLE_DESC_JSON_PATH = os.path.join(PROJECT_ROOT, "..", "tmp", "table_desc.json")

# ── Schema 检索配置 ────────────────────────────────────────────────────

SCHEMA_CHROMA_DIR = os.path.join(PROJECT_ROOT, "schema_chroma_db")
MAX_SCHEMA_TABLES = 5
MAX_SCHEMA_COLUMNS_PER_TABLE = 5
GARBAGE_TABLES = {"名称自动更正保存失败", "表1"}

# ── LLM ────────────────────────────────────────────────────────────────

LLM_PROVIDER: Literal["deepseek", "ollama"] = "ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

MAX_SQL_RETRY = 3
MAX_REVIEW_RETRY = 2

# ── 知识库路径 ─────────────────────────────────────────────────────────

KB_DIR = os.path.join(PROJECT_ROOT, "kb")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
SUMMARY_CACHE_PATH = os.path.join(PROJECT_ROOT, ".kb_summaries.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── 支持的知识库文档格式 ───────────────────────────────────────────────

_SUPPORTED_EXTS: set[str] = {".txt", ".md", ".csv", ".json", ".yaml", ".pdf", ".docx", ".doc"}


# ── 危险 SQL 关键词黑名单 ──────────────────────────────────────────────

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "RENAME", "LOAD",
    "IMPORT", "EXEC", "EXECUTE", "CALL", "MERGE",
]


# （SCHEMA_DEFINITION 已移除，Schema 元数据改由 SchemaIndexer 从 JSON 文件动态加载）
