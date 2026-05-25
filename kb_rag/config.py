"""KB RAG — 配置常量、环境变量、日志系统。"""

import os
import logging
import threading

from dotenv import load_dotenv
load_dotenv()

# ── 项目根目录 ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 日志 ──────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "log", "kb_rag.log")
LOG_INTERVAL = 10

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
logger = logging.getLogger("kb_rag")

# ── LLM ────────────────────────────────────────────────────────────────

from typing import Literal  # noqa: E402
LLM_PROVIDER: Literal["deepseek", "ollama"] = "deepseek"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

MAX_REVIEW_RETRY = 2

# ── 知识库路径 ─────────────────────────────────────────────────────────

KB_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "kb")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "kb_chroma_db")
SUMMARY_CACHE_PATH = os.path.join(PROJECT_ROOT, ".kb_summaries.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── 支持的知识库文档格式 ───────────────────────────────────────────────

_SUPPORTED_EXTS: set[str] = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".pdf", ".docx", ".doc",
}
