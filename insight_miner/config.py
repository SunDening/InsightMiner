"""Configuration — paths are parameterized by kb_id for multi-tenant readiness."""

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

# Intent tree
INTENT_CONFIG_PATH = CONFIG_DIR / "intent.yaml"

# RabbitMQ
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "insight_ingestion")
RABBITMQ_DLQ = os.getenv("RABBITMQ_DLQ", "insight_ingestion_dlq")

# PostgreSQL
PG_DATABASE_URL = os.getenv(
    "PG_DATABASE_URL",
    "postgresql://insight:insight@localhost:5432/insight",
)

# Ingestion
INGESTION_WORKER_CONCURRENCY = int(os.getenv("INGESTION_WORKER_CONCURRENCY", "4"))
INGESTION_TASK_TTL = 3600  # task status TTL in Redis (seconds)

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "deepseek" | "ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Rate Limiter
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_DEFAULT = int(os.getenv("RATE_LIMIT_DEFAULT", "60"))  # requests per minute
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))     # burst size

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))  # failures before open
CIRCUIT_BREAKER_WINDOW = int(os.getenv("CIRCUIT_BREAKER_WINDOW", "60"))       # sliding window (s)
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30"))     # half-open after (s)

# Degradation
DEGRADATION_LEVEL = int(os.getenv("DEGRADATION_LEVEL", "0"))  # 0=full, 1=no-rerank, 2=no-llm, 3=cache-only

# Metrics
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Vector store
USE_PGVECTOR = os.getenv("USE_PGVECTOR", "false").lower() == "true"
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))

# RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
MAX_REVIEW_RETRY = 2
RERANK_LOW_THRESHOLD = -2.0

SUPPORTED_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".pdf", ".docx", ".doc"}

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Cache / Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SHORT = 120       # L1 memory TTL (seconds)
CACHE_TTL_LONG = 3600       # L2 Redis TTL (seconds)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# ── Email ─────────────────────────────────────────────────────────────

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.qq.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "465"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")

# ── JWT ───────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

# Retrieval
DENSE_TOP_K = 20
BM25_TOP_K = 20
GRAPH_TOP_K = 40
RRF_FUSION_K = 60
RERANK_TOP_K = 3


def get_kb_dir(kb_id: str = "default") -> Path:
    return DATA_DIR / "knowledge_bases" / kb_id


def get_chroma_dir(kb_id: str = "default") -> Path:
    return get_kb_dir(kb_id) / "chroma_db"


def get_docs_dir(kb_id: str = "default") -> Path:
    return get_kb_dir(kb_id) / "documents"


def get_manifest_path(kb_id: str = "default") -> Path:
    return get_kb_dir(kb_id) / "manifest.json"


def get_bm25_path(kb_id: str = "default") -> Path:
    idx = get_kb_dir(kb_id) / "index"
    idx.mkdir(parents=True, exist_ok=True)
    return idx / "bm25_index.pkl"


def get_graph_path(kb_id: str = "default") -> Path:
    idx = get_kb_dir(kb_id) / "index"
    idx.mkdir(parents=True, exist_ok=True)
    return idx / "entity_graph.pkl"


# ── Logging ──────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    """Configure root logger for the application."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
