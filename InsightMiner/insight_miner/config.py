"""Configuration — paths are parameterized by kb_id for multi-tenant readiness."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # "deepseek" | "ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
MAX_REVIEW_RETRY = 2
RERANK_LOW_THRESHOLD = -2.0

SUPPORTED_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".pdf", ".docx", ".doc"}

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

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
    return get_chroma_dir(kb_id) / "bm25_index.pkl"


def get_graph_path(kb_id: str = "default") -> Path:
    return get_chroma_dir(kb_id) / "entity_graph.pkl"
