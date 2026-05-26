"""LLM factory — supports DeepSeek and Ollama (migrated from kb_rag/llm.py)."""

import os

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from insight_miner.config import LLM_PROVIDER, OLLAMA_MODEL


def create_llm(provider: str | None = None, temperature: float = 0.1):
    provider = provider or LLM_PROVIDER

    if provider == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)

    return ChatDeepSeek(
        model=os.getenv("LLM_MODEL_ID", "deepseek-v4-pro"),
        api_key=os.getenv("LLM_API_KEY", ""),
        api_base=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=temperature,
    )
