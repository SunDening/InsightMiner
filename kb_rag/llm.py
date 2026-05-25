"""LLM 工厂 — DeepSeek / Ollama 切换。"""

import os
from typing import Literal

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from .config import LLM_PROVIDER, OLLAMA_MODEL


def create_llm(provider: Literal["deepseek", "ollama"] = LLM_PROVIDER,
               temperature: float = 0.1):
    if provider == "deepseek":
        return ChatDeepSeek(
            model=os.getenv("LLM_MODEL_ID", "deepseek-v4-pro"),
            api_key=os.getenv("LLM_API_KEY", ""),
            api_base=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            temperature=temperature,
        )
    elif provider == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
    else:
        raise ValueError(f"不支持的 provider: {provider}")
