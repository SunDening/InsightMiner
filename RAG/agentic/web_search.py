"""Web 搜索 — Tavily 封装。"""

import json

from langchain_tavily import TavilySearch

_tavily_tool = None


def _get_tavily():
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = TavilySearch(max_results=5, topic="general")
    return _tavily_tool


def search_web(question: str) -> str:
    """使用 Tavily 搜索网络。"""
    try:
        tool = _get_tavily()
        result = tool.invoke(question)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"[Web 搜索失败] {e}"
