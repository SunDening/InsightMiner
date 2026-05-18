"""辅助函数：JSON 提取、Markdown 清理、SQL 清理、对话历史格式化。"""

import re
import json

from langchain.messages import HumanMessage, AIMessage

from .config import logger


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON 字符串（去掉可能的 Markdown 包裹）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


def _clean_sql_output(raw: str) -> str:
    """清理 LLM 输出中的 Markdown 标记和多余空白。"""
    raw = re.sub(r"^```(?:sql)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return raw.strip().rstrip(";")


def _sanitize_text(text: str) -> str:
    """移除 Markdown 格式符号，转为纯文本。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-| :]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_json_response(raw: str, question: str) -> str:
    """将 LLM 输出解析为规范的 JSON 回答字符串。解析失败则包裹为 fallback JSON。"""
    json_str = _extract_json(raw)
    try:
        parsed = json.loads(json_str)
        parsed.setdefault("answer", "")
        parsed.setdefault("rewritten_query", "")
        parsed.setdefault("data_sources", [])
        parsed["answer"] = _sanitize_text(parsed["answer"])
        parsed["rewritten_query"] = _sanitize_text(parsed["rewritten_query"])
        for src in parsed.get("data_sources", []):
            src.setdefault("source_type", "llm_knowledge")
            src.setdefault("evidence", "")
            src["evidence"] = _sanitize_text(src["evidence"])
            src.pop("method", None)
            src.pop("description", None)
            src.pop("analysis", None)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        logger.warning("[Parse] JSON 解析失败，使用 fallback 格式")
        return json.dumps({
            "answer": raw[:2000],
            "rewritten_query": "",
            "data_sources": [{
                "source_type": "llm_knowledge",
                "evidence": raw[:500]
            }]
        }, ensure_ascii=False, indent=2)


def _format_history(messages: list, max_pairs: int = 6) -> str:
    """将最近 N 条消息格式化为对话历史文本。"""
    if not messages:
        return "（无历史对话）"
    recent = messages[-max_pairs:]
    lines = []
    for msg in recent:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)
