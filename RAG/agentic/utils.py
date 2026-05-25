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
    """清理 LLM 输出中的 Markdown 标记和多余空白，并修复常见 Access SQL 兼容性问题。"""
    raw = re.sub(r"^```(?:sql)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    raw = raw.strip().rstrip(";")
    return _fix_access_sql(raw)


def _fix_access_sql(sql: str) -> str:
    """将常见 MySQL 风格 SQL 转换为 Access Jet SQL 兼容语法。

    修复项：
      1. 裸 JOIN → INNER JOIN（Access 不支持裸 JOIN）
      2. LIMIT N → SELECT TOP N（如果误写了 MySQL 语法）
    """
    # 1. 裸 JOIN（前面不是 INNER/LEFT/RIGHT/OUTER/CROSS/FULL）
    sql = re.sub(r'\b(?<!INNER\s)(?<!LEFT\s)(?<!RIGHT\s)(?<!OUTER\s)(?<!CROSS\s)(?<!FULL\s)JOIN\b',
                 'INNER JOIN', sql, flags=re.IGNORECASE)

    # 2. 清理可能产生的双重 INNER
    sql = re.sub(r'\bINNER\s+INNER\s+JOIN\b', 'INNER JOIN', sql, flags=re.IGNORECASE)

    # 3. LIMIT N（MySQL 残余）→ 如果末尾有 LIMIT，转为在最前面加 SELECT TOP N
    #    这是启发式修复，不能完全自动化，仅在 SQL 不是以 SELECT TOP 开头且末尾有 LIMIT 时尝试
    limit_match = re.search(r'\bLIMIT\s+(\d+)\s*$', sql, re.IGNORECASE)
    if limit_match and not re.search(r'\bSELECT\s+TOP\b', sql, re.IGNORECASE):
        limit_num = limit_match.group(1)
        sql = re.sub(r'\bLIMIT\s+\d+\s*$', '', sql, flags=re.IGNORECASE).rstrip()
        sql = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_num}', sql, count=1, flags=re.IGNORECASE)

    return sql


def _sanitize_text(text: str) -> str:
    """移除 Markdown 格式符号，转为纯文本（用于 answer 字段）。"""
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


def _sanitize_evidence(text: str) -> str:
    """清理 evidence 文本中的 Markdown 表格格式，但保留数据完整性。

    与 _sanitize_text 的区别：不运行 _(.+?)_ 正则，避免损坏
    下划线分隔的列名和数据（如 ntc_id, beam_name）。
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 只处理行首的标题/列表格式，不触碰下划线
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-| :]+$", "", text, flags=re.MULTILINE)
    # Markdown 表格的 | 分隔符转为空格，但保留内部内容
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repair_json(json_str: str) -> str:
    """尝试修复常见的 LLM JSON 输出问题。"""
    # 1. 移除尾部多余逗号（最常见的问题）
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    # 2. 忽略 JSON 前导/尾部非 JSON 文本
    m = re.search(r'\{[\s\S]*\}', json_str)
    if m:
        json_str = m.group(0)
    return json_str


def _parse_json_response(raw: str, question: str) -> str:
    """将 LLM 输出解析为规范的 JSON 回答字符串。解析失败则包裹为 fallback JSON。"""
    json_str = _extract_json(raw)

    # 尝试直接解析，失败则尝试修复后重试
    parsed = None
    for attempt in range(2):
        try:
            parsed = json.loads(json_str)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                json_str = _repair_json(json_str)
            else:
                logger.warning("[Parse] JSON 解析失败，使用 fallback 格式")

    if parsed is not None:
        parsed.setdefault("answer", "")
        parsed.setdefault("rewritten_query", "")
        parsed.setdefault("data_sources", [])
        parsed["answer"] = _sanitize_text(parsed["answer"])
        parsed["rewritten_query"] = _sanitize_text(parsed["rewritten_query"])
        for src in parsed.get("data_sources", []):
            src.setdefault("source_type", "llm_knowledge")
            src.setdefault("evidence", "")
            # 用 evidence 专用函数，避免 _(.+?)_ 损坏数据
            src["evidence"] = _sanitize_evidence(src["evidence"])
            src.pop("method", None)
            src.pop("description", None)
            src.pop("analysis", None)
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    # Fallback
    return json.dumps({
        "answer": raw[:2000],
        "rewritten_query": "",
        "data_sources": [{
            "source_type": "llm_knowledge",
            "evidence": raw[:500]
        }]
    }, ensure_ascii=False, indent=2)


def _extract_table_names(text: str, known_tables: set[str]) -> tuple[list[str], list[str]]:
    """从用户查询中提取明确提到的表名。

    策略：扫描查询中的所有 ASCII token，与已知表名精确匹配或前缀匹配。
    例如 "notice表中" → notice 匹配 known_tables 中的 notice。
    返回 (匹配到的已知表, 疑似表名但不在库中的字符串)。
    """
    mentioned = []
    unknown = []
    # 提取所有纯 ASCII 英文/数字/下划线 token（表名只可能是这些字符）
    # 注意：不用 \w，因为 Python re 的 \w 默认包含 Unicode 字母（如中文）
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    known_lower = {t.lower(): t for t in known_tables}

    for token in tokens:
        token_lower = token.lower()
        # 仅精确匹配——像 "adm" 这样的列名不应匹配 "adm_assoc" 表
        if token_lower in known_lower:
            tbl = known_lower[token_lower]
            if tbl not in mentioned:
                mentioned.append(tbl)
            continue
        # 不在已知表中 → 如果看起来像表名（纯小写英文+下划线, >=3 字符），标记为 unknown
        if len(token) >= 3 and re.match(r'^[a-z_]+$', token_lower):
            unknown.append(token)

    return mentioned, unknown


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
