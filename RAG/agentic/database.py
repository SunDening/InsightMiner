"""MySQL 连接与只读查询。"""

import re

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .config import MYSQL_CONFIG, FORBIDDEN_KEYWORDS

MYSQL_URL = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    f"?charset=utf8mb4"
)
engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)


def _remove_sql_strings(sql: str) -> str:
    """移除 SQL 字符串字面量，避免安全校验误判。"""
    cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)
    cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', '""', cleaned)
    return cleaned


def execute_mysql_query(sql: str) -> dict:
    """只读执行 SQL，返回结构化结果或错误。"""
    normalized = sql.strip().upper()
    cleaned = _remove_sql_strings(normalized)

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", cleaned):
            return {"ok": False, "error": f"安全限制：禁止 {kw} 操作，仅支持 SELECT。"}

    is_select = normalized.startswith("SELECT")
    is_cte = normalized.startswith("WITH") and "SELECT" in normalized
    if not (is_select or is_cte):
        return {"ok": False, "error": f"仅允许 SELECT 或 WITH...SELECT 查询。收到: {sql[:60]}..."}

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [list(row) for row in result.fetchall()]
            columns = list(result.keys())
            return {"ok": True, "columns": columns, "rows": rows, "row_count": len(rows)}
    except SQLAlchemyError as e:
        return {"ok": False, "error": str(e)}


def format_query_result(result: dict, max_rows: int = 50) -> str:
    """将查询结果格式化为 LLM 可读的 Markdown 表格。"""
    if not result["ok"]:
        return f"[查询错误] {result['error']}"
    columns = result["columns"]
    rows = result["rows"]
    if result["row_count"] == 0:
        return f"查询返回 0 行。列: {', '.join(columns)}"
    truncated = rows[:max_rows]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---" for _ in columns]) + "|",
    ]
    for row in truncated:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    output = "\n".join(lines)
    if result["row_count"] > max_rows:
        output += f"\n\n（结果已截断，仅显示前 {max_rows} 行，共 {result['row_count']} 行）"
    else:
        output += f"\n\n（共 {result['row_count']} 行）"
    return output
