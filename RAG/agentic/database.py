"""Access .mdb 数据库连接与只读查询。"""

import re
import threading

import pyodbc

from .config import MDB_PATH, FORBIDDEN_KEYWORDS, logger

_conn_lock = threading.Lock()


def _get_connection():
    """创建 Access .mdb 数据库连接。"""
    conn_str = (
        f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};"
        f"DBQ={MDB_PATH};"
    )
    return pyodbc.connect(conn_str)


def _remove_sql_strings(sql: str) -> str:
    """移除 SQL 字符串字面量，避免安全校验误判。"""
    cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)
    cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', '""', cleaned)
    return cleaned


def execute_access_query(sql: str) -> dict:
    """只读执行 Access SQL，返回结构化结果或错误。"""
    normalized = sql.strip().upper()
    cleaned = _remove_sql_strings(normalized)

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", cleaned):
            return {"ok": False, "error": f"安全限制：禁止 {kw} 操作，仅支持 SELECT。"}

    is_select = normalized.startswith("SELECT")
    if not is_select:
        return {"ok": False, "error": f"仅允许 SELECT 只读查询。收到: {sql[:60]}..."}

    try:
        with _conn_lock:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = [list(row) for row in cursor.fetchall()]
            columns = [d[0] for d in cursor.description] if cursor.description else []
            cursor.close()
            conn.close()
            return {"ok": True, "columns": columns, "rows": rows, "row_count": len(rows)}
    except pyodbc.Error as e:
        err_msg = str(e)
        # 针对常见 Access SQL 语法错误给出修复提示
        if "-3506" in err_msg:
            err_msg += (
                " Access SQL 语法提示：1) 必须使用 INNER JOIN / LEFT JOIN / RIGHT JOIN，"
                "禁止裸写 JOIN；2) 字符串字面量用单引号 'xxx'，不用双引号 \"xxx\"；"
                "3) 表别名可省略 AS（如 FROM notice T1）；4) 时间值用 #YYYY-MM-DD# 格式"
            )
        elif "-3010" in err_msg:
            err_msg += (
                " Access SQL 语法提示：字符串字面量必须用单引号 'xxx'，"
                "不能使用双引号 \"xxx\"（双引号会被视为字段名或参数）"
            )
        elif "-3007" in err_msg:
            err_msg += (
                " Access SQL 语法提示：列名在多表中存在歧义，请使用 表名.列名 或 别名.列名 明确指定"
            )
        return {"ok": False, "error": err_msg}
    except Exception as e:
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
        "| " + " | ".join(str(c) for c in columns) + " |",
        "|" + "|".join(["---" for _ in columns]) + "|",
    ]
    for row in truncated:
        lines.append("| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |")
    output = "\n".join(lines)
    if result["row_count"] > max_rows:
        output += f"\n\n（结果已截断，仅显示前 {max_rows} 行，共 {result['row_count']} 行）"
    else:
        output += f"\n\n（共 {result['row_count']} 行）"
    return output
