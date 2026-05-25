"""Schema Enricher — 用 LLM 将技术 Schema 描述转换为业务语义描述。

批量生成表级和列级的业务描述，保存到 .schema_enriched.json。
SchemaIndexer 加载富化描述后重建索引，大幅提升语义检索精度。
"""

import os
import json
import asyncio

from langchain.messages import HumanMessage

from .config import (
    logger, SCHEMA_JSON_PATH, TABLE_DESC_JSON_PATH,
    SCHEMA_ENRICHED_PATH, SCHEMA_ENRICHED_MANIFEST,
    GARBAGE_TABLES, LLM_PROVIDER,
)
from .llm import create_llm


TABLE_ENRICH_PROMPT = """你是一个 ITU-R 空间网络通知系统（SNS）数据库专家。请为以下数据库表生成一段"业务人员可理解的描述"，用于语义搜索。

## 表信息
- 表名: {table_name}
- 业务含义（来自文档）: {description}
- 所属模块: {module}
- 主键: {primary_key}
- 列名列表: {column_list}
- 关联关系（该表的列 → 其他表.列）: {relationships}

## 要求
请用中文生成一段 100-150 字的描述，包含以下要素：
1. 这张表存储了什么业务数据
2. 什么场景下需要查询这张表（给出 2-3 个使用场景）
3. 核心字段及其业务含义（选最重要的 3-5 个）
4. 与其他表的关键关联

输出格式（纯文本，不要 Markdown）：
[TBL] {table_name} — {{一句话业务定义}}
业务含义: ...
使用场景: ...
核心字段: ...
关键关联: ..."""


COLUMN_ENRICH_PROMPT = """你是一个 ITU-R 空间网络通知系统（SNS）数据库专家。请为以下数据库列生成一段"业务人员可理解的描述"。

## 列信息
- 所属表: {table_name}（{table_desc}）
- 列名: {column_name}
- 类型: {column_type}
- 英文描述: {desc_en}
- 中文描述: {desc_zh}
- 是否主键: {is_pk}

## 要求
用中文生成一段描述，包含：
1. 该列的业务含义
2. 典型值示例（如有）
3. 在查询中的使用方式（WHERE / GROUP BY / ORDER BY / JOIN 等）

输出格式（纯文本，不要 Markdown）：
[COL] {table_name}.{column_name} | {column_type} | {{一句话含义}}
业务含义: ...
使用方式: ..."""


class SchemaEnricher:
    """为 Schema 元数据生成业务语义描述。

    遍历所有表/列，批量调 LLM 生成描述，缓存到 .schema_enriched.json。
    通过 manifest 检测，仅在 schema.json 变更后重新生成。
    """

    def __init__(self, schema_json_path: str = SCHEMA_JSON_PATH,
                 table_desc_path: str = TABLE_DESC_JSON_PATH,
                 enriched_path: str = SCHEMA_ENRICHED_PATH,
                 manifest_path: str = SCHEMA_ENRICHED_MANIFEST):
        self.schema_json_path = schema_json_path
        self.table_desc_path = table_desc_path
        self.enriched_path = enriched_path
        self.manifest_path = manifest_path

    def _compute_manifest(self) -> dict:
        mtime = os.path.getmtime(self.schema_json_path)
        if os.path.exists(self.table_desc_path):
            mtime2 = os.path.getmtime(self.table_desc_path)
        else:
            mtime2 = 0
        return {"schema_mtime": mtime, "table_desc_mtime": mtime2}

    def _manifest_valid(self) -> bool:
        if not os.path.exists(self.manifest_path):
            return False
        if not os.path.exists(self.enriched_path):
            return False
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return saved == self._compute_manifest()
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def _load_raw_metadata(self) -> list[dict]:
        """从 schema.json + table_desc.json 加载原始元数据（未富化）。"""
        with open(self.schema_json_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

        table_descriptions: dict[str, str] = {}
        table_modules: dict[str, str] = {}
        if os.path.exists(self.table_desc_path):
            with open(self.table_desc_path, "r", encoding="utf-8") as f:
                desc_data = json.load(f)
            for module in desc_data.get("modules", []):
                mod_name = module.get("module_name", "")
                for t in module.get("tables", []):
                    table_descriptions[t["table_name"]] = t.get("description", "")
                    table_modules[t["table_name"]] = mod_name

        tables = []
        for t in schema_data.get("tables", []):
            name = t["name"]
            if name in GARBAGE_TABLES:
                continue
            tables.append({
                "name": name,
                "description": table_descriptions.get(name, ""),
                "module": table_modules.get(name, ""),
                "primary_key": t.get("primary_key", []),
                "columns": t.get("columns", []),
                "relationships": t.get("relationships", []),
            })
        return tables

    async def enrich_all(self) -> list[dict]:
        """批量生成所有表的业务语义描述。如有有效缓存则直接返回。"""
        if self._manifest_valid():
            logger.info("[SchemaEnricher] 缓存有效，直接加载")
            with open(self.enriched_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("[SchemaEnricher] 开始批量生成业务语义描述...")
        raw_tables = self._load_raw_metadata()
        llm = create_llm(LLM_PROVIDER, temperature=0.2)
        enriched = []

        for i, t in enumerate(raw_tables):
            name = t["name"]
            logger.info(f"[SchemaEnricher] 处理表 {i + 1}/{len(raw_tables)}: {name}")

            # 生成表级描述
            col_list = ", ".join(c["name"] for c in t["columns"])
            rel_list = "; ".join(
                f"{r['column']} → {r['target_table']}.{r.get('target_column', '')}"
                for r in t["relationships"][:10]
            ) if t["relationships"] else "无"

            table_prompt = TABLE_ENRICH_PROMPT.format(
                table_name=name,
                description=t["description"] or "无",
                module=t["module"] or "无",
                primary_key=", ".join(t["primary_key"]) if t["primary_key"] else "无",
                column_list=col_list,
                relationships=rel_list,
            )
            try:
                resp = await llm.ainvoke([HumanMessage(content=table_prompt)])
                table_enriched_desc = resp.content.strip()
            except Exception as e:
                logger.warning(f"[SchemaEnricher] 表 {name} 描述生成失败: {e}")
                table_enriched_desc = f"[TBL] {name} — {t['description']}"

            # 列级富化：使用 schema.json 中已有的中英文描述（不额外调 LLM）
            col_enriched = {}
            for c in t["columns"]:
                parts = []
                if c.get("description_zh"):
                    parts.append(c["description_zh"])
                if c.get("description_en"):
                    parts.append(c["description_en"])
                if c.get("primary_key"):
                    parts.insert(0, "主键")
                if parts:
                    col_enriched[c["name"]] = (
                        f"[COL] {name}.{c['name']} | {c.get('type', 'VARCHAR')} | "
                        + " | ".join(parts)
                    )

            enriched.append({
                "name": name,
                "description": t["description"],
                "module": t["module"],
                "primary_key": t["primary_key"],
                "tabble_enriched_desc": table_enriched_desc,
                "columns": t["columns"],
                "col_enriched_desc": col_enriched,
                "relationships": t["relationships"],
            })

        # 保存
        os.makedirs(os.path.dirname(self.enriched_path), exist_ok=True)
        with open(self.enriched_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._compute_manifest(), f, ensure_ascii=False, indent=2)

        logger.info(
            f"[SchemaEnricher] 完成，{len(enriched)} 张表的语义描述已保存"
        )
        return enriched
