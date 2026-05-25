"""LangGraph 节点函数、状态定义与 Prompt 模板。

KB 相关的节点使用工厂函数（闭包捕获 RagKnowledgeBase）：
  - make_classify_intent_node(kb)
  - make_retrieve_kb_node(kb)
  - make_tavily_search_node(kb)

其余节点为普通 async 函数。
"""

import json
import re
from typing import Annotated, TypedDict

from langchain.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from .config import (
    logger, LLM_PROVIDER,
    MAX_SQL_RETRY, MAX_REVIEW_RETRY, FORBIDDEN_KEYWORDS,
)
from .llm import create_llm
from .database import execute_access_query, format_query_result, _remove_sql_strings
from .utils import (
    _extract_json, _clean_sql_output, _parse_json_response, _format_history,
)
from .web_search import search_web
from .knowledge_base import RagKnowledgeBase
from .schema_indexer import SchemaIndexer
from .query_memory import QueryMemory


# ════════════════════════════════════════════════════════════════════════
# AgentState
# ════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    question:        str
    rewritten_query: str
    intent:          str          # chat | sql | kb | web | clarify
    current_time:    str
    # SQL 路径
    schema_context:       str
    full_schema_context:  str    # build_schema 的原始完整输出，重试时回退使用
    schema_tables:        list   # build_schema 检索到的全部表名
    explicit_tables:      list   # 用户明确指定的表名
    bridge_tables:        list   # 连接指定表所需的桥接表（FK 图自动发现）
    query_memory_fewshot: str    # 从 QueryMemory 检索到的 few-shot 示例
    sql:                  str
    query_result:         str
    retry_count:          int
    error_info:           str
    # KB / Web 路径
    retrieved_docs:  str
    kb_confidence:   float         # 精排最高分，用于判断是否 LLM 兜底
    web_results:     str
    # 统一输出
    final_answer:    str
    review_count:    int
    review_feedback: str


# ════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ════════════════════════════════════════════════════════════════════════

INTENT_CLASSIFY_PROMPT = """你是一个 ITU-R 空间网络通知系统（SNS）的智能助手。请同时完成意图分类和查询重写两项任务。

**重要：如果用户问题包含代词（"它们""这些""其""他"等），请参考对话历史确定指代对象。如果对话历史中刚讨论过某组具体实体（如某几颗卫星），当前问题是对这些实体的追问，则意图应为 sql / kb（根据追问内容判断），而不是 chat 或 clarify。**

意图分类优先级（从高到低）：
- **sql**: 询问数据库中某张表的具体数据，需要执行 SQL 才能回答（如统计数量、列出记录、关联分析）。包括对上一轮查询结果的追问
- **web**: 用户**明确要求**上网搜索（消息中包含"上网查""搜索网络""在网上找""上网搜""帮我查一下网络"等短语）
- **clarify**: 用户问题过于模糊或存在歧义，且对话历史中也无法确定指代对象时使用
- **kb**（默认）: 除以上三类外的所有问题，包括但不限于：概念解释、人物事件、专业知识、历史背景、理论分析。即使你不确定知识库是否包含相关内容，也要路由到 kb，不要擅自判断 kb 中没有而走 chat
- **chat**: **仅限**问候寒暄（"你好""早上好"）和助手自我介绍（"你是谁""你能做什么"）。其他任何内容型问题都必须走 kb

查询重写规则（仅 sql/kb/web 需要重写，chat/clarify 保持原文）：
- 模糊时间词（"最近""近期""今天"）→ 结合当前时间替换为具体日期范围
- 代词（"那个""它"）→ 根据对话历史确定具体实体
- 口语化表达 → 数据库查询/文档检索友好的措辞
- SQL 意图：只补全用户提到的条件的字段名，不要自行添加用户没提的表名或关联路径。例如用户说"查XX表中YY=ZZ的记录"，改写为"查询XX表中YY=ZZ的记录"即可，不要写成"通过AA表关联XX表"
- KB 意图提取核心关键词；Web 意图优化搜索词

## 对话历史
{history}

## 当前时间
{current_time}

数据库系统：ITU-R SNS（空间网络通知与频率管理数据库），Access .mdb，共 {table_count} 张表
数据表：{table_list}
知识库文档：{kb_description}

## 用户问题
{question}

## 输出格式
只输出一个 JSON 对象（不要 Markdown 代码块）：
{{"intent": "sql", "rewritten_query": "重写后的查询文本（chat/clarify 时为原文）"}}"""


CLARIFY_PROMPT = """你是一个航天测控领域的智能助手。用户的问题过于模糊，你无法确定该如何回答。

## 对话历史
{history}

## 用户问题
{question}

## 任务
向用户提出一个友好的反问，帮助澄清ta的意图。要求：
- 反问要具体，给出可能的选项或示例
- 语气友好，不要让用户感到被质问
- 用中文，一句话即可

只输出反问句本身，不要额外文本。"""


CHAT_PROMPT = """你是一个航天测控领域的智能助手，知识涵盖运载火箭、卫星测控、空间天气等方面。

## 当前时间
{current_time}

## 对话历史
{history}

## 用户问题
{question}

## 回答要求
请以 JSON 格式输出（只输出 JSON，不要 Markdown 代码块）：

{{
  "answer": "对用户问题的自然语言回答（中文）",
  "rewritten_query": "",
  "data_sources": [
    {{
      "source_type": "llm_knowledge",
      "evidence": "基于预训练知识和逻辑推理"
    }}
  ]
}}

注意：
- ⚠️ 禁止使用任何 Markdown 格式：禁止 **加粗**、*斜体*、`代码块`、# 标题、- 列表、| 表格等
- 只输出纯文本和中文/英文标点符号
- 如果用户问题涉及时间（如"今天""最近""当前"），请结合上方提供的当前时间作答。"""


SQL_GEN_PROMPT = """你是一个 Microsoft Access (Jet SQL) 专家。

## 对话历史（请依据此历史理解用户的指代和上下文）
{history}

## 数据库 Schema
{schema}

## 当前用户问题
{question}

{error_block}

## SQL 生成要求
- 如果用户的当前问题中包含"这些""它们""其"等代词，请根据对话历史确定指代的具体实体，并在 SQL 中加上对应的过滤条件
- 如果对话历史中的上一次查询已经限定了范围，当前问题若在该范围内追问，SQL 应继承相同的过滤条件
- 使用 Access Jet SQL 语法：
  - 限制行数用 SELECT TOP N，不用 LIMIT
  - 字符串连接用 & 运算符，不用 CONCAT()
  - 时间值用 #YYYY-MM-DD HH:MM:SS# 格式
  - 字符串字面量用单引号 'xxx'，不能用双引号 "xxx"
  - ⚠️ JOIN 必须写明类型：INNER JOIN / LEFT JOIN / RIGHT JOIN，绝对禁止裸写 JOIN
  - 不支持 WITH ... AS (CTE)，复杂查询用子查询或嵌套 SELECT
  - 不支持 CASE WHEN，用 IIF() 或 SWITCH() 代替
- **表数量最小化原则**：只使用回答问题绝对必要的表。每多 JOIN 一张不必要的表，用户的查询结果就会包含错误数据。如果查询只用一张表就能完成，就不要 JOIN 任何其他表。绝对不要为了"查询更全面"而增加额外的表 JOIN。
- **单表优先原则**：如果用户问题只涉及一张表就只查那一张表，禁止为了"更准确"而添加不必要的 JOIN。只有当用户明确要求跨表关联时才 JOIN
- 聚合查询正确使用 GROUP BY
- 默认 SELECT TOP 50

## 反面示例
用户问："查询 notice 表中 2024 年的通知数量"。
错误做法：SELECT COUNT(*) FROM notice INNER JOIN adm_assoc ON notice.ntc_id = adm_assoc.ntc_id WHERE YEAR(notice.date) = 2024
正确做法：SELECT COUNT(*) FROM notice WHERE YEAR(date) = 2024

用户问："列出 grp 表中的所有频率指配记录"。
错误做法：SELECT grp.* FROM grp INNER JOIN s_beam ON grp.grp_id = s_beam.grp_id
正确做法：SELECT TOP 50 * FROM grp

## 输出前自检
在最终 SQL 之前，用一行 SQL 注释列出你使用的所有表，并简要说明每张表为什么必不可少。格式：
-- 表: notice (存储通知数据，问题直接询问此表), adm_assoc (用户要求关联主管部门信息)
SELECT ...

只输出 SQL 语句和自检注释，不要额外解释。"""


SYNTHESIZE_PROMPT = """你是一个航天测控领域的智能数据分析助手。

## 当前时间
{current_time}

## 对话历史
{history}

## 用户原始问题
{question}

## 重写后的查询
{rewritten_query}

{context_block}

## 输出要求
请以 JSON 格式输出（只输出 JSON，不要 Markdown 代码块标记）：

{{
  "answer": "对用户问题的自然语言回答，用中文，专业且易懂",
  "rewritten_query": "{rewritten_query}",
  "data_sources": [
    {{
      "source_type": "{source_type}",
      "evidence": "原始数据或原文（查询结果表格/文档原文字段/API返回内容/模型知识声明）"
    }}
  ]
}}

注意：
- answer 要直接回应用户的问题，不要泛泛而谈
- ⚠️ 禁止使用任何 Markdown 格式：禁止 **加粗**、*斜体*、`代码块`、# 标题、- 列表、| 表格、水平线等
- 只输出纯文本和中文/英文标点符号
- data_sources 数组可以包含多个来源（如同时查了 mysql 和 kb）
- evidence 字段填入原始数据（纯文本格式）
- 如果用户问题涉及时间（如"今天""最近""当前"），请结合上方提供的当前时间作答
- 如果查询结果为空，在 answer 中诚实告知并给出可能的原因"""


REVIEW_PROMPT = """你是一个严格的质量审核员。审视以下 JSON 格式的回答，判断质量是否合格。

## 用户问题
{question}

## 待审核回答
{answer}

## 审核标准
1. answer 是否直接、准确地回答了用户问题？
2. data_sources 中的 evidence 是否具体且支撑结论？
3. 有无明显幻觉、矛盾或遗漏？

## 输出格式
请只输出一个 JSON：
{{
  "score": 1-5,
  "issues": "如无问题写'无'，有问题则明确列出"
}}

评分含义：5=完美  4=良好  3=可接受  2=有缺陷  1=严重错误"""


# ════════════════════════════════════════════════════════════════════════
# KB 相关节点（工厂函数，闭包捕获 RagKnowledgeBase）
# ════════════════════════════════════════════════════════════════════════

def make_classify_intent_node(kb: RagKnowledgeBase, schema_indexer: SchemaIndexer):
    """意图分类 + 查询重写，合并为一次 LLM 调用。"""
    async def classify_intent_node(state: AgentState) -> dict:
        llm = create_llm(LLM_PROVIDER, temperature=0.1)
        question = state["question"]
        current_time = state.get("current_time", "未知")
        history = _format_history(state.get("messages", []))
        kb_description = kb.get_descriptions_text()
        table_list = schema_indexer.get_table_list_text()
        table_count = len(schema_indexer.tables_meta)

        prompt = INTENT_CLASSIFY_PROMPT.format(
            current_time=current_time, history=history, question=question,
            kb_description=kb_description, table_list=table_list,
            table_count=table_count,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        intent = "chat"
        rewritten_query = question
        try:
            parsed = json.loads(_extract_json(raw))
            intent = parsed.get("intent", "chat")
            rewritten_query = parsed.get("rewritten_query", question)
        except json.JSONDecodeError:
            raw_lower = raw.lower()
            for candidate in ["clarify", "chat", "sql", "kb", "web"]:
                if candidate in raw_lower:
                    intent = candidate
                    break

        logger.info(f"[Intent] {question[:50]}... → intent={intent}, rewrite={rewritten_query[:60]}")
        return {"intent": intent, "rewritten_query": rewritten_query}
    return classify_intent_node


def make_retrieve_kb_node(kb: RagKnowledgeBase):
    """从 ChromaDB 知识库检索相关文档，返回内容和置信度。"""
    async def retrieve_kb_node(state: AgentState) -> dict:
        question = state.get("rewritten_query") or state["question"]
        docs, confidence = await kb.retrieve(question)
        logger.info(f"[KB] 检索完成，置信度 {confidence:.2f}，结果长度 {len(docs)} 字符")
        return {"retrieved_docs": docs, "kb_confidence": confidence}
    return retrieve_kb_node


def make_tavily_search_node(kb: RagKnowledgeBase):
    """使用 Tavily 搜索网络，拆分结果后 CrossEncoder 精排取 Top-3。"""
    async def tavily_search_node(state: AgentState) -> dict:
        question = state.get("rewritten_query") or state["question"]
        raw_results = search_web(question)

        try:
            parsed = json.loads(raw_results)
            passages = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        content = item.get("content") or item.get("snippet") or ""
                        if content:
                            passages.append(content)
            if not passages and isinstance(parsed, dict):
                content = parsed.get("content") or parsed.get("snippet") or ""
                if content:
                    passages.append(content)
        except (json.JSONDecodeError, TypeError):
            passages = []

        if passages and kb.reranker_model is not None:
            passages = kb.rerank(question, passages, top_k=3)
            results = "\n\n---\n\n".join(passages)
        else:
            results = raw_results

        logger.info(f"[Web] 搜索完成，精排后 {len(results)} 字符")
        return {"web_results": results}
    return tavily_search_node


# ════════════════════════════════════════════════════════════════════════
# 直接节点函数（无 KB 依赖）
# ════════════════════════════════════════════════════════════════════════

async def ask_clarification_node(state: AgentState) -> dict:
    """生成反问句，引导用户澄清模糊意图。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.3)
    question = state["question"]
    history = _format_history(state.get("messages", []))

    prompt = CLARIFY_PROMPT.format(history=history, question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    clarification = response.content.strip()

    logger.info(f"[Clarify] 反问: {clarification[:80]}")
    final_answer = json.dumps({
        "answer": clarification,
        "rewritten_query": "",
        "data_sources": [{
            "source_type": "llm_knowledge",
            "evidence": f"原始问题: {question}"
        }]
    }, ensure_ascii=False, indent=2)
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }


async def chat_respond_node(state: AgentState) -> dict:
    """闲聊路径 — LLM 直接回答。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.3)
    question = state["question"]
    history = _format_history(state.get("messages", []))
    current_time = state.get("current_time", "未知")

    prompt = CHAT_PROMPT.format(current_time=current_time, history=history, question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    final_answer = _parse_json_response(response.content, question)

    logger.info(f"[Chat] 生成回答，长度 {len(final_answer)} 字符")
    return {
        "final_answer": final_answer,
        "review_count": 0,
        "review_feedback": "",
        "messages": [AIMessage(content=final_answer)],
    }


def make_build_schema_node(indexer: SchemaIndexer, entity_router=None):
    """动态检索相关表的 schema 上下文，并强制包含用户明确指定的表。

    集成 EntityRouter：先做实体路由缩小范围，再做精细检索。
    """
    async def build_schema_node(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        raw_question = state.get("question", "")

        # 提前提取用户明确提到的表名（仅从原始问题提取，rewrite 可能引入多余表）
        from .utils import _extract_table_names
        mentioned, _ = _extract_table_names(
            raw_question, set(indexer.tables_meta.keys())
        )

        # 发现桥接表
        bridges = indexer.find_bridge_tables(mentioned) if len(mentioned) >= 2 else []
        force_tables = mentioned + bridges

        # 实体路由：仅在用户未指定表名时使用，缩小搜索空间
        entity_scope = None
        if entity_router is not None and not mentioned:
            entity_scope = entity_router.route(query, max_entities=2)

        schema_context = await indexer.build_schema_context(
            query, force_tables=force_tables, entity_scope=entity_scope,
        )
        # 从 schema_context 文本中提取表名（匹配 ### [标签] table_name 格式）
        _parsed_tables = re.findall(r'^###\s+\[.+?\]\s+(\w+)', schema_context, re.MULTILINE)
        # 确保用户指定表和桥接表在列表中
        _all_schema_tables = list(dict.fromkeys(force_tables + _parsed_tables))

        logger.info(f"[Schema] 动态组装完成，{len(schema_context)} 字符, "
                    f"{len(_all_schema_tables)} 张表"
                    f"{' (指定: ' + ', '.join(mentioned) + ')' if mentioned else ''}"
                    f"{' (桥接: ' + ', '.join(bridges) + ')' if bridges else ''}"
                    f"{' (实体: ' + str(len(entity_scope)) + '表)' if entity_scope else ''}")
        return {
            "schema_context": schema_context,
            "full_schema_context": schema_context,
            "schema_tables": _all_schema_tables,
            "explicit_tables": mentioned,
            "bridge_tables": bridges,
            "retry_count": 0,
            "error_info": "",
        }
    return build_schema_node


def _filter_schema_context(schema_text: str, keep_tables: list[str]) -> str:
    """从完整 schema_context 中仅保留指定表的段落。

    结构划分：
      1. 全局头部 — 第一个 '### [' 之前的所有内容（如 "## 相关表 (N/M)"）
      2. 表段落 — 以 '### [' 开头，到下一个 '### [' 或 '## ' 为止
      3. 全局尾部 — 表段落结束后，以 '## ' 开头的全局信息
         （如 "## 表间关联"、"## SQL 注意事项"、"## ⚠️ JOIN 约束"）
    保留：全局头部 + 匹配表的段落 + 全局尾部。
    """
    keep_set = set(t.strip().lower() for t in keep_tables)
    lines = schema_text.split("\n")

    # 第一遍：定位各区域的边界
    first_table_idx: int | None = None
    footer_start_idx: int | None = None

    for i, line in enumerate(lines):
        if re.match(r'^###\s+\[.+?\]\s+\w+', line):
            if first_table_idx is None:
                first_table_idx = i
        elif first_table_idx is not None and re.match(r'^##\s', line):
            footer_start_idx = i
            break  # 第一个 ## 头部之后即全局尾部

    if first_table_idx is None:
        return schema_text  # 无表段落，返回原文

    result: list[str] = []

    # 1) 全局头部
    result.extend(lines[:first_table_idx])

    # 2) 遍历表段落，仅保留匹配者
    table_zone_end = footer_start_idx if footer_start_idx is not None else len(lines)
    i = first_table_idx
    while i < table_zone_end:
        m = re.match(r'^###\s+\[.+?\]\s+(\w+)', lines[i])
        if m:
            tbl_name = m.group(1).strip().lower()
            # 找到此表段落的结束：下一个 ### [ 或 ##  或 table_zone_end
            j = i + 1
            while j < table_zone_end:
                if re.match(r'^###\s+\[.+?\]\s+\w+', lines[j]):
                    break
                j += 1
            if tbl_name in keep_set:
                result.extend(lines[i:j])
            i = j
        else:
            # 表段落内的非表头行（不应出现，防御性跳过）
            i += 1

    # 3) 全局尾部
    if footer_start_idx is not None:
        result.extend(lines[footer_start_idx:])

    return "\n".join(result)


def make_select_tables_node(indexer: SchemaIndexer):
    """表必要性判断节点 — 在 build_schema 和 generate_sql 之间，

    用一次轻量 LLM 调用选出真正需要的表（通常 1-3 张），
    裁剪 schema_context 后再送入 SQL 生成，根除"看到就想 JOIN"的冲动。
    """
    async def select_tables_node(state: AgentState) -> dict:
        llm = create_llm(LLM_PROVIDER, temperature=0.0)
        question = state.get("rewritten_query") or state["question"]
        schema_tables = state.get("schema_tables", [])
        explicit_tables = state.get("explicit_tables", [])
        bridge_tables = state.get("bridge_tables", [])
        full_schema = state.get("full_schema_context", "")

        # 如果用户已指定表且只有 1-2 张，跳过 LLM 调用
        mandatory = list(dict.fromkeys(explicit_tables + bridge_tables))
        if len(schema_tables) <= 2:
            return {"schema_context": full_schema}

        # 构建轻量表描述（仅表名 + 一句话描述，不给列详情和 JOIN 路径）
        table_descs = []
        for tbl_name in schema_tables:
            meta = indexer.tables_meta.get(tbl_name, {})
            desc = meta.get("description", "")
            enriched = meta.get("enriched_desc", "")
            # 优先取富化描述的首句，否则取原始描述
            if enriched:
                lines = enriched.strip().split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("[TBL]"):
                        desc = stripped[:120]
                        break
            elif desc:
                desc = desc[:120]
            table_descs.append(f"- {tbl_name}: {desc}" if desc else f"- {tbl_name}")

        mandatory_hint = ""
        if mandatory:
            mandatory_hint = (
                f"\n以下表为用户明确指定或必需的桥接表，必须包含在结果中："
                f"{', '.join(mandatory)}"
            )

        prompt = (
            f"你是一个数据库专家。用户问题：\"{question}\"\n\n"
            f"可用的数据库表及简要说明：\n"
            f"{chr(10).join(table_descs)}\n"
            f"{mandatory_hint}\n\n"
            f"请严格分析解决问题所必需的最小表集合（通常1-3张），"
            f"不要包含任何可被替代或冗余的表。"
            f"只输出JSON：{{\"necessary_tables\": [\"table_a\", \"table_b\"]}}"
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        necessary_tables: list[str] = []
        try:
            parsed = json.loads(_extract_json(raw))
            llm_selected = parsed.get("necessary_tables", [])
            if isinstance(llm_selected, list) and llm_selected:
                necessary_tables = [t for t in llm_selected
                                    if isinstance(t, str) and t.strip() in indexer.tables_meta]
        except json.JSONDecodeError:
            pass

        # 保底：LLM 输出为空或解析失败 → 使用全部表
        if not necessary_tables:
            logger.info("[TableSelect] 未能选出表，使用完整 schema")
            return {"schema_context": full_schema}

        # 合并强制表
        for t in mandatory:
            if t not in necessary_tables:
                necessary_tables.insert(0, t)

        # 裁剪 schema
        filtered = _filter_schema_context(full_schema, necessary_tables)
        logger.info(
            f"[TableSelect] {len(schema_tables)} 表 → "
            f"{len(necessary_tables)} 表: {', '.join(necessary_tables[:8])}"
        )
        return {"schema_context": filtered if filtered else full_schema}

    return select_tables_node


def make_generate_sql_node(memory: QueryMemory | None = None):
    """根据用户问题 + schema + 历史 + QueryMemory → 生成 SQL。"""
    async def generate_sql_node(state: AgentState) -> dict:
        llm = create_llm(LLM_PROVIDER, temperature=0.1)

        question = state.get("rewritten_query") or state["question"]
        history = _format_history(state.get("messages", []))
        retry_count = state.get("retry_count", 0)
        error_info = state.get("error_info", "")

        # 首次用过滤后的 schema，重试时回退到全量 schema
        if retry_count > 0:
            schema = state.get("full_schema_context", "")
            if schema:
                logger.info("[SQL-Gen] 重试模式：使用全量 schema 回退")
        else:
            schema = state.get("schema_context", "")
        if not schema:
            schema = state.get("full_schema_context", "")

        # 检索相似历史查询作为 few-shot 示例
        fewshot = ""
        raw_question = state.get("question", "")
        if memory is not None and retry_count == 0:
            fewshot = memory.search(raw_question)
            if fewshot:
                logger.info("[SQL-Gen] 注入 QueryMemory few-shot 示例")

        # 从对话历史中提取上一轮查询上下文
        previous_context_hint = ""
        messages = state.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    try:
                        prev = json.loads(msg.content)
                        prev_answer = prev.get("answer", "")
                        prev_rewritten = prev.get("rewritten_query", "")
                        prev_evidence = ""
                        for src in prev.get("data_sources", []):
                            if src.get("source_type") == "mysql":
                                prev_evidence = src.get("evidence", "")
                                break
                        if prev_answer or prev_rewritten:
                            parts = []
                            if prev_rewritten:
                                parts.append(f"上一轮查询: {prev_rewritten}")
                            if prev_answer:
                                parts.append(f"上一轮回答: {prev_answer[:500]}")
                            if prev_evidence:
                                parts.append(f"上一轮查询结果: {prev_evidence[:500]}")
                            previous_context_hint = (
                                f"\n\n## 上一轮对话的上下文（用于理解当前问题的代词和指代）\n"
                                + "\n".join(parts) + "\n\n"
                                f"如果当前问题使用了代词（如'它们''这些''其'），"
                                f"请根据上一轮的查询结果确定具体指代实体，"
                                f"并在当前 SQL 中加上对应的过滤条件。"
                            )
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass

        if error_info:
            error_block = (
                f"## ⚠️ 上一次 SQL 失败\n"
                f"上次 SQL: {state.get('sql', 'N/A')}\n"
                f"失败原因: {error_info}\n"
                f"请修正错误。只输出修正后的 SQL，不要解释。"
            )
        else:
            error_block = ""

        # 从 state 中获取用户指定的表和系统发现的桥接表
        explicit_tables_hint = ""
        mentioned_tables = state.get("explicit_tables", [])
        bridge_tables = state.get("bridge_tables", [])

        if mentioned_tables:
            if len(mentioned_tables) == 1 and not bridge_tables:
                explicit_tables_hint = (
                    f"\n\n## ⚠️ 用户指定了唯一表：{mentioned_tables[0]}"
                    f"\n这是一个单表查询！只允许 SELECT FROM {mentioned_tables[0]}。"
                    f"\n绝对禁止 JOIN 任何其他表！"
                )
            else:
                all_specified = mentioned_tables + bridge_tables
                table_list_str = ' 和 '.join(mentioned_tables)
                bridge_list_str = ('（需要桥接表 ' + ', '.join(bridge_tables) + ' 来连接）') if bridge_tables else ''
                n_allowed = len(all_specified)
                explicit_tables_hint = (
                    f"\n\n## ⚠️ 用户指定了 {len(mentioned_tables)} 张表：{table_list_str}"
                    f"{bridge_list_str}"
                    f"\n只允许使用这 {n_allowed} 张表进行查询。"
                    f"\n绝对禁止引入其他表！"
                    f"\n\n关于 WHERE 条件：查询中提到的列名，请优先使用用户指定表中的该列"
                    f"（如 {mentioned_tables[0]}.列名），不要自动换成其他表的同名列。"
                )

        prompt = SQL_GEN_PROMPT.format(
            history=history, schema=schema, question=question, error_block=error_block
        ) + previous_context_hint + explicit_tables_hint + fewshot

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        sql = _clean_sql_output(response.content.strip())

        logger.info(f"[SQL-Gen] 生成 SQL (retry={retry_count}): {sql[:200]}")
        return {"sql": sql, "retry_count": retry_count}
    return generate_sql_node


async def validate_sql_node(state: AgentState) -> dict:
    """校验 SQL — 检查是否只读 SELECT 及基本语法。"""
    sql = state.get("sql", "")

    if not sql or not sql.strip():
        return {"error_info": "SQL 为空，请生成一条有效的 SELECT 查询语句。"}

    normalized = sql.strip().upper()
    cleaned = _remove_sql_strings(normalized)
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", cleaned):
            return {
                "error_info": (
                    f"你生成了包含 {kw} 的语句，但本系统仅支持 SELECT 只读查询。"
                    f"请重新生成一条纯 SELECT 语句来完成用户的需求。"
                )
            }

    if normalized.startswith("SELECT"):
        pass
    elif normalized.startswith("WITH") and "SELECT" in normalized:
        pass
    else:
        return {
            "error_info": (
                f"你生成的 SQL 不是合法的只读语句 (必须以 SELECT 或 WITH 开头): "
                f"{sql[:80]}。请生成一条纯 SELECT 查询语句（可使用 WITH CTE 语法）。"
            )
        }

    if "FROM" not in normalized:
        return {"error_info": "SQL 缺少 FROM 子句，请补充。"}

    # 统计 SQL 中涉及的表数量（仅用于日志记录，不做硬拦截）
    _table_matches = re.findall(
        r'\b(?:FROM|JOIN)\s+(\w+)', cleaned, re.IGNORECASE
    )
    _table_count = len(set(_table_matches))

    logger.info(f"[Validate] SQL 校验通过 (涉及 {_table_count} 张表: {', '.join(sorted(set(_table_matches)))})")
    return {"error_info": ""}


def make_execute_sql_node(memory: QueryMemory | None = None):
    """执行 SQL 并返回格式化结果。成功后记录到 QueryMemory。"""
    async def execute_sql_node(state: AgentState) -> dict:
        sql = state["sql"]
        result = execute_access_query(sql)
        formatted = format_query_result(result)

        if result["ok"]:
            logger.info(f"[Execute] 查询成功，{result['row_count']} 行")
            # 记录成功查询到 QueryMemory
            if memory is not None and result["row_count"] > 0:
                try:
                    memory.add(
                        question=state.get("question", ""),
                        rewritten=state.get("rewritten_query", ""),
                        sql=sql,
                        tables=state.get("explicit_tables", []),
                    )
                except Exception as e:
                    logger.warning(f"[Execute] 查询记忆记录失败: {e}")
            return {"query_result": formatted, "error_info": ""}
        else:
            logger.warning(f"[Execute] 查询失败: {result['error']}")
            return {
                "query_result": "",
                "error_info": result["error"],
                "retry_count": state.get("retry_count", 0) + 1,
            }
    return execute_sql_node


async def synthesize_answer_node(state: AgentState) -> dict:
    """统一合成最终 JSON 格式回答。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.2)

    question = state["question"]
    intent = state.get("intent", "chat")
    history = _format_history(state.get("messages", []))
    review_feedback = state.get("review_feedback", "")
    current_time = state.get("current_time", "未知")

    if intent == "sql":
        sql = state.get("sql", "")
        query_result = state.get("query_result", "")
        error_info = state.get("error_info", "")

        if error_info and not query_result:
            context_block = (
                f"## 执行的 SQL\n```sql\n{sql}\n```\n\n"
                f"## 错误\n{error_info}\n"
                f"请向用户解释查询失败的原因并建议解决方案。"
            )
            source_type = "none"
        elif query_result.startswith("[查询错误]"):
            context_block = f"## 查询结果\n{query_result}\n请向用户解释问题所在。"
            source_type = "none"
        else:
            # 截断过长结果，防止 LLM 生成的 JSON 中 evidence 字段过大导致解析失败
            max_result_len = 2000
            truncated_result = query_result[:max_result_len]
            if len(query_result) > max_result_len:
                truncated_result += "\n\n（结果已截断，仅展示前 2000 字符）"
            context_block = (
                f"## 执行的 SQL\n```sql\n{sql}\n```\n\n"
                f"## 查询结果（请将此表格原文填入 evidence 字段）\n{truncated_result}"
            )
            source_type = "access"

    elif intent == "kb":
        docs = state.get("retrieved_docs", "")
        confidence = state.get("kb_confidence", 0.0)

        if confidence < -2.0:
            # 检索置信度极低，知识库中很可能没有相关内容
            confidence_note = (
                f"⚠️ 检索置信度极低（{confidence:.2f}），知识库中未找到与用户问题直接相关的内容。\n"
                f"请在 answer 中明确告知用户，参考句式：\n"
                f"'知识库中未找到与该问题直接相关的内容。以下回答基于 LLM 自身知识生成，可能不完全准确：...'\n"
                f"data_sources 的 source_type 请设为 'llm_knowledge'，"
                f"evidence 中注明 '知识库检索未命中，回答由 LLM 生成，未经知识库验证'。\n\n"
            )
            source_type = "llm_knowledge"
        else:
            confidence_note = ""
            source_type = "knowledge_base"

        context_block = (
            f"{confidence_note}"
            f"## 知识库检索结果（请将原文片段填入 evidence 字段）\n{docs}"
        )

    elif intent == "web":
        web_results = state.get("web_results", "")
        context_block = f"## Web 搜索结果（请将原文填入 evidence 字段）\n{web_results}"
        source_type = "web_search"

    else:
        context_block = ""
        source_type = "llm_knowledge"

    if review_feedback:
        context_block += f"\n\n## ⚠️ 上一版回答的问题\n{review_feedback}\n请改进后重新输出。"

    rewritten_query = state.get("rewritten_query") or question

    prompt = SYNTHESIZE_PROMPT.format(
        current_time=current_time,
        history=history,
        question=question,
        rewritten_query=rewritten_query,
        context_block=context_block,
        source_type=source_type,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    final_answer = _parse_json_response(response.content, question)

    logger.info(f"[Synthesize] 生成回答，长度 {len(final_answer)} 字符 (intent={intent})")
    return {
        "final_answer": final_answer,
        "review_feedback": "",
        "messages": [AIMessage(content=final_answer)],
    }


async def self_review_node(state: AgentState) -> dict:
    """审视最终回答质量，低于阈值则触发重写。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.0)

    question = state["question"]
    answer = state.get("final_answer", "")
    review_count = state.get("review_count", 0)

    if not answer:
        return {"review_feedback": "回答为空，请重新生成。"}

    prompt = REVIEW_PROMPT.format(question=question, answer=answer[:3000])
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    try:
        review = json.loads(_extract_json(response.content))
        score = int(review.get("score", 3))
        issues = review.get("issues", "无")
    except (json.JSONDecodeError, ValueError):
        score = 3
        issues = "无法解析审核结果"

    logger.info(f"[Review] 评分: {score}/5 (第 {review_count + 1} 次)")

    if score >= 3:
        return {"review_count": review_count + 1}
    else:
        return {
            "review_count": review_count + 1,
            "review_feedback": f"审核未通过（分数 {score}/5）。问题: {issues}。请修正后重新输出。",
        }
