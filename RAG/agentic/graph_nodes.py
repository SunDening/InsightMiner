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
    logger, SCHEMA_DEFINITION, LLM_PROVIDER,
    MAX_SQL_RETRY, MAX_REVIEW_RETRY, FORBIDDEN_KEYWORDS,
)
from .llm import create_llm
from .database import execute_mysql_query, format_query_result, _remove_sql_strings
from .utils import (
    _extract_json, _clean_sql_output, _parse_json_response, _format_history,
)
from .web_search import search_web
from .knowledge_base import RagKnowledgeBase


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
    schema_context:  str
    sql:             str
    query_result:    str
    retry_count:     int
    error_info:      str
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

INTENT_CLASSIFY_PROMPT = """你是一个航天测控领域的智能助手。请同时完成意图分类和查询重写两项任务。

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
- SQL 意图补充表名和字段；KB 意图提取核心关键词；Web 意图优化搜索词

## 对话历史
{history}

## 当前时间
{current_time}

数据库表：launch_vehicles、launch_missions、satellites、ground_stations、frequency_allocations、tracking_sessions、electromagnetic_events、space_weather_bulletins
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


SQL_GEN_PROMPT = """你是一个 MySQL 8.0 专家。

## 对话历史（请依据此历史理解用户的指代和上下文）
{history}

## 数据库 Schema
{schema}

## 当前用户问题
{question}

{error_block}

## SQL 生成要求
- 如果用户的当前问题中包含"这些""它们""其"等代词，请根据对话历史确定指代的具体实体，并在 SQL 中加上对应的过滤条件
- 如果对话历史中的上一次查询已经限定了范围（如某个厂商 provider_name），当前问题若在该范围内追问，SQL 应继承相同的过滤条件
- 复杂的嵌套查询可使用 WITH ... AS (...) SELECT ... 语法（MySQL 8.0 CTE），这是完全合法的只读查询
- 只输出 SQL 语句本身，不要 Markdown 代码块标记、不要解释
- 字符串精确匹配用 =，模糊搜索用 LIKE '%keyword%'
- 多表查询正确使用 JOIN 和外键关系
- 聚合查询正确使用 GROUP BY
- 默认 LIMIT 50"""


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

def make_classify_intent_node(kb: RagKnowledgeBase):
    """意图分类 + 查询重写，合并为一次 LLM 调用。"""
    async def classify_intent_node(state: AgentState) -> dict:
        llm = create_llm(LLM_PROVIDER, temperature=0.1)
        question = state["question"]
        current_time = state.get("current_time", "未知")
        history = _format_history(state.get("messages", []))
        kb_description = kb.get_descriptions_text()

        prompt = INTENT_CLASSIFY_PROMPT.format(
            current_time=current_time, history=history, question=question,
            kb_description=kb_description
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
        docs, confidence = kb.retrieve(question)
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


async def build_schema_node(state: AgentState) -> dict:
    """构建全量 schema 上下文。"""
    logger.info("[Schema] 构建全量 schema 上下文")
    return {"schema_context": SCHEMA_DEFINITION, "retry_count": 0, "error_info": ""}


async def generate_sql_node(state: AgentState) -> dict:
    """根据用户问题 + schema + 历史 → 生成 SQL。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.1)

    question = state.get("rewritten_query") or state["question"]
    schema = state.get("schema_context", SCHEMA_DEFINITION)
    history = _format_history(state.get("messages", []))
    retry_count = state.get("retry_count", 0)
    error_info = state.get("error_info", "")

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
                            f"请根据上一轮的查询结果确定具体指代实体（如具体的卫星编号），"
                            f"并在当前 SQL 中加上对应的过滤条件（如 WHERE satellite_code IN (...) 或 WHERE satellite_id IN (...)）。"
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

    prompt = SQL_GEN_PROMPT.format(
        history=history, schema=schema, question=question, error_block=error_block
    ) + previous_context_hint

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    sql = _clean_sql_output(response.content.strip())

    logger.info(f"[SQL-Gen] 生成 SQL (retry={retry_count}): {sql[:200]}")
    return {"sql": sql, "retry_count": retry_count}


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

    logger.info("[Validate] SQL 校验通过")
    return {"error_info": ""}


async def execute_sql_node(state: AgentState) -> dict:
    """执行 SQL 并返回格式化结果。"""
    sql = state["sql"]
    result = execute_mysql_query(sql)
    formatted = format_query_result(result)

    if result["ok"]:
        logger.info(f"[Execute] 查询成功，{result['row_count']} 行")
        return {"query_result": formatted, "error_info": ""}
    else:
        logger.warning(f"[Execute] 查询失败: {result['error']}")
        return {
            "query_result": "",
            "error_info": result["error"],
            "retry_count": state.get("retry_count", 0) + 1,
        }


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
            context_block = (
                f"## 执行的 SQL\n```sql\n{sql}\n```\n\n"
                f"## 查询结果（请将此表格原文填入 evidence 字段）\n{query_result}"
            )
            source_type = "mysql"

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
