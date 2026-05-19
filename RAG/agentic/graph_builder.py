"""LangGraph 图构建 + 路由函数。"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .config import logger, MAX_SQL_RETRY, MAX_REVIEW_RETRY
from .knowledge_base import RagKnowledgeBase
from .schema_indexer import SchemaIndexer
from .graph_nodes import (
    AgentState,
    make_classify_intent_node,
    make_build_schema_node,
    make_retrieve_kb_node,
    make_tavily_search_node,
    ask_clarification_node,
    chat_respond_node,
    generate_sql_node,
    validate_sql_node,
    execute_sql_node,
    synthesize_answer_node,
    self_review_node,
)


# ════════════════════════════════════════════════════════════════════════
# 路由函数
# ════════════════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> Literal[
    "ask_clarification", "chat_respond", "build_schema", "retrieve_kb", "tavily_search"
]:
    intent = state.get("intent", "chat")
    logger.info(f"[Route] intent={intent}")
    if intent == "clarify":
        return "ask_clarification"
    elif intent == "sql":
        return "build_schema"
    elif intent == "kb":
        return "retrieve_kb"
    elif intent == "web":
        return "tavily_search"
    else:
        return "chat_respond"


def route_after_validate(state: AgentState) -> Literal["execute_sql", "generate_sql"]:
    error = state.get("error_info", "")
    if error:
        logger.info(f"[Route] validate 拦截: {error[:80]}")
        return "generate_sql"
    return "execute_sql"


def route_after_execute(state: AgentState) -> Literal["generate_sql", "synthesize_answer"]:
    error = state.get("error_info", "")
    retry = state.get("retry_count", 0)

    if not error:
        return "synthesize_answer"

    if retry < MAX_SQL_RETRY:
        logger.info(f"[Route] SQL 失败，重试 {retry}/{MAX_SQL_RETRY}")
        return "generate_sql"

    logger.warning(f"[Route] SQL 已达最大重试 {MAX_SQL_RETRY}，终止")
    return "synthesize_answer"


def route_after_review(state: AgentState) -> Literal["synthesize_answer", "__end__"]:
    feedback = state.get("review_feedback", "")
    review_count = state.get("review_count", 0)

    if not feedback:
        logger.info("[Route] 自检通过，结束")
        return "__end__"

    if review_count < MAX_REVIEW_RETRY:
        logger.info(f"[Route] 自检不通过，重试 {review_count}/{MAX_REVIEW_RETRY}")
        return "synthesize_answer"

    logger.warning(f"[Route] 自检已达最大重试 {MAX_REVIEW_RETRY}，强制结束")
    return "__end__"


# ════════════════════════════════════════════════════════════════════════
# 图构建
# ════════════════════════════════════════════════════════════════════════

def build_agent_graph(kb: RagKnowledgeBase,
                      schema_indexer: SchemaIndexer) -> CompiledStateGraph:
    """构建完整的 LangGraph 状态图。

    kb 和 schema_indexer 必须已完成 initialize()。
    """
    # 通过工厂函数创建 KB/Schema 相关节点（闭包捕获）
    classify_intent = make_classify_intent_node(kb, schema_indexer)
    build_schema = make_build_schema_node(schema_indexer)
    retrieve_kb = make_retrieve_kb_node(kb)
    tavily_search = make_tavily_search_node(kb)

    checkpointer = InMemorySaver()

    builder = StateGraph(AgentState)  # type: ignore

    # 添加所有节点
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("chat_respond", chat_respond_node)
    builder.add_node("build_schema", build_schema)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("validate_sql", validate_sql_node)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("retrieve_kb", retrieve_kb)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("synthesize_answer", synthesize_answer_node)
    builder.add_node("self_review", self_review_node)
    builder.add_node("ask_clarification", ask_clarification_node)

    # START → classify_intent
    builder.add_edge(START, "classify_intent")

    # classify_intent → 按意图分发
    builder.add_conditional_edges("classify_intent", route_by_intent, {
        "ask_clarification": "ask_clarification",
        "chat_respond": "chat_respond",
        "build_schema": "build_schema",
        "retrieve_kb": "retrieve_kb",
        "tavily_search": "tavily_search",
    })

    # SQL 管线
    builder.add_edge("build_schema", "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges("validate_sql", route_after_validate, {
        "execute_sql": "execute_sql",
        "generate_sql": "generate_sql",
    })
    builder.add_conditional_edges("execute_sql", route_after_execute, {
        "synthesize_answer": "synthesize_answer",
        "generate_sql": "generate_sql",
    })

    # KB / Web / Clarify → synthesize_answer → self_review 自检循环
    # chat 路径简单，直接结束（不走自检）
    builder.add_edge("retrieve_kb", "synthesize_answer")
    builder.add_edge("tavily_search", "synthesize_answer")
    builder.add_edge("synthesize_answer", "self_review")
    builder.add_edge("ask_clarification", "self_review")
    builder.add_edge("chat_respond", END)

    # 自检循环
    builder.add_conditional_edges("self_review", route_after_review, {
        "synthesize_answer": "synthesize_answer",
        "__end__": END,
    })

    return builder.compile(checkpointer=checkpointer)
