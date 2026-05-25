"""LangGraph 图构建 — 仅 KB / Chat / Web / Clarify 路径。"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .config import logger, MAX_REVIEW_RETRY
from .knowledge_base import RagKnowledgeBase
from .graph_nodes import (
    AgentState,
    make_classify_intent_node,
    make_retrieve_kb_node,
    make_tavily_search_node,
    ask_clarification_node,
    chat_respond_node,
    synthesize_answer_node,
    self_review_node,
)


def route_by_intent(state: AgentState) -> Literal[
    "ask_clarification", "chat_respond", "retrieve_kb", "tavily_search"
]:
    intent = state.get("intent", "chat")
    logger.info(f"[Route] intent={intent}")
    if intent == "clarify":
        return "ask_clarification"
    elif intent == "kb":
        return "retrieve_kb"
    elif intent == "web":
        return "tavily_search"
    else:
        return "chat_respond"


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


def build_agent_graph(kb: RagKnowledgeBase) -> CompiledStateGraph:
    """构建 LangGraph 状态图（无 SQL 路径）。"""
    classify_intent = make_classify_intent_node(kb)
    retrieve_kb = make_retrieve_kb_node(kb)
    tavily_search = make_tavily_search_node(kb)

    checkpointer = InMemorySaver()
    builder = StateGraph(AgentState)  # type: ignore

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("chat_respond", chat_respond_node)
    builder.add_node("retrieve_kb", retrieve_kb)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("synthesize_answer", synthesize_answer_node)
    builder.add_node("self_review", self_review_node)
    builder.add_node("ask_clarification", ask_clarification_node)

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges("classify_intent", route_by_intent, {
        "ask_clarification": "ask_clarification",
        "chat_respond": "chat_respond",
        "retrieve_kb": "retrieve_kb",
        "tavily_search": "tavily_search",
    })

    builder.add_edge("retrieve_kb", "synthesize_answer")
    builder.add_edge("tavily_search", "synthesize_answer")
    builder.add_edge("synthesize_answer", "self_review")
    builder.add_edge("ask_clarification", "self_review")
    builder.add_edge("chat_respond", END)

    builder.add_conditional_edges("self_review", route_after_review, {
        "synthesize_answer": "synthesize_answer",
        "__end__": END,
    })

    return builder.compile(checkpointer=checkpointer)
