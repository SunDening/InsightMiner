"""LangGraph state machine — intent classification, retrieval, synthesis, self-review."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from insight_miner.config import MAX_REVIEW_RETRY, RERANK_LOW_THRESHOLD
from insight_miner.core.document_processor import KnowledgeBaseIndex
from insight_miner.core.llm_factory import create_llm
from insight_miner.utils.helpers import format_history, parse_json_response


# ── State ──

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    rewritten_query: str
    intent: str  # chat | kb | web | clarify
    current_time: str
    retrieved_docs: str
    kb_confidence: float
    web_results: str
    query_entities: list[str]
    final_answer: str
    review_count: int
    review_feedback: str
    streaming: bool  # True = SSE 流式模式, False = batch 模式


# ── Prompts ──

INTENT_CLASSIFY_PROMPT = """你是一个智能问答系统的意图分类引擎。请根据用户问题、对话历史和知识库描述，确定用户意图。

意图说明：
- chat: 日常问候或聊天天（"你好""再见""谢谢"等）
- kb: 需要基于知识库文档回答的问题（默认意图）
- web: 需要联网搜索的实时信息（新闻、天气、股价等，优先级最高）
- clarify: 问题模糊需要追问澄清

规则：
1. 如果用户问题包含代词（"它们""这些""其""他"等），请参考对话历史确定指代对象
2. 如果用户提到"最近""今天""最新"等时间词，将模糊时间替换为具体日期
3. web 优先级最高，其次是 clarify，默认 kb，chat 最低

请以 JSON 格式输出：
{{"intent": "kb", "rewritten_query": "重写后的查询", "entities": ["实体1", "实体2"]}}

entities 是从 rewritten_query 中提取的关键实体列表（人名、地名、术语等）。

当前时间：{current_time}
知识库简介：{kb_description}

对话历史：
{history}

用户问题：{question}"""

CLARIFY_PROMPT = """用户的问题是模糊的，需要澄清。请生成一句友好的反问，引导用户提供更多信息。

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}"""

CHAT_PROMPT = """你是一个友好的AI助手，请直接回答用户的问题。

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}

请以 JSON 格式输出：
{{"answer": "你的回答", "rewritten_query": "", "data_sources": []}}"""

SYNTHESIZE_PROMPT = """你是一个严谨的知识库问答助手。请基于提供的上下文信息回答用户问题。

要求：
- 如果上下文中包含足够信息，给出准确、详细的回答
- 基于知识库内容直接回答，不要在回答中添加"根据文档""证据来源""文档提到"等引用说明
- 如果上下文信息不足，诚实告知用户无法从知识库中找到答案
- 不要编造事实

上下文：
{context_block}

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}

请以 JSON 格式输出：
{{"answer": "你的回答", "rewritten_query": "", "data_sources": [{{"source_type": "knowledge_base", "evidences": [{{"evidence": "原文片段", "score": 0.95}}]}}]}}"""

STREAM_SYNTHESIZE_PROMPT = """你是一个严谨的知识库问答助手。请基于提供的上下文信息直接回答用户问题。

要求：
- 如果上下文中包含足够信息，给出准确、详细的回答
- 基于知识库内容直接回答，不要在回答中添加"根据文档""证据来源""文档提到"等引用说明
- 如果上下文信息不足，诚实告知用户无法从知识库中找到答案
- 不要编造事实

上下文：
{context_block}

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}"""

STREAM_CHAT_PROMPT = """你是一个友好的AI助手，请直接回答用户的问题。

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}"""

STREAM_CLARIFY_PROMPT = """用户的问题是模糊的，需要澄清。请生成一句友好的反问，引导用户提供更多信息。

当前时间：{current_time}
对话历史：
{history}
用户问题：{question}"""

REVIEW_PROMPT = """请审核以下回答的质量。评分标准（1-5分）：
5=完美：回答准确、完整、引用恰当
4=良好：回答正确，少量可改进之处
3=可接受：回答基本正确，但有一些缺陷
2=有缺陷：回答有明显错误或遗漏
1=严重错误：回答包含幻觉或与上下文矛盾

审核要点：
- 是否直接准确地回答了用户问题
- 是否有具体证据支撑回答中的关键论点
- 是否存在幻觉、矛盾或重要信息遗漏

用户问题：{question}
模型回答：{answer}
参考上下文：{context}

请以 JSON 格式输出：
{{"score": 5, "issues": ""}} 或 {{"score": 2, "issues": "扣分原因"}}

通过阈值：score >= 3"""


# ── Helpers ──

def _build_context_block(state: AgentState) -> str:
    """Build context block from retrieved docs or web results, with low-confidence warning and review feedback."""
    intent = state["intent"]
    context_block = state.get("web_results", "") if intent == "web" else state.get("retrieved_docs", "")
    confidence = state.get("kb_confidence", 0.0)
    if confidence < RERANK_LOW_THRESHOLD:
        context_block += "\n\n注意：以上内容的置信度极低，知识库中可能没有相关文档。"
    review_feedback = state.get("review_feedback", "")
    if review_feedback:
        context_block += f"\n\n上一轮审核反馈（请针对性改进）：{review_feedback}"
    return context_block


async def _stream_and_collect(llm, prompt: str) -> str:
    """Call llm.astream() and collect all tokens into a single string."""
    full = ""
    async for chunk in llm.astream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full += token
    return full


# ── Node factories ──

def make_classify_intent_node(kb_index: KnowledgeBaseIndex, llm=None):
    if llm is None:
        llm = create_llm(temperature=0.1)

    async def classify_intent(state: AgentState) -> dict:
        history = format_history(state.get("messages", []))
        kb_desc = kb_index.kb_id
        prompt = INTENT_CLASSIFY_PROMPT.format(
            current_time=state["current_time"],
            kb_description=kb_desc,
            history=history or "无",
            question=state["question"],
        )
        logger.info("classifying intent via LLM… question=%.50s", state["question"])
        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_response(raw, state["question"])

        intent = parsed.get("rewritten_query", "")
        if not intent:
            rl = raw.lower()
            if "clarify" in rl:
                intent = "clarify"
            elif "chat" in rl:
                intent = "chat"
            elif "web" in rl:
                intent = "web"
            else:
                intent = "kb"
        else:
            try:
                extracted = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
                intent = extracted.get("intent", "kb")
            except (ValueError, json.JSONDecodeError):
                intent = "kb"

        # 知识库为空时走纯聊天路径
        if intent == "kb" and not kb_index.chunk_texts:
            intent = "chat"

        rewritten = parsed.get("rewritten_query", state["question"])
        entities = []
        try:
            extracted = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
            entities = extracted.get("entities", [])
        except (ValueError, json.JSONDecodeError):
            pass

        logger.info("classify_intent question=%.50s intent=%s entities=%s", state["question"], intent, entities)
        return {
            "intent": intent,
            "rewritten_query": rewritten,
            "query_entities": entities,
        }

    return classify_intent


def make_retrieve_kb_node(kb_index: KnowledgeBaseIndex):
    async def retrieve_kb(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        entities = state.get("query_entities", [])

        dense_results = kb_index.dense_search(query, k=20)
        bm25_results = kb_index.bm25_search(query, k=20)
        graph_results = kb_index.graph_search(query, k=40, query_entities=entities)

        fused = KnowledgeBaseIndex.rrf_fusion(dense_results, bm25_results, graph_results, k=60)
        top_candidates = fused[:10]

        if not top_candidates or not kb_index.chunk_texts:
            logger.info("retrieve_kb no_results query=%.50s", query)
            return {"retrieved_docs": "", "kb_confidence": -10.0}

        doc_texts = [kb_index.chunk_texts[idx] for idx, _ in top_candidates]
        doc_scores = [score for _, score in top_candidates]

        reranked = kb_index.rerank(query, doc_texts, top_k=len(doc_texts))
        max_score = reranked[0][1] if reranked else -10.0

        # 记录完整候选池的分数范围，用于置信度归一化
        pool_scores = [s for _, s in reranked]
        pool_min = min(pool_scores) if pool_scores else -5.0
        pool_max = max(pool_scores) if pool_scores else 5.0
        pool_range = pool_max - pool_min

        # 动态截断：保留分数 ≥ top-1 × 0.7 的证据，至少 1 条，至多 5 条
        if reranked:
            threshold = max_score * 0.7
            selected = [reranked[0]]
            for t, s in reranked[1:]:
                if s >= threshold and len(selected) < 5:
                    selected.append((t, s))
            reranked = selected

        logger.info(
            "retrieve_kb query=%.50s dense=%d bm25=%d graph=%d top_score=%.3f pool_range=[%.2f, %.2f] evidences=%d",
            query, len(dense_results), len(bm25_results), len(graph_results),
            max_score, pool_min, pool_max, len(reranked),
        )

        lines: list[str] = []
        final_evidences: list[dict] = []
        for i, (text, score) in enumerate(reranked):
            lines.append(f"--- 文档片段 {i + 1} (score: {score:.4f}) ---\n{text}")
            src_doc = ""
            for idx, _ in top_candidates:
                if kb_index.chunk_texts[idx] == text:
                    src_doc = kb_index.chunk_metas[idx].get("filename", "")
                    break
            # min-max 归一化到完整候选池，让百分数体现"相对于所有候选的位置"
            confidence_pct = round(((score - pool_min) / pool_range) * 100, 1) if pool_range > 0 else 100.0
            final_evidences.append({
                "evidence": text[:500],
                "score": score,
                "confidence_pct": confidence_pct,
                "source_document": src_doc,
            })

        formatted = "\n\n".join(lines)

        return {
            "retrieved_docs": formatted,
            "kb_confidence": max_score,
            "final_evidences": final_evidences,
        }

    return retrieve_kb


def make_synthesize_answer_node(llm=None):
    if llm is None:
        llm = create_llm(temperature=0.2)

    async def synthesize_answer(state: AgentState) -> dict:
        context_block = _build_context_block(state)
        history = format_history(state.get("messages", []))
        logger.info("synthesize_answer intent=%s review_count=%d", state.get("intent"), state.get("review_count", 0))

        if state.get("streaming", False):
            # ── 流式模式：纯文本 + astream ──
            prompt = STREAM_SYNTHESIZE_PROMPT.format(
                context_block=context_block or "无可用上下文",
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            full = await _stream_and_collect(llm, prompt)
            return {
                "final_answer": full,
                "review_count": state.get("review_count", 0) + 1,
                "review_feedback": "",
                "messages": [AIMessage(content=full)],
            }
        else:
            # ── 批处理模式：JSON + ainvoke ──
            prompt = SYNTHESIZE_PROMPT.format(
                context_block=context_block or "无可用上下文",
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            response = await llm.ainvoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            parsed = parse_json_response(raw, state["question"])

            final_evidences = state.get("final_evidences", [])
            llm_evidences = []
            for src in parsed.get("data_sources", []):
                llm_evidences.extend(src.get("evidences", []))

            result = {
                "final_answer": json.dumps(parsed, ensure_ascii=False),
                "review_count": state.get("review_count", 0) + 1,
                "review_feedback": "",
                "messages": [AIMessage(content=parsed.get("answer", ""))],
            }
            if llm_evidences:
                result["final_evidences"] = llm_evidences
            return result

    return synthesize_answer


def make_self_review_node(llm=None):
    if llm is None:
        llm = create_llm(temperature=0.0)

    async def self_review(state: AgentState) -> dict:
        try:
            answer_data = json.loads(state["final_answer"])
            answer_text = answer_data.get("answer", "")
        except (json.JSONDecodeError, KeyError):
            answer_text = state["final_answer"]

        context = state.get("retrieved_docs", "") or state.get("web_results", "")
        prompt = REVIEW_PROMPT.format(
            question=state["question"],
            answer=answer_text[:2000],
            context=context[:2000] if context else "无",
        )

        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        try:
            extracted = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
            score = int(extracted.get("score", 3))
            issues = extracted.get("issues", "")
        except (ValueError, json.JSONDecodeError):
            score = 3
            issues = ""

        logger.info("self_review score=%d/5 issues=%.100s", score, issues or "none")

        if score >= 3:
            return {"review_feedback": "", "review_count": state.get("review_count", 0)}

        return {
            "review_feedback": issues or f"质量评分{score}/5，需要改进",
            "review_count": state.get("review_count", 0),
        }

    return self_review


# ── Routing ──

def route_by_intent(state: AgentState) -> Literal["ask_clarification", "chat_respond", "retrieve_kb", "tavily_search"]:
    intent = state.get("intent", "kb")
    if intent == "clarify":
        return "ask_clarification"
    if intent == "chat":
        return "chat_respond"
    if intent == "web":
        return "tavily_search"
    return "retrieve_kb"


def route_after_review(state: AgentState) -> Literal["synthesize_answer", END]:
    # 流式模式跳過自審循环
    if state.get("streaming", False):
        return END
    if state.get("review_feedback") and state.get("review_count", 0) < MAX_REVIEW_RETRY:
        return "synthesize_answer"
    return END


# ── Graph builder ──

def build_rag_graph(kb_index: KnowledgeBaseIndex, llm=None):
    if llm is None:
        llm = create_llm(temperature=0.1)

    classify_intent = make_classify_intent_node(kb_index, llm)
    retrieve_kb = make_retrieve_kb_node(kb_index)
    synthesize_answer = make_synthesize_answer_node(llm)
    self_review = make_self_review_node(llm)

    async def tavily_search(state: AgentState) -> dict:
        try:
            from langchain_community.tools.tavily_search import TavilySearch
            tool = TavilySearch(max_results=5, topic="general")
            question = state.get("rewritten_query") or state["question"]
            result = tool.invoke(question)
            text = str(result)
        except Exception:
            text = "Web search unavailable."
        return {"web_results": text}

    async def chat_respond(state: AgentState) -> dict:
        llm_local = create_llm(temperature=0.3)
        history = format_history(state.get("messages", []))

        if state.get("streaming", False):
            prompt = STREAM_CHAT_PROMPT.format(
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            full = await _stream_and_collect(llm_local, prompt)
            return {
                "final_answer": full,
                "review_count": 0,
                "review_feedback": "",
                "messages": [AIMessage(content=full)],
            }
        else:
            prompt = CHAT_PROMPT.format(
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            response = await llm_local.ainvoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            parsed = parse_json_response(raw, state["question"])
            return {
                "final_answer": json.dumps(parsed, ensure_ascii=False),
                "review_count": 0,
                "review_feedback": "",
                "messages": [AIMessage(content=parsed.get("answer", ""))],
            }

    async def ask_clarification(state: AgentState) -> dict:
        llm_local = create_llm(temperature=0.3)
        history = format_history(state.get("messages", []))

        if state.get("streaming", False):
            prompt = STREAM_CLARIFY_PROMPT.format(
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            full = await _stream_and_collect(llm_local, prompt)
            return {
                "final_answer": full,
                "review_count": 0,
                "review_feedback": "",
                "messages": [AIMessage(content=full)],
            }
        else:
            prompt = CLARIFY_PROMPT.format(
                current_time=state["current_time"],
                history=history or "无",
                question=state["question"],
            )
            response = await llm_local.ainvoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            parsed = parse_json_response(raw, state["question"])
            answer_text = parsed.get("answer", raw)[:500]
            fallback_json = json.dumps({
                "answer": answer_text,
                "rewritten_query": "",
                "data_sources": [{"source_type": "clarification", "evidences": [{"evidence": state["question"][:200], "score": 1.0}]}],
            }, ensure_ascii=False)
            return {
                "final_answer": fallback_json,
                "review_count": 0,
                "review_feedback": "",
                "messages": [AIMessage(content=answer_text)],
            }

    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("chat_respond", chat_respond)
    builder.add_node("retrieve_kb", retrieve_kb)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("self_review", self_review)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)
    builder.add_edge("retrieve_kb", "synthesize_answer")
    builder.add_edge("tavily_search", "synthesize_answer")
    builder.add_edge("synthesize_answer", "self_review")
    builder.add_edge("ask_clarification", "self_review")
    builder.add_conditional_edges("self_review", route_after_review)
    builder.add_edge("chat_respond", END)

    return builder.compile(checkpointer=MemorySaver())
