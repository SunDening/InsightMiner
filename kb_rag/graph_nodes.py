"""LangGraph 节点函数、状态定义与 Prompt 模板 — 仅 KB / Chat / Web / Clarify 路径。"""

import json
from typing import Annotated, TypedDict

from langchain.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from .config import logger, LLM_PROVIDER, MAX_REVIEW_RETRY
from .llm import create_llm
from .utils import (
    _extract_json, _parse_json_response, _format_history,
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
    intent:          str          # chat | kb | web | clarify
    current_time:    str
    # KB / Web 路径
    retrieved_docs:  str
    kb_confidence:   float
    web_results:     str
    # 图检索
    query_entities:  list[str]
    # 统一输出
    final_answer:    str
    review_count:    int
    review_feedback: str


# ════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ════════════════════════════════════════════════════════════════════════

INTENT_CLASSIFY_PROMPT = """你是一个智能助手。请同时完成意图分类和查询重写两项任务。

如果用户问题包含代词（"它们""这些""其""他"等），请参考对话历史确定指代对象。

意图分类优先级：
- **web**: 用户明确要求上网搜索（消息中包含"上网查""搜索网络""在网上找"等短语）
- **clarify**: 用户问题过于模糊或存在歧义，且对话历史中也无法确定指代对象
- **kb**（默认）: 除以上三类外的所有问题，包括概念解释、专业知识、历史背景等
- **chat**: 仅限问候寒暄（"你好""早上好"）和助手自我介绍

查询重写规则（仅 kb/web 需要重写，chat/clarify 保持原文）：
- 模糊时间词 → 结合当前时间替换为具体日期
- 代词 → 根据对话历史确定具体实体
- 口语化表达 → 检索友好的措辞
- KB 意图提取核心关键词；Web 意图优化搜索词

## 对话历史
{history}

## 当前时间
{current_time}

知识库文档：{kb_description}

## 用户问题
{question}

## 输出格式
只输出一个 JSON 对象（不要 Markdown 代码块）：
{{"intent": "kb", "rewritten_query": "重写后的查询文本（chat/clarify 时为原文）", "entities": ["关键实体1", "关键实体2"]}}

## 实体提取要求（仅 kb 和 web 意图）
提取用户问题中的**关键实体**（专有名词、人名、地名、技术术语、产品名等），
用于知识库图检索。chat/clarify 意图请输出空数组 []。"""


CLARIFY_PROMPT = """你是一个智能助手。用户的问题过于模糊，你无法确定该如何回答。

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


CHAT_PROMPT = """你是一个智能助手，知识面广泛。

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
      "evidences": [
        {{"evidence": "基于预训练知识和逻辑推理", "score": 1.0}}
      ]
    }}
  ]
}}

注意：
- 禁止使用任何 Markdown 格式
- 只输出纯文本和中文/英文标点符号
- 如果用户问题涉及时间（如"今天""最近"），请结合上方提供的当前时间作答。"""


SYNTHESIZE_PROMPT = """你是一个智能数据分析助手。

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
      "evidences": [
        {{"evidence": "原始文本摘录，不超过200字", "score": 0.95}},
        {{"evidence": "另一个证据片段", "score": 0.87}}
      ]
    }}
  ]
}}

注意：
- answer 要直接回应用户的问题
- 每个 evidence 的 score 从上下文文档片段的 (score: X.XXX) 标记中提取
- 如果上下文没有给出分数，为每条 evidence 使用 1.0
- 禁止使用任何 Markdown 格式
- data_sources 数组可以包含多个来源
- 如果查询结果为空，在 answer 中诚实告知"""


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
        entities: list[str] = []
        try:
            parsed = json.loads(_extract_json(raw))
            intent = parsed.get("intent", "chat")
            rewritten_query = parsed.get("rewritten_query", question)
            raw_entities = parsed.get("entities", [])
            if isinstance(raw_entities, list):
                entities = [str(e) for e in raw_entities if e]
        except json.JSONDecodeError:
            raw_lower = raw.lower()
            for candidate in ["clarify", "chat", "kb", "web"]:
                if candidate in raw_lower:
                    intent = candidate
                    break

        logger.info(f"[Intent] {question[:50]}... → intent={intent}, entities={entities}")
        return {"intent": intent, "rewritten_query": rewritten_query, "query_entities": entities}
    return classify_intent_node


def make_retrieve_kb_node(kb: RagKnowledgeBase):
    """从知识库检索相关文档（三路混合检索）。"""
    async def retrieve_kb_node(state: AgentState) -> dict:
        question = state.get("rewritten_query") or state["question"]
        entities = state.get("query_entities", [])
        docs, confidence = await kb.retrieve(question, query_entities=entities)
        logger.info(f"[KB] 检索完成，置信度 {confidence:.2f}")
        return {"retrieved_docs": docs, "kb_confidence": confidence}
    return retrieve_kb_node


def make_tavily_search_node(kb: RagKnowledgeBase):
    """使用 Tavily 搜索网络，拆分结果后 CrossEncoder 精排 Top-3。"""
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
            "evidences": [{"evidence": f"原始问题: {question}", "score": 1.0}],
        }],
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

    prompt = CHAT_PROMPT.format(
        current_time=current_time, history=history, question=question
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    final_answer = _parse_json_response(response.content, question)

    logger.info(f"[Chat] 生成回答，长度 {len(final_answer)} 字符")
    return {
        "final_answer": final_answer,
        "review_count": 0,
        "review_feedback": "",
        "messages": [AIMessage(content=final_answer)],
    }


async def synthesize_answer_node(state: AgentState) -> dict:
    """统一合成最终 JSON 格式回答。"""
    llm = create_llm(LLM_PROVIDER, temperature=0.2)

    question = state["question"]
    intent = state.get("intent", "chat")
    history = _format_history(state.get("messages", []))
    review_feedback = state.get("review_feedback", "")
    current_time = state.get("current_time", "未知")

    if intent == "kb":
        docs = state.get("retrieved_docs", "")
        confidence = state.get("kb_confidence", 0.0)

        if confidence < -2.0:
            confidence_note = (
                f"⚠️ 检索置信度极低（{confidence:.2f}），知识库中未找到与用户问题直接相关的内容。\n"
                f"请在 answer 中明确告知用户，参考句式：\n"
                f"'知识库中未找到与该问题直接相关的内容。以下回答基于 LLM 自身知识生成：...'\n"
                f"data_sources 的 source_type 请设为 'llm_knowledge'。\n\n"
            )
            source_type = "llm_knowledge"
        else:
            confidence_note = ""
            source_type = "knowledge_base"

        context_block = (
            f"{confidence_note}"
            f"## 知识库检索结果\n{docs}"
        )

    elif intent == "web":
        web_results = state.get("web_results", "")
        context_block = f"## Web 搜索结果\n{web_results}"
        source_type = "web_search"

    else:
        context_block = ""
        source_type = "llm_knowledge"

    if review_feedback:
        context_block += (
            f"\n\n## ⚠️ 上一版回答的问题\n{review_feedback}\n请改进后重新输出。"
        )

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

    logger.info(f"[Synthesize] 生成回答，长度 {len(final_answer)} 字符")
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
            "review_feedback": (
                f"审核未通过（分数 {score}/5）。问题: {issues}。请修正后重新输出。"
            ),
        }
