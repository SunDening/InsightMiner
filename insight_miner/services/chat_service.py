"""Chat orchestration — wraps the RAG engine with memory and threading."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage

from insight_miner.core.cache import AnswerCache
from insight_miner.core.rag_engine import build_rag_graph
from insight_miner.models.schemas import ChatResponse, EvidenceItem
from insight_miner.services.kb_manager import KnowledgeBaseManager
from insight_miner.services.memory_service import MemoryService


def _score_to_pct(score: float) -> float:
    normalized = (score + 5.0) / 10.0
    clipped = max(0.0, min(1.0, normalized))
    # sqrt 拉伸让中间段的百分数更有区分度
    return round(math.sqrt(clipped) * 100.0, 1)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatService:
    def __init__(self, kb_manager: KnowledgeBaseManager, memory: MemoryService):
        self._kb_manager = kb_manager
        self._memory = memory
        self._graphs: dict[str, object] = {}
        self._cache = AnswerCache()

    async def _get_or_create_graph(self, kb_id: str):
        if kb_id not in self._graphs:
            idx = await self._kb_manager.get_index(kb_id)
            graph = build_rag_graph(idx)
            self._graphs[kb_id] = graph
        return self._graphs[kb_id]

    async def _ensure_thread(self, thread_id: str | None, kb_id: str) -> str:
        if not thread_id:
            thread_id = uuid.uuid4().hex[:12]
        created = await self._memory.create_thread(thread_id, kb_id)
        if not created:
            existing_kb = await self._memory.get_thread_kb_id(thread_id)
            if existing_kb != kb_id:
                thread_id = uuid.uuid4().hex[:12]
                await self._memory.create_thread(thread_id, kb_id)
        return thread_id

    async def _build_lc_history(self, thread_id: str):
        msgs = await self._memory.get_history(thread_id, limit=20)
        lc = []
        for m in msgs:
            if m["role"] == "user":
                from langchain_core.messages import HumanMessage
                lc.append(HumanMessage(content=m["content"]))
            else:
                from langchain_core.messages import AIMessage
                lc.append(AIMessage(content=m["content"]))
        return lc

    # ── Non-streaming chat (LangGraph) ──

    async def chat(
        self,
        question: str,
        thread_id: str | None = None,
        kb_id: str = "default",
    ) -> ChatResponse:
        thread_id = await self._ensure_thread(thread_id, kb_id)
        await self._memory.save_message(thread_id, "user", question)
        logger.info("chat thread=%s kb=%s question=%.50s", thread_id, kb_id, question)

        # Cache lookup
        cached = await self._cache.get(kb_id, question)
        if cached is not None:
            logger.info("cache hit thread=%s kb=%s", thread_id, kb_id)
            answer_text = cached.get("answer", "")
            rewritten_query = cached.get("rewritten_query", "")
            raw_evidences = cached.get("evidences", [])
            intent = cached.get("intent", "kb")
            kb_confidence = cached.get("kb_confidence", 0.0)
            evidences = self._format_evidences(raw_evidences, kb_confidence)
            await self._memory.save_message(thread_id, "assistant", answer_text)
            return ChatResponse(
                answer=answer_text,
                thread_id=thread_id,
                rewritten_query=rewritten_query,
                evidences=evidences[:5],
                kb_confidence=kb_confidence,
                intent=intent,
            )

        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        graph = await self._get_or_create_graph(kb_id)

        input_state = {
            "question": question,
            "current_time": current_time,
            "intent": "",
            "rewritten_query": "",
            "retrieved_docs": "",
            "kb_confidence": 0.0,
            "query_entities": [],
            "final_answer": "",
            "review_count": 0,
            "review_feedback": "",
            "messages": [HumanMessage(content=question)],
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(input_state, config)

        final_answer_str = result.get("final_answer", "{}")
        try:
            answer_data = json.loads(final_answer_str)
            answer_text = answer_data.get("answer", final_answer_str)
            rewritten_query = answer_data.get("rewritten_query", "")
        except (json.JSONDecodeError, TypeError):
            answer_text = final_answer_str
            rewritten_query = ""

        raw_evidences = result.get("final_evidences", [])
        evidences: list[EvidenceItem] = []
        kb_confidence = result.get("kb_confidence", 0.0)

        if raw_evidences:
            for ev in raw_evidences:
                score = float(ev.get("score", 0.0))
                content = ev.get("evidence", ev.get("content", ""))
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                source = ev.get("source_document", ev.get("source", ""))
                # 优先使用 retrieve_kb 节点预计算的 min-max 置信度
                confidence_pct = float(ev.get("confidence_pct", _score_to_pct(score)))
                evidences.append(EvidenceItem(
                    content=str(content)[:1000],
                    score=score,
                    confidence_pct=confidence_pct,
                    source_document=str(source),
                    chunk_index=ev.get("chunk_index", 0),
                ))
        else:
            doc_lines = result.get("retrieved_docs", "")
            if doc_lines:
                for block in doc_lines.split("\n\n--- "):
                    if "score:" in block and "---" in block:
                        try:
                            score_line = block.split("---")[0].strip()
                            score = float(score_line.split("score:")[1].split(")")[0])
                            text = "---".join(block.split("---")[1:]).strip()
                            evidences.append(EvidenceItem(
                                content=text[:500],
                                score=score,
                                confidence_pct=_score_to_pct(score),
                                source_document="",
                            ))
                        except (IndexError, ValueError):
                            pass

        await self._memory.save_message(thread_id, "assistant", answer_text)

        # Cache the result for future requests
        cache_data = {
            "answer": answer_text,
            "rewritten_query": rewritten_query,
            "evidences": [ev.model_dump() for ev in evidences[:5]],
            "kb_confidence": kb_confidence,
            "intent": result.get("intent", "kb"),
        }
        asyncio.ensure_future(self._cache.set(kb_id, question, cache_data))
        logger.info("chat cached thread=%s kb=%s", thread_id, kb_id)

        return ChatResponse(
            answer=answer_text,
            thread_id=thread_id,
            rewritten_query=rewritten_query,
            evidences=evidences[:5],
            kb_confidence=kb_confidence,
            intent=result.get("intent", "kb"),
        )

    # ── Format evidences from raw data ──

    def _format_evidences(
        self,
        raw_evidences: list[dict],
        kb_confidence: float,
    ) -> list[EvidenceItem]:
        evidences: list[EvidenceItem] = []
        for ev in raw_evidences:
            score = float(ev.get("score", 0.0))
            content = ev.get("evidence", ev.get("content", ""))
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            source = ev.get("source_document", ev.get("source", ""))
            confidence_pct = float(ev.get("confidence_pct", _score_to_pct(score)))
            evidences.append(EvidenceItem(
                content=str(content)[:1000],
                score=score,
                confidence_pct=confidence_pct,
                source_document=str(source),
                chunk_index=ev.get("chunk_index", 0),
            ))
        return evidences

    # ── Streaming chat (SSE via LangGraph astream_events) ──

    async def stream_chat(
        self,
        question: str,
        thread_id: str | None = None,
        kb_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        thread_id = await self._ensure_thread(thread_id, kb_id)
        await self._memory.save_message(thread_id, "user", question)
        logger.info("stream_chat thread=%s kb=%s question=%.50s", thread_id, kb_id, question)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        graph = await self._get_or_create_graph(kb_id)

        # 1. thread_id
        yield _sse("thread_id", {"thread_id": thread_id})

        # Build conversation history
        lc_msgs = await self._build_lc_history(thread_id)

        # 2. Run graph with astream_events
        input_state = {
            "question": question,
            "current_time": current_time,
            "intent": "",
            "rewritten_query": "",
            "retrieved_docs": "",
            "kb_confidence": 0.0,
            "web_results": "",
            "query_entities": [],
            "final_answer": "",
            "review_count": 0,
            "review_feedback": "",
            "streaming": True,
            "messages": lc_msgs + [HumanMessage(content=question)],
        }

        config = {"configurable": {"thread_id": thread_id}}

        full_answer = ""
        error_occurred = False

        try:
            async for event in graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                # For on_chain_start/end, name is the node name.
                # For on_chat_model_stream, metadata.langgraph_node indicates the source node.
                source = event.get("name", "") or event.get("metadata", {}).get("langgraph_node", "")

                # Node boundaries → status SSE
                if kind == "on_chain_start":
                    if source == "classify_intent":
                        yield _sse("status", {"phase": "classifying", "message": "正在理解问题..."})
                    elif source == "retrieve_kb":
                        yield _sse("status", {"phase": "retrieving", "message": "正在检索知识库..."})
                    elif source in ("synthesize_answer", "chat_respond", "ask_clarification"):
                        yield _sse("status", {"phase": "generating", "message": "正在生成回答..."})

                # Node outputs → intent / evidence SSE
                elif kind == "on_chain_end":
                    if source == "classify_intent":
                        output = event["data"].get("output", {}) if isinstance(event.get("data"), dict) else {}
                        if isinstance(output, dict) and "intent" in output:
                            yield _sse("intent", {"intent": output["intent"]})
                    elif source == "retrieve_kb":
                        output = event["data"].get("output", {}) if isinstance(event.get("data"), dict) else {}
                        raw_evidences = output.get("final_evidences", []) if isinstance(output, dict) else []
                        if raw_evidences:
                            formatted = []
                            for ev in raw_evidences:
                                score = float(ev.get("score", 0.0))
                                content = ev.get("evidence", ev.get("content", ""))
                                if isinstance(content, list):
                                    content = " ".join(str(c) for c in content)
                                formatted.append({
                                    "content": str(content)[:1000],
                                    "score": score,
                                    "confidence_pct": float(ev.get("confidence_pct", _score_to_pct(score))),
                                    "source_document": str(ev.get("source_document", "")),
                                })
                            yield _sse("evidence", {"evidences": formatted})

                # Answer tokens → token SSE
                elif kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "") or source
                    if node in ("synthesize_answer", "chat_respond", "ask_clarification"):
                        chunk = event["data"]["chunk"]
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if token:
                            full_answer += token
                            yield _sse("token", {"token": token})

        except Exception as e:
            logger.error("stream_chat error thread=%s: %s", thread_id, e)
            yield _sse("error", {"message": str(e) or "生成回答时出错"})
            error_occurred = True

        if not error_occurred:
            yield _sse("done", {"rewritten_query": ""})
            if full_answer:
                await self._memory.save_message(thread_id, "assistant", full_answer)

    # ── Mind Map ──

    async def generate_mindmap(self, thread_id: str) -> dict:
        """Use LLM to turn the last assistant answer into a mind-map tree JSON."""
        messages = await self._memory.get_history(thread_id)
        logger.info("generate_mindmap thread=%s total_messages=%d", thread_id, len(messages))
        if messages:
            logger.info("generate_mindmap roles=%s", [m["role"] for m in messages])
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs:
            roles_found = {m["role"] for m in messages} if messages else "none"
            raise ValueError(
                f"No assistant messages found in thread {thread_id} "
                f"(total={len(messages)}, roles={roles_found})"
            )

        last_answer = assistant_msgs[-1]["content"]

        from insight_miner.core.llm_factory import create_llm

        logger.info("generate_mindmap step=creating_llm last_answer_len=%d", len(last_answer))
        llm = create_llm(temperature=0.3)

        logger.info("generate_mindmap step=invoking_llm")
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="""你是一个思维导图专家。将用户的回答提炼为层级清晰的思维导图 JSON。

仅输出合法的 JSON，不要包含说明文字或 markdown 代码块：

{
  "root": {
    "content": "主题（≤10字）",
    "children": [
      {
        "content": "分支",
        "children": [
          { "content": "子节点" }
        ]
      }
    ]
  }
}

规则：
1. 根节点概括核心主题（≤10字）
2. 第一层分支 3-7 个要点（≤10字）
3. 最大嵌套 3 层
4. 所有节点用精炼短语，不要用句子
5. 输出语言：回答是中文就用中文，是英文就用英文
6. 仅输出 JSON，不要其他文字"""),
            HumanMessage(content=f"回答内容：\n\n{last_answer}\n\n请生成思维导图 JSON："),
        ]
        result = await llm.ainvoke(messages)

        logger.info("generate_mindmap step=processing_result")
        raw = result.content.strip()
        # Strip possible markdown code fences
        if raw.startswith("```"):
            first_nl = raw.find("\n")
            if first_nl != -1:
                raw = raw[first_nl + 1:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        import json
        from insight_miner.utils.helpers import extract_json, repair_json

        logger.info("generate_mindmap raw_llm_output=%.200s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            extracted = extract_json(raw)
            if extracted:
                try:
                    data = json.loads(extracted)
                except json.JSONDecodeError:
                    repaired = repair_json(extracted)
                    if repaired:
                        data = json.loads(repaired)
                    else:
                        raise
            else:
                repaired = repair_json(raw)
                if repaired:
                    data = json.loads(repaired)
                else:
                    raise

        # markmap 要求每个节点必须有 children（空数组也⾏，不能是 None）
        def _fill_children(n: dict):
            if n.get("children") is None:
                n["children"] = []
            else:
                for c in n["children"]:
                    _fill_children(c)

        _fill_children(data.get("root", data))
        return data

    # ── History ──

    async def get_history(self, thread_id: str) -> list[dict]:
        return await self._memory.get_history(thread_id)

    async def list_threads(self, kb_id: str | None = None) -> list[dict]:
        return await self._memory.list_threads(kb_id)

    async def delete_thread(self, thread_id: str):
        await self._memory.delete_thread(thread_id)
