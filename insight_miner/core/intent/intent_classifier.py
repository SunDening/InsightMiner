"""LLM-based intent classifier.

Sends all leaf nodes to the LLM in one call and asks it to score each.
Returns a list of {"id": str, "score": float, "reason": str} entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from insight_miner.core.intent.intent_tree import IntentTree
from insight_miner.core.llm_factory import create_llm
from insight_miner.utils.helpers import parse_json_response

logger = logging.getLogger(__name__)

INTENT_CLASSIFIER_PROMPT = """你是一个智能问答系统的意图分类引擎。请根据用户问题和可选对话历史完成两项任务：

任务一：如果问题中包含代词（"它们""这些""其""他"等），模糊时间词（"最近""今天"），或过于口语化，请将其改写成更适合检索的表达。
任务二：从以下意图分类中选择最匹配的一个或多个，并给出评分。

每个意图分类包含以下信息：
- id: 唯一标识
- path: 分类路径
- description: 分类说明
- examples: 示例问题

请按以下 JSON 格式输出：
{{
  "rewritten_query": "改写后的查询（如无需改写则保持原问题）",
  "entities": ["实体1", "实体2"],
  "intents": [
    {{"id": "分类id", "score": 0.95, "reason": "匹配原因"}},
    {{"id": "分类id", "score": 0.5, "reason": "匹配原因"}}
  ]
}}

评分规则：
- 1.0: 完全匹配，用户问题与示例/描述高度一致
- 0.7-0.9: 高度匹配，语义相近
- 0.4-0.6: 部分匹配，可能相关
- 0.1-0.3: 弱匹配，仅有一些关联
- 0.0: 完全不匹配

如果问题明显不属于任何已知分类，请将最高分分类设为 "clarify" 并给出低分。

可用分类：
{intent_list}

当前时间：{current_time}
对话历史：
{history}

用户问题：{question}"""


class LLMIntentClassifier:
    """Classifies user questions against an IntentTree using an LLM."""

    def __init__(self, tree: IntentTree, llm=None):
        self._tree = tree
        self._llm = llm or create_llm(temperature=0.1)

    async def classify(
        self,
        question: str,
        history: str = "",
        kb_description: str = "",
    ) -> dict:
        """Returns { "rewritten_query": str, "entities": list[str], "intents": list[dict] }."""
        prompt = self._build_prompt(question, history, kb_description)
        logger.info("classifying intent via tree… question=%.50s", question)
        response = await self._llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(raw)

    def _build_prompt(
        self,
        question: str,
        history: str,
        kb_description: str,
    ) -> str:
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build leaf node listing
        lines: list[str] = []
        for node in self._tree.leaf_nodes:
            lines.append(f"- id={node.id}")
            lines.append(f"  path={node.full_path or node.name}")
            lines.append(f"  description={node.description or '无'}")
            if node.examples:
                lines.append(f"  examples={' / '.join(node.examples)}")
            lines.append("")
        intent_list = "\n".join(lines)

        return INTENT_CLASSIFIER_PROMPT.format(
            intent_list=intent_list,
            current_time=current_time,
            history=history or "无",
            question=question,
        )

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM response into {rewritten_query, entities, intents}."""
        result = {
            "rewritten_query": "",
            "entities": [],
            "intents": [],
        }
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end <= start:
                return result

            data = json.loads(raw[start : end + 1])

            # Extract rewritten_query
            rewritten = data.get("rewritten_query", "")
            result["rewritten_query"] = rewritten if isinstance(rewritten, str) else str(rewritten)

            # Extract entities
            entities = data.get("entities", [])
            if isinstance(entities, list):
                result["entities"] = [str(e) for e in entities if isinstance(e, str)]

            # Extract intents
            intents = data.get("intents", data.get("results", data.get("scores", [])))
            if isinstance(intents, list):
                result["intents"] = intents
            elif isinstance(data, list):
                result["intents"] = data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("parse_response failed: %s raw=%.100s", e, raw)

        return result
