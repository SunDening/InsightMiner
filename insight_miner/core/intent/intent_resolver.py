"""IntentResolver — converts raw LLM scores into actionable routing decisions."""

from __future__ import annotations

from dataclasses import dataclass

from insight_miner.core.intent.intent_node import IntentNode
from insight_miner.core.intent.intent_tree import IntentTree


@dataclass
class NodeScore:
    """A scored intent node from classification."""

    node: IntentNode
    score: float
    reason: str = ""


class IntentResolver:
    """Resolves scored intents into routing decisions.

    Takes the raw list of {"id", "score", "reason"} from the classifier,
    filters by threshold, sorts, and maps back to the simple "chat|kb|clarify"
    intent kind used by the LangGraph router.
    """

    def __init__(
        self,
        tree: IntentTree,
        min_score: float = 0.3,
        top_n: int = 3,
    ):
        self._tree = tree
        self._min_score = min_score
        self._top_n = top_n

    def resolve(self, scores: list[dict]) -> list[NodeScore]:
        """Filter, sort, and convert raw scores to NodeScore list.

        Accepts both the full classifier result (dict with "intents" key)
        and a plain list for backward compatibility.
        """
        if isinstance(scores, dict):
            scores = scores.get("intents", scores.get("results", []))

        result: list[NodeScore] = []
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            node_id = entry.get("id", "")
            score = float(entry.get("score", 0.0))
            reason = entry.get("reason", "")

            node = self._tree.get_node(node_id)
            if node is None:
                continue
            if score < self._min_score:
                continue
            result.append(NodeScore(node=node, score=score, reason=reason))

        # Sort by score descending
        result.sort(key=lambda ns: -ns.score)
        return result[: self._top_n]

    def resolve_intent_kind(self, scores: list[dict] | dict) -> str:
        """Convenience: get the primary intent kind for routing."""
        if isinstance(scores, dict):
            scores = scores.get("intents", [])
        resolved = self.resolve(scores)
        if not resolved:
            return "kb"

        # Check for clarify first (low confidence)
        top = resolved[0]
        if top.score < 0.3 and top.node.is_clarify():
            return "clarify"
        if top.score < 0.3:
            return "clarify"

        # Return the kind of the top-scoring node
        kind = top.node.kind
        if kind in ("chat", "clarify"):
            return kind
        return "kb"

    def resolve_leaf_id(self, scores: list[dict]) -> str:
        """Get the leaf node ID of the top intent."""
        resolved = self.resolve(scores)
        if not resolved:
            return ""
        return resolved[0].node.id

    def resolve_rewrite_query(self, scores: list[dict] | dict, original: str) -> str:
        """Get the rewritten query from the classifier result."""
        if isinstance(scores, dict):
            rq = scores.get("rewritten_query", "")
            if rq:
                return rq
        return original

    def resolve_entities(self, scores: list[dict] | dict) -> list[str]:
        """Extract entities from the classifier result."""
        if isinstance(scores, dict):
            entities = scores.get("entities", [])
            if isinstance(entities, list):
                return [str(e) for e in entities if isinstance(e, str)]
        return []
