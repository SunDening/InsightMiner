"""IntentNode — tree node definition for hierarchical intent classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IntentKind = Literal["kb", "chat", "clarify", "system"]


@dataclass
class IntentNode:
    """A node in the intent classification tree.

    Attributes:
        id: Unique identifier (e.g. "sys-greet", "kb-general").
        name: Human-readable display name.
        description: Semantic description used in LLM prompts.
        kind: Node type — kb, chat, clarify, or system.
        parent_id: Parent node ID (None for root nodes).
        children: Child nodes (None for leaf nodes).
        examples: Example questions to help LLM classify.
        prompt_template: Optional custom prompt for this intent.
        full_path: Auto-generated full path (e.g. "系统交互 > 问候").
    """

    id: str
    name: str = ""
    description: str = ""
    kind: IntentKind = "kb"
    parent_id: str | None = None
    children: list[IntentNode] | None = None
    examples: list[str] | None = None
    prompt_template: str | None = None
    full_path: str = ""

    def is_leaf(self) -> bool:
        """Leaf nodes are the ones that participate in intent classification."""
        return not self.children

    def is_kb(self) -> bool:
        return self.kind == "kb"

    def is_chat(self) -> bool:
        return self.kind == "chat"

    def is_clarify(self) -> bool:
        return self.kind == "clarify"

    def is_system(self) -> bool:
        return self.kind == "system"
