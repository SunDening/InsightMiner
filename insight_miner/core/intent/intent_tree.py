"""IntentTree — configurable tree of intent nodes.

Supports loading from Python dict or YAML file, so the intent hierarchy
can be tuned without code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from insight_miner.core.intent.intent_node import IntentNode


class IntentTree:
    """A tree of IntentNodes that can be flattened and queried."""

    def __init__(self, roots: list[IntentNode]) -> None:
        self._roots = roots
        self._all_nodes: dict[str, IntentNode] = {}
        self._leaf_nodes: list[IntentNode] = []
        self._flatten(roots)

    # ── Public API ──

    @property
    def roots(self) -> list[IntentNode]:
        return list(self._roots)

    @property
    def leaf_nodes(self) -> list[IntentNode]:
        return list(self._leaf_nodes)

    @property
    def all_nodes(self) -> dict[str, IntentNode]:
        return dict(self._all_nodes)

    def get_node(self, node_id: str) -> IntentNode | None:
        return self._all_nodes.get(node_id)

    # ── Factory methods ──

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntentTree:
        """Build tree from a nested dict structure.

        Expected format (YAML-equivalent):
            intent_tree:
              sys:
                kind: system
                children:
                  sys-greet:
                    kind: chat
                    description: "..."
                    examples: ["你好"]
              kb-general:
                kind: kb
                description: "..."
        """
        raw_tree = data.get("intent_tree", data)
        roots = _build_nodes(raw_tree, parent_id=None)
        _fill_full_path(roots, prefix="")
        return cls(roots)

    @classmethod
    def from_yaml(cls, path: str | Path) -> IntentTree:
        """Load intent tree from a YAML config file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    # ── Internal ──

    def _flatten(self, nodes: list[IntentNode]) -> None:
        for node in nodes:
            self._all_nodes[node.id] = node
            if node.is_leaf():
                self._leaf_nodes.append(node)
            if node.children:
                self._flatten(node.children)


# ── Tree-building helpers ──


def _build_nodes(
    raw: dict[str, Any],
    parent_id: str | None = None,
) -> list[IntentNode]:
    """Recursively build IntentNode list from a raw dict."""
    nodes: list[IntentNode] = []
    for node_id, cfg in raw.items():
        children_raw = cfg.pop("children", None) if isinstance(cfg, dict) else None
        if isinstance(cfg, str):
            # Shorthand: "kb" or "chat"
            cfg = {"kind": cfg}
        if not isinstance(cfg, dict):
            continue

        node = IntentNode(
            id=node_id,
            name=cfg.get("name", node_id),
            description=cfg.get("description", ""),
            kind=cfg.get("kind", "kb"),
            parent_id=parent_id,
            examples=cfg.get("examples"),
            prompt_template=cfg.get("prompt_template"),
        )
        if children_raw:
            node.children = _build_nodes(children_raw, parent_id=node_id)
        nodes.append(node)
    return nodes


def _fill_full_path(nodes: list[IntentNode], prefix: str) -> None:
    for node in nodes:
        node.full_path = f"{prefix} > {node.name}" if prefix else node.name
        if node.children:
            _fill_full_path(node.children, node.full_path)
