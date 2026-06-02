"""SearchContext — data passed through the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchContext:
    """Context shared across all search channels and post-processors."""

    query: str
    entities: list[str] = field(default_factory=list)
    top_k: int = 10
    # Internal: reference to the KB index (set by the engine)
    kb_index: object = None
    # Channel-level top-k overrides
    dense_top_k: int = 20
    bm25_top_k: int = 20
    graph_top_k: int = 40
    # RRF
    rrf_k: int = 60
    rrf_candidates: int = 10
    # Rerank
    rerank_top_k: int = 5
