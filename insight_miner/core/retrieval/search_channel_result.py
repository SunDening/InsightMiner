"""SearchChannelResult — output of a single channel."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchChannelResult:
    """Result from one search channel."""

    channel_name: str
    chunks: list[tuple[int, float]]  # [(chunk_index, score), …]
    metadata: dict = field(default_factory=dict)
