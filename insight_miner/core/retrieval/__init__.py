"""Multi-channel retrieval engine — pluggable search channels with post-processing."""

from insight_miner.core.retrieval.search_channel import SearchChannel
from insight_miner.core.retrieval.search_context import SearchContext
from insight_miner.core.retrieval.search_channel_result import SearchChannelResult
from insight_miner.core.retrieval.multi_channel_engine import MultiChannelEngine

__all__ = [
    "SearchChannel",
    "SearchContext",
    "SearchChannelResult",
    "MultiChannelEngine",
]
